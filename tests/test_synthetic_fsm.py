import numpy as np

from evaluation.synthetic_fsm import (
    generate_fsm,
    generate_dataset,
    make_prototypes,
    flatten_true_states,
    oracle_transitions,
)
from evaluation.action_conditioned import conditional_structure


def test_shapes_and_alignment():
    T = generate_fsm(8, 3, seed=0)
    assert T.shape == (8, 3)
    trajs, true = generate_dataset(T, 20, hidden_dim=16, seed=0)
    assert len(trajs) == len(true) == 20
    for tr, s in zip(trajs, true):
        assert tr.states.shape[0] == s.shape[0]          # one embedding per state
        assert tr.op_ids.shape[0] == s.shape[0] - 1       # one action per transition
    assert flatten_true_states(true).shape[0] == sum(t.states.shape[0] for t in trajs)


def test_oracle_is_deterministic():
    T = generate_fsm(12, 3, seed=1)
    trajs, true = generate_dataset(T, 300, hidden_dim=8, seed=1)
    st, at, sn = oracle_transitions(true, trajs)
    # (s, a) -> s' is deterministic by construction: zero conditional entropy,
    # full deterministic mass.
    struct = conditional_structure(list(zip(st.tolist(), at.tolist())), sn.tolist())
    assert struct["global_entropy"] < 1e-9
    # det_frac_mass is conservative (excludes (s,a) pairs seen < min_count), so
    # a few rare pairs at this sample size keep it just under 1.0.
    assert struct["det_frac_mass"] > 0.95
    # State alone is ambiguous (multiple actions -> different successors).
    s_only = conditional_structure(st.tolist(), sn.tolist())
    assert s_only["global_entropy"] > 0.3


def test_structured_embeddings_are_separable():
    T = generate_fsm(10, 3, seed=2)
    protos_recovered_purity = _nearest_prototype_purity(T, structured=True)
    floor_purity = _nearest_prototype_purity(T, structured=False)
    # Structured: a state's embedding is closest to its own prototype mean.
    assert protos_recovered_purity > 0.95
    # Noise floor: embeddings carry no state info -> chance-level purity.
    assert floor_purity < 0.30


def test_shared_prototypes_transfer_across_splits():
    """Train/eval splits must share prototype geometry.

    The prototypes are the environment's fixed ``state -> hidden vector`` map.
    If each split draws its own prototypes, a VQ trained on the train split
    sees an unrelated point cloud at eval time and collapses every eval state
    onto a single code — which silently saturates all downstream
    train->eval predictability metrics at 0/1. With shared prototypes the VQ
    encodes the eval split into the same codes it learned on train.
    """
    from training.train_vq import train_vq_quantizer, VQTrainConfig
    from data_processing.discrete_trajectory_dataset import (
        encode_trajectories_to_codes, all_codes,
    )
    from sklearn.metrics import adjusted_mutual_info_score

    K, H = 8, 64
    T = generate_fsm(K, 3, seed=0)
    protos = make_prototypes(K, hidden_dim=H, seed=7)

    tr, _ = generate_dataset(T, 300, H, noise=0.3, structured=True, seed=0, protos=protos)
    va, ts_va = generate_dataset(T, 150, H, noise=0.3, structured=True, seed=1, protos=protos)
    vq = train_vq_quantizer(tr, va, config=VQTrainConfig(num_codes=K, epochs=12,
                                                         batch_size=256, seed=0))
    shared_codes = all_codes(encode_trajectories_to_codes(vq, va)).numpy()
    assert len(np.unique(shared_codes)) >= K // 2          # eval does NOT collapse
    ami_shared = adjusted_mutual_info_score(flatten_true_states(ts_va), shared_codes)
    assert ami_shared > 0.5                                # codes recover eval states

    # Guard: an independently-seeded eval split (own prototypes) lands on an
    # unrelated point cloud, so its codes no longer track the true states.
    va_indep, ts_indep = generate_dataset(T, 150, H, noise=0.3, structured=True, seed=1)
    indep_codes = all_codes(encode_trajectories_to_codes(vq, va_indep)).numpy()
    ami_indep = adjusted_mutual_info_score(flatten_true_states(ts_indep), indep_codes)
    assert ami_indep < 0.2                                 # the bug this guards against
    assert ami_shared > ami_indep + 0.3                    # sharing is decisively better


def _nearest_prototype_purity(T, structured):
    num_states = T.shape[0]
    trajs, true = generate_dataset(
        T, 400, hidden_dim=32, noise=0.3, structured=structured, seed=3)
    X = np.concatenate([t.states.numpy() for t in trajs], axis=0)
    y = flatten_true_states(true)
    # Empirical per-state means, then nearest-mean classification accuracy.
    means = np.stack([X[y == s].mean(0) if (y == s).any() else np.zeros(X.shape[1])
                      for s in range(num_states)])
    d = ((X[:, None, :] - means[None]) ** 2).sum(-1)
    pred = d.argmin(1)
    return float((pred == y).mean())
