# Hidden-State Extraction Audit

Targets: `scripts/extract_hidden_states.py` (repo) and the trajectory pickles (results). The pickle objects expose: `all_hidden [seq_len, 2048]`, `input_ids [seq_len]`, `state_indices [n_states]`, `op_ids [n_ops]`, `operands [n_ops, 2]`, `numbers`, `target`. Note `n_states = n_ops + 1` (4 states, 3 ops in the inspected sample) — consistent.

## Bugs and risks in the repo extraction script

1. **Off-by-one / layer-indexing ambiguity (`--layer -1`).** `outputs.hidden_states` is a tuple of length `L+1` where index 0 is the embedding layer. `hidden_states[-1]` returns the **final** transformer layer (correct intent), but the CLI default `-1` combined with the comment "1 to L are transformer layers" invites silent off-by-one: a user asking for "layer 1" gets the *embedding-adjacent* first block, and there is no assertion guarding the range. No record of which layer the Phase C tensors came from is stored in the pickle metadata.

2. **State-boundary alignment is unverifiable in the repo.** The repo extractor stores **every token position**. The Phase C pickles instead store `state_indices` (specific token positions designated as "symbolic states"). The code that computes `state_indices` (the alignment from reasoning step → token position) **is not in the repo** (`data_processing/trajectory_dataset.py` missing). This is the single most leakage-prone step and it cannot be audited.

3. **Causal-LM future-information risk.** Hidden state at position `p` in a causal model already integrates tokens `0..p`. If `state_indices[j]` points at the token *after* the `=` of step `j` (i.e., the result token), then the hidden state has already *seen the answer* of that step. A probe reading that vector is not predicting the next operation — it is reading a number it already consumed. Whether the indices point before or after the result token determines whether downstream probes "see the future." **This cannot be checked without the indexing code.**

4. **Prompt formatting bakes the full solution into context.** Both `extract_hidden_states.py` and `train_teacher.py` format the prompt as `Problem: ... Solution:\n{cot}` — i.e., the *entire* chain of thought is in the context window during extraction. Every per-step hidden state is therefore conditioned on the complete solution, not just the prefix. Any "the hidden state encodes the remaining plan" claim is confounded: the remaining plan is literally in the prompt tokens to the right being attended to is prevented by causal masking, but the representation still reflects teacher-forced full-solution context, not autoregressive planning.

5. **No dtype/normalization record.** Hidden states cast `.float()` at save in the repo path; the pickles are float (2048-dim). Anisotropy (mean cosine 0.664 between *random* states, below) is not removed — see intrinsic_noise_audit.

## Reproduced sanity checks on the pickles

- `state_indices` length equals `op_ids` length + 1 in inspected samples → state/op count consistent.
- 5000 train trajectories × ~4 states = 20,022 state vectors total.
- `all_hidden` rows are 2048-dim TinyLlama hidden states; norms vary widely (L2 within-state 28.8 ± 12.5, i.e. ±43% — large heteroscedasticity), which interacts badly with cosine/L2 metrics.

## Answers to required questions

- **Token boundaries correct?** Cannot verify — indexing code absent.
- **State boundaries correct?** Cannot verify — `state_indices` provenance absent.
- **Aligned to intended reasoning step?** Unknown; high risk because prompts contain the full solution and result tokens may be included.
- **Off-by-one bug?** Latent risk in both the `--layer` CLI and the (missing) state-index computation; no guard, no metadata.
- **Could probes see future information?** **Yes, plausibly** — full `cot` is in the prompt and result tokens may be the indexed states. This is a first-order threat to any probe/coherence claim.

## Verdict

Extraction correctness is **unestablished**. The repo script is simplistic but the actual results depend on missing alignment code, and the prompt design (full solution in context) means hidden states are not "autoregressive planning states" — they are teacher-forced read-outs of a fully-revealed solution. This alone can manufacture apparent "planning" signal.
