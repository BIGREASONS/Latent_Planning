# Limitations

While our methodology isolates and tests latent representations, several limitations contextualize our findings:

1. **Model Scope and Scale:** Our primary conclusions currently rest on models up to 7B parameters. While Qwen2.5-7B and Mistral-7B represent state-of-the-art open weights in their class, it remains possible (though unlikely, given the trend from 1B to 7B) that abstract, reusable planning states are an emergent property that only crystallizes at the 70B+ or frontier scale (e.g., GPT-4 class).
2. **Domain Specificity:** We focus on symbolic, arithmetic reasoning tasks (Countdown, Game of 24) because they provide a strict, verifiable action grammar. It is possible that latent planning structures emerge differently in semantic or spatial domains where transition rules are softer.
3. **Freezing the LM:** We probe frozen representations. It is entirely possible that explicit, contrastive end-to-end training (like the Coconut architecture) can *force* a model to learn a transition space. Our claim is specifically that standard autoregressive next-token pre-training and supervised fine-tuning do *not* naturally induce these structures.
4. **Decoder Bottleneck:** Our evaluation relies on a shallow diagnostic decoder mapping states back to tokens. While we validated this with a positive control (FSM), the true internal computation might utilize pathways the decoder cannot easily linearize.
