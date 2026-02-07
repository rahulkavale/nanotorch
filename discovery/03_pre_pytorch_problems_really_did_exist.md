Short answer: some of those problems are well‑documented in the pre‑PyTorch, static‑graph era, but not all of them are equally “proven.” The list mixes hard, documented constraints with more subjective experience.

What is clearly documented for that era
- Static‑graph “define‑then‑run” was the dominant model in TensorFlow 1.x: you build a `tf.Graph` and then execute it via a session. That’s a real source of ceremony and less immediate debugging compared to an imperative runtime. citeturn1search2turn0search2
- Control flow in static graphs required special graph constructs like `tf.while_loop` (and `tf.cond`). That’s extra boilerplate compared to normal Python control flow and makes dynamic models harder to express. citeturn1search5turn1search0
- Theano explicitly documents that function compilation can be time‑consuming and suggests fast‑compile modes to speed iteration. That supports the “feedback latency” pain point. citeturn1search1

What is plausible but not strictly “proven” by docs
- “Too many prerequisites / boilerplate / data loaders” is more about user experience and ecosystem maturity than a hard constraint. It’s likely true for many users in that era, but it’s not a formal limitation in the way static graphs or compilation are.
- “GPU/environment friction” is a general systems issue; it existed, but it’s not specific to any single framework’s design.

So are we sure? Yes for the structural constraints (static graph ceremony, control‑flow boilerplate, compile latency). For the rest, we should treat them as hypotheses based on user experience, not as formal limitations.

If you want, I can refine the use‑case list into two columns: “documented constraints” vs “experience‑based pain points,” and we’ll only build requirements from the first column.
