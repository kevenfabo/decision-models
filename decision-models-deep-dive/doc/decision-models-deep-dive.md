# Decision Models: Jev, System 1 Models, and Open Alternatives

*Team Engineering & Research Initiative — Edition #1*
*2026-11-18*

## 0. Terminology note

"Jev" here refers to TypeSafe AI's proprietary **System 1 model**, released 2026-09-15 — not JEPA (Joint Embedding Predictive Architecture). Jev is closed: no published weights, no paper, API-only. Everything we'll run ourselves in the demo comes from the independent open-source ecosystem that grew up around the same idea (Torchcast Decision, Juspay's `xor`), not from Jev itself.

## 1. Why decision models exist

Daniel Kahneman's *Thinking, Fast and Slow* splits cognition into two systems: **System 1** is fast, intuitive, automatic — the kind of judgment you make in a fraction of a second with no conscious deliberation. **System 2** is slow, effortful, deliberate — reasoning you do step by step, out loud or on paper.

Most of what a risk, fraud, or evaluation pipeline needs is System 1 work at scale: *is this transaction suspicious, yes or no? Does this answer satisfy the rubric, pass or fail? Are these two customer records the same entity?* Millions of small, bounded, repeatable judgments, each needing a calibrated confidence, not an essay.

Large language models are extraordinary at System 2-shaped work — multi-step reasoning, drafting, synthesis — but they're generative by construction, even when the task doesn't call for generation at all. Ask an LLM "is this fraud, yes or no" and it still produces a chain of tokens: preamble, maybe a rationale, then the answer, decoded one token at a time. That mismatch — using a token-generating machine to answer a task that's really just "pick one, and tell me how sure you are" — is the gap decision models are built to close.

## 2. The challenge it solves

Four problems show up when you use a generative LLM for a decision task:

- **Latency.** Autoregressive decoding is sequential; every extra token costs another forward pass. A one-word answer still pays for the scaffolding around it.
- **Cost.** Paying per-token for a yes/no is wasteful at volume, and volume is exactly what risk and eval pipelines have.
- **Calibration.** A token's sampling probability is not a trustworthy confidence score. "I'm 80% sure" written in prose is not the same as a calibrated 0.8 that downstream automation can threshold against.
- **Brittleness.** Getting an LLM to emit well-formed, parseable JSON reliably — across prompt rewordings, model updates, and edge cases — is its own maintenance burden.

Decision models remove the generation step entirely. They take a **state** (free text or JSON describing the situation) plus one or more **typed questions** (choice / yes-no / score, with the valid options or scale spelled out), and return calibrated probabilities over those typed outputs in a single forward pass — no tokens, no decode loop, typically 70–500ms end to end, sometimes faster on smaller open models with warm weights.

## 3. Architecture & variants, vs. traditional LLM architecture

**Traditional decoder-only LLM:** input tokens → transformer stack → next-token distribution → sample a token → feed it back in → repeat until done. The output *is* a sequence of tokens; "deciding" among options is something you coax out of that sequence via prompting.

**Decision model:** input = state + a typed question schema → shared transformer backbone (same embeddings, same attention machinery you already know from LLMs) → a classification/scoring **head** replacing the language-modeling head → one parallel forward pass → calibrated probabilities over the question's typed answer space (a choice distribution, a yes/no pair, or a score distribution over ordinal levels). No decode loop, because there's nothing to decode — the answer was never a sequence of tokens to begin with.

The detail worth sitting with: **a decision model is usually not a new architecture family.** It's a fine-tune (often with a swapped output head) on top of an existing open LLM backbone. Juspay's `xor` is a post-trained version of Qwen3.6-35B-A3B. Torchcast Decision ships three sizes built on Qwen3.5-4B, Gemma-4-12B, and Qwen3.8-27B respectively. The innovation is concentrated in the output stage and the training objective (Section 4), not in a wholesale rethink of the transformer. That's good news for anyone who already understands LLM internals — the mental model transfers almost directly.

| | Traditional LLM | Decision model |
|---|---|---|
| Output | Token sequence | Typed value (choice / yes-no / score) + calibrated probability |
| Inference | Autoregressive decode loop | Single parallel pass |
| Typical latency | Seconds, scales with output length | 70–500ms, independent of a "response length" that doesn't exist |
| Backbone | Purpose-trained or fine-tuned decoder | Usually a fine-tune/head-swap of an existing decoder LLM |
| Who consumes the output | A human reading prose | Downstream automated logic thresholding a probability |

Jev itself is proprietary and API-only (`POST /v1/systemone`, model route `jev-latest`). The open landscape it spawned already out-benchmarks it on community leaderboards: `xor` (Apache-2.0, Hugging Face: `juspay/xor`) and the Torchcast Decision family (4B/12B/27B, mixed Apache-2.0 and CC-BY-NC licensing) are both usable today without a TypeSafe account.

## 4. Training objective & logic

Standard LLM training has two stages with two different objectives: **pretraining** on next-token cross-entropy (predict the next token, given everything before it) and, often, **RLHF** — reinforcement learning from human feedback, optimizing for which *generated continuation* a human or a preference model prefers. Both stages are shaped around producing good text.

Decision models are trained with something TypeSafe calls **RLCD — Reinforcement Learning for Calibrated Decisions**. The reward signal isn't "did a human like this continuation" — it's "was the probability this model assigned actually well-calibrated against ground truth outcomes." If the model says 0.8 confidence across a thousand similar cases, roughly 800 of them should actually resolve true. That distinction matters mechanically, not just philosophically: a generative model's output is consumed by a human reading prose, who can discount a hedge or notice uncertainty between the lines. A decision model's output is consumed by code — a threshold, a router, an escalation rule — which has no way to "read between the lines" of a miscalibrated number. Calibration has to be trained in directly, because nothing downstream will compensate for its absence.

## 5. Use cases

- **Fast guardrails/routers** inside agent pipelines — a cheap typed gate before a more expensive step runs.
- **Moderation and transaction flagging** — score-based gating at the volume these tasks actually run at.
- **A/B and multi-way routing** — "which of these three paths should this request take," answered directly instead of parsed out of generated text.
- **Replacing brittle "LLM returns JSON" classification** — the typed-output contract is enforced by the model's output head, not by prompt engineering.
- **Entity mapping/resolution** — "are these two records the same entity, yes or no," with a calibrated match probability, reading messy free-text fields the way an LLM would but answering the way a classifier does.
- **Decision-model-as-judge** — in evaluation pipelines, a calibrated pass/fail or score in place of an LLM-as-judge's prose rationale: faster, cheaper, and the output is already in the shape your eval dashboard wants.

## 6. Why it matters for us

We sit in risk management at a bank, and three threads of our actual work connect directly to this:

- **GenAI evaluation / RAG.** Full LLM-as-judge evaluation — prompting a large model to critique a RAG answer in prose, then parsing a verdict back out — is slow and expensive to run on every response. A decision-model-as-judge can act as a cheap triage layer: auto-pass or auto-fail the clear cases at a fraction of the latency and cost, and reserve full LLM-as-judge scrutiny for the genuinely ambiguous ones. Same idea, applied to our RAG evaluation pipelines rather than a demo.

- **Graph ML / entity resolution.** Entity resolution is already a typed-decision problem — match or no-match, with a confidence we act on. A decision model gives us a way to bring unstructured case text and messy record fields into that same scoring step, reading like an LLM but answering like the classifier our resolution pipeline already expects, as a complementary fast-path signal alongside our existing graph-based matching.

- **Risk and compliance, specifically.** Low-latency, calibrated, auditable decisioning isn't an interesting side benefit for a risk department — it's the core job description. Approve/deny, flag/escalate, these are exactly the typed, calibrated, high-volume decisions this entire category of model is built around. Decision models are a plausible, technically legible bridge between LLM-adjacent capabilities and the regulated, explainable scoring behavior compliance already requires of us — worth evaluating seriously rather than treating as a curiosity.

## Sources

- [What is Jev? TypeSafe AI's System One decision model](https://www.sanity.io/glossary/jev-typesafe-ai-model)
- [What Is Jev? TypeSafe's Decision Model Explained for Developers](https://openrouter.ai/blog/insights/what-is-jev/)
- [What is TypeSafe AI? The company behind Jev](https://befailproof.ai/jev/typesafe-ai/)
- [GitHub — andyrewlee/awesome-system-one](https://github.com/andyrewlee/awesome-system-one)
- [AI Decision Models: Jev, Laya, and Open Alternatives](https://www.layer3labs.io/guides/system-one-decision-models)
- [juspay/xor on Hugging Face](https://huggingface.co/juspay/xor)
- [GitHub — juspay/neurolink](https://github.com/juspay/neurolink)
- [GitHub — Torchcast-AI/torchcast-decision-4b](https://github.com/Torchcast-AI/torchcast-decision-4b)
- [systemonemodels.org — AI decision models: every model listed](https://systemonemodels.org/models/)
