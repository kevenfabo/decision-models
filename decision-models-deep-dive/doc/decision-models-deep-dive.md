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

The detail worth sitting with — stated more carefully than the usual pitch: **a decision model is usually built on an existing pretrained backbone, not a new architecture family, but that backbone can be an encoder or a decoder.** The new parts are the readout, the request layout (state + typed questions), and a calibration-oriented training objective — not a wholesale rethink of the transformer.

- **Decoder-based** (SII-Decision on Qwen3.5-4B/Qwen3.8-27B, SearchJev, JevLite): read the next-token softmax at the answer position, restricted to the option-label tokens. No new parameters needed — the LM head's existing output-embedding rows *are* the decision head. Inherits the backbone's world knowledge.
- **Encoder-based** (Laya on ModernBERT-large, 421M params; Open-Jev on DeBERTa-v3-large): a marker token or pooled representation per option, fed to a small head. Bidirectional within a block, cheap, can guarantee option-order invariance by construction, easy to self-host — but shorter context (Laya: 512 tokens English) and weaker on knowledge-heavy tasks.
- A finding worth knowing before you reach for a "decision model" library: on matched backbone, data, and budget, a **dedicated typed head performs about the same as just reading the ordinary LM head at the option tokens** (.761 vs .761 in one study) — a freshly-initialized head can even score *lower*. The gain isn't the head architecture; it's the objective it's trained with (Section 4).

| | Traditional LLM | Decision model |
|---|---|---|
| Output | Token sequence | Typed value (choice / yes-no / score) + calibrated probability |
| Inference | Autoregressive decode loop | Single parallel pass (prefill only) |
| Typical latency | Seconds, scales with output length | Hosted Jev: independently measured 0.15–0.45s; self-hosted open models: 30–60ms |
| Backbone | Purpose-trained or fine-tuned decoder | Usually an existing pretrained backbone — encoder or decoder |
| Who consumes the output | A human reading prose | Downstream automated logic thresholding a probability |

Jev itself is proprietary and API-only (`POST /v1/systemone`, model route `jev-latest`), architecture undisclosed. TypeSafe says only that it is "neither small nor an LLM" — unverifiable from outside. On the public **JEVal** benchmark (11k instances, 36 datasets), the picture is more mixed than "open already wins": only **SII-Decision-27B** edges past Jev (79.24% vs 78.38% accuracy), and it's meaningfully better calibrated (ECE 0.015 vs 0.053). Most open decision models still score below Jev, and some far below — Laya 26.3%, a "NanoJev" variant 25.3%. The defensible claim: **the best open model now roughly matches Jev on accuracy and is better calibrated** — not that open has broadly overtaken it. `xor` (Apache-2.0, Hugging Face: `juspay/xor`) and Torchcast Decision (4B/12B/27B, mixed Apache-2.0/CC-BY-NC) are both usable today without a TypeSafe account, but neither appears in the published evaluation literature — treat vendor/community benchmark claims about them as unverified until independently tested, the same way you would for Jev's own numbers.

### Usage patterns: Noul, Choice, Score

Every Jev-like model, proprietary or open, is called the same way regardless of backbone: a `state` (the situation, as text or JSON) plus one or more typed `questions` against a TypeSafe-compatible `/v1/systemone` endpoint. Each question declares which of three answer shapes it wants back:

- **Noul (yes/no).** A binary decision with a calibrated probability pair (`{"yes": 0.93, "no": 0.07}`). The field is literally named `noul` in the response. This is the shape behind our entity-resolution demo ("same entity: yes/no") and decision-model-as-judge ("does this answer pass the rubric: yes/no").
- **Choice.** A fixed, mutually exclusive option set, each with a probability that sums to 1 across the set. This is the shape behind our news-sentiment eval (negative/neutral/positive) and fits any routing decision — "which support queue," "which of these three remediation paths."
- **Score.** A probability distribution across an ordinal scale (e.g., a 1–5 severity score). The model returns the *expected* level, not just the argmax, so it can land between two levels — "mostly a 3, meaningfully some chance it's a 4" stays visible rather than collapsing to one number. Training adds a ranked probability score on top of log-loss, so predicting "high" when the truth is "critical" costs less than predicting "none" — distance along the scale matters, not just exact match.

One practical detail worth knowing before integrating any of this: a single call can carry **up to 10 questions against the same state** — e.g., asking both "escalate: yes/no" (Noul) and "which team: billing/technical/other" (Choice) about the same transaction in one round trip, rather than one call per question. Questions in the same call are answered in isolation — each sees the shared state but not the other questions' answers.

### Limitations and watch-outs

Worth stating plainly, especially presenting to a risk/compliance-literate audience: **type-safe does not mean faithful.** The output is always a valid value from the declared option set — it is not guaranteed to be the *right* one, and the guarantee of validity can create false confidence.

- **Option framing changes answers, not just confidence.** Swapping which option name means "yes" versus "no" across a set of rubrics moved one model's AUC from .815 to .581 — with zero type errors, because every answer was still validly one of the allowed options.
- **Calibration is a training goal, not a free property — verify it on your own data.** Jev's pooled ECE (0.047 in one audit) looks good, but the *median per-benchmark* ECE was 0.168 — pooling across many tasks hid substantial miscalibration on individual ones. High stated confidence also doesn't guarantee correctness: one study found wrong answers carrying a mean confidence of .778.
- **Context can flip an initially-correct decision.** Answer-preserving additions to the input (text that doesn't change the right answer) redirected 61% of initially-correct decisions in one adversarial test, versus 2% for a neutral addition of similar length.
- **Hosted determinism is imperfect.** Roughly 1% of answers changed on identical repeated hosted calls to Jev in one measurement. A self-hosted model with pinned weights can be made fully deterministic; a hosted black-box API cannot be assumed to be.

None of this is a reason to dismiss decision models — it's the same due diligence you'd apply to any new scoring model before it touches a regulated decision: measure calibration on your own workload, test option-name robustness, and don't treat "always returns a valid value" as "always returns a correct one."

## 4. Training objective & logic

Standard LLM training has two stages with two different objectives: **pretraining** on next-token cross-entropy (predict the next token, given everything before it) and, often, **RLHF** — reinforcement learning from human feedback, optimizing for which *generated continuation* a human or a preference model prefers. Both stages are shaped around producing good text.

Decision models are trained with something TypeSafe calls **RLCD — Reinforcement Learning for Calibrated Decisions**. TypeSafe hasn't published the exact recipe behind Jev specifically, so treat the name as the vendor's framing rather than a confirmed description of what Jev's training actually does. What *is* public is the general principle it's presented against: a **proper scoring rule** — a loss whose unique minimum is the true probability (Gneiting & Raftery, 2007). The reward signal isn't "did a human like this continuation" — it's "was the probability this model assigned actually well-calibrated against ground truth outcomes." If the model says 0.8 confidence across a thousand similar cases, roughly 800 of them should actually resolve true.

There's a clean piece of math underneath this worth knowing, because it also explains why RL is even necessary. Take an RLCD-style reward r = c − p, where c ∈ {0,1} is whether the chosen option was actually correct and p is the probability the model assigned it. The expected REINFORCE gradient of that reward turns out to be **exactly half the negative gradient of the Brier score** — the standard proper-scoring loss. Which means: whenever you actually know the ground truth label, plain supervised training directly on a proper scoring rule reaches the *same* optimum, without the sampling variance RL introduces. RL only earns its keep when you *don't* get to see the label — only the outcome of the specific action you took (bandit feedback). This also clarifies why the wrong reward produces the wrong failure mode: a plain 0/1 "was it correct" reward (as in RLVR-style verifiable-reward RL) optimizes accuracy, not calibration — and a model optimized that way is expected to become *overconfident*, the opposite of what a downstream threshold needs.

That distinction matters mechanically, not just philosophically: a generative model's output is consumed by a human reading prose, who can discount a hedge or notice uncertainty between the lines. A decision model's output is consumed by code — a threshold, a router, an escalation rule — which has no way to "read between the lines" of a miscalibrated number. Calibration has to be trained in directly, because nothing downstream will compensate for its absence.

## 5. Use cases

- **Fast guardrails/routers** inside agent pipelines — a cheap typed gate before a more expensive step runs.
- **Moderation and transaction flagging** — score-based gating at the volume these tasks actually run at.
- **A/B and multi-way routing** — "which of these three paths should this request take," answered directly instead of parsed out of generated text.
- **Replacing brittle "LLM returns JSON" classification** — the typed-output contract is enforced by the model's output head, not by prompt engineering.
- **Entity mapping/resolution** — "are these two records the same entity, yes or no," with a calibrated match probability, reading messy free-text fields the way an LLM would but answering the way a classifier does. Worth flagging: accuracy at high option cardinality varies a lot by model choice — on a 77-label benchmark, one decision model scored 0.870 while a smaller one scored 0.425. Routing/resolution tasks with many candidate entities are exactly where that gap would bite.
- **Decision-model-as-judge** — in evaluation pipelines, a calibrated pass/fail or score in place of an LLM-as-judge's prose rationale: faster, cheaper, and the output is already in the shape your eval dashboard wants. The retrieval side has independent support too: in one hands-on test, a decision model used as a BM25 reranker took R@1 from 33% to 58%, while a general LLM asked to score the same passages returned ties on 231 of 234 groups.

## 6. Why it matters for us

We sit in risk management at a bank, and three threads of our actual work connect directly to this.

**GenAI evaluation / RAG.** Full LLM-as-judge evaluation — prompting a large model to critique a RAG answer in prose, then parsing a verdict back out — is slow and expensive to run on every response. A decision-model-as-judge can act as a cheap triage layer: auto-pass or auto-fail the clear cases at a fraction of the latency and cost, and reserve full LLM-as-judge scrutiny for the genuinely ambiguous ones. The realistic first pilot is **shadow mode**: run the decision model alongside the existing LLM-as-judge without acting on it yet, compare the two, measure calibration against our own labels, and only fix an escalation threshold once that comparison looks right. One caveat worth carrying in from the start: when an LLM judge is used as the escalation fallback, it tends to **repeat the decision model's own mistakes** on the cases the decision model was most confidently wrong about — in one study, 96% of the time. That caps how much accuracy escalation can actually recover, so the pilot needs to measure that correlation directly rather than assume escalation is a safety net.

**Graph ML / entity resolution.** Entity resolution is already a typed-decision problem — match or no-match, with a confidence we act on. A decision model gives us a way to bring unstructured case text and messy record fields into that same scoring step, reading like an LLM but answering like the classifier our resolution pipeline already expects, as a complementary fast-path signal alongside our existing graph-based matching.

**Risk and compliance, specifically.** Low-latency, calibrated, auditable decisioning isn't an interesting side benefit for a risk department — it's the core job description. Approve/deny, flag/escalate — these are exactly the typed, calibrated, high-volume decisions this category of model is built around. But the fit cuts both ways, and the honest version matters more here than anywhere else in this doc:

- **No explanations.** A decision model returns a probability, not a reason. Auditability has to come from *decomposition and composition* — breaking a decision into atomic typed questions and combining them through a versioned rubric that lives in our own code, not from the model explaining itself.
- **Hosted Jev is a governance-relevant black box.** Undisclosed architecture and training data, a single serving region, an early-stage vendor, and roughly 1% nondeterminism on repeated identical calls. For anything touching a regulated decision, a **self-hosted open or narrow model** — with pinned weights, on-premises data, and architecture we can actually inspect — is the realistic route, not the hosted API.
- **Calibration needs ongoing monitoring, not a one-time check.** It drifts when prevalence, workload, or model version changes. Monitor error (e.g. ECE) per segment, not just pooled, and recalibrate on a schedule.
- **Human review budgets can be small and still effective — if measured.** In one case, reviewing just 0.8% of flagged records (the lowest-confidence ones) reached 90% precision overall — but the per-variable budget needed to hit that bar varied up to 72% depending on the field. The budget has to be measured per use case, not assumed.

None of this argues against piloting decision models here — if anything, typed, thresholdable, composable output is a better fit for model-risk governance than free-text LLM output ever was. It argues for evaluating them with the same rigor we'd apply to any new scoring model before it touches a regulated decision.

## Sources

**Primary research (via internal corpus, not independently re-verified by this doc's author):**
- Tang & Zheng, *Typed Decision Models: An Early Evidence Audit and Evaluation Checklist*, arXiv:2609.32160
- dos Santos, *Calibrated Decision Models for Autonomous Penetration-Testing Harnesses: Jev and Laya as System One Decision Layers*, arXiv:2609.28940
- Duan, Lin, Wang, Xiang et al., *General Decision Models: Benchmarking and Insights Beyond Jev*, arXiv:2610.03935
- Wan, Jünger et al., *Retriever-Free Retrieval: Navigating Documents with Decision Models*, arXiv:2610.35868
- Antunes et al., *Narrow Decision Models: A System-1 Architecture for Calibrated, Typed Decisions*, arXiv:2610.25553
- Wu & Lim, *REFLEX with Jev for Efficient Selective Control in LLM Agents*, arXiv:2609.26532
- Cao et al., *SearchJev: A Calibrated System-1 Model for Search Agents*, arXiv:2610.05107

**Secondary/vendor sources:**

- [What is Jev? TypeSafe AI's System One decision model](https://www.sanity.io/glossary/jev-typesafe-ai-model)
- [What Is Jev? TypeSafe's Decision Model Explained for Developers](https://openrouter.ai/blog/insights/what-is-jev/)
- [What is TypeSafe AI? The company behind Jev](https://befailproof.ai/jev/typesafe-ai/)
- [GitHub — andyrewlee/awesome-system-one](https://github.com/andyrewlee/awesome-system-one)
- [AI Decision Models: Jev, Laya, and Open Alternatives](https://www.layer3labs.io/guides/system-one-decision-models)
- [juspay/xor on Hugging Face](https://huggingface.co/juspay/xor)
- [GitHub — juspay/neurolink](https://github.com/juspay/neurolink)
- [GitHub — Torchcast-AI/torchcast-decision-4b](https://github.com/Torchcast-AI/torchcast-decision-4b)
- [systemonemodels.org — AI decision models: every model listed](https://systemonemodels.org/models/)
