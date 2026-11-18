---
marp: true
theme: research-eng
paginate: true
size: 16:9
header: 'DECISION MODELS — EDITION 01'
footer: 'Team Engineering & Research Initiative · 2026-11-18'
---

# Decision Models
### Jev, System 1 Models, and Open Alternatives

Team Engineering & Research Initiative — Edition #1
2026-11-18

<!--
Welcome / framing: this is edition #1 of a new recurring team session.
Note up front: "Jev" = TypeSafe AI's System 1 model, not JEPA.
-->

---

## A quick terminology note

- **Jev** = TypeSafe AI's proprietary **System 1 model**, released 2026-09-15
- Not JEPA (Joint Embedding Predictive Architecture) — different thing entirely
- Jev itself: closed, API-only, no published weights, no paper
- Today's demo runs on the **open** ecosystem that grew up around the same idea

<!--
One sentence, don't dwell — just prevents confusion for the rest of the deck.
-->

---

## Who's behind Jev

**Diogo Almeida** — founder, TypeSafe AI
- Ex-OpenAI: worked on the instruction-following methods behind ChatGPT; co-author of **InstructGPT**
- Driving question: *"Models have been superhuman at chat for years, so where is all the automation?"*
- Two years in stealth → launch 2026-09-15, **$40M seed**

**The name:** after **William Stanley Jevons** — the Jevons paradox.
*"Every order of magnitude drop in the cost of intelligence unlocks orders of magnitude more use cases."*

<!--
Sources: launch post [G p.166, p.175]; $40M [F p.144]; InstructGPT author list [A p.19].
TypeSafe's docs say he "co-invented RLHF" [H p.178] — vendor framing; RLHF predates InstructGPT
(Christiano et al. 2017). Safe phrasing: "part of the team that applied RLHF to InstructGPT/ChatGPT".
-->

---

## Part 1 — Why decision models exist

### System 1 vs. System 2
*Thinking, Fast and Slow* (Kahneman)

- **System 1** — fast, intuitive, automatic. Sub-second judgment, no deliberation.
- **System 2** — slow, effortful, deliberate. Step-by-step reasoning.

<!--
4-5 min. Land the framing before anything technical.
-->

---

## Most of risk/eval work is System 1 at scale

- Is this transaction suspicious — yes or no?
- Does this answer satisfy the rubric — pass or fail?
- Are these two records the same entity?

Millions of small, bounded, repeatable judgments.
Each needs a **calibrated confidence** — not an essay.

---

## The mismatch

LLMs are generative by construction — **even when the task isn't**.

Ask an LLM "is this fraud, yes or no" and it still:
preamble → maybe a rationale → answer → decoded **one token at a time**

Using a token-generating machine to answer
"pick one, and tell me how sure you are."

**That mismatch is the gap decision models close.**

---

## Part 2 — The challenge it solves

Four problems with using a generative LLM for a decision task:

1. **Latency** — sequential decoding, every extra token = another forward pass
2. **Cost** — per-token pricing for a yes/no, at risk/eval volume
3. **Calibration** — "I'm 80% sure" in prose ≠ a trustworthy 0.8
4. **Brittleness** — reliable JSON-emission is its own maintenance burden

<!-- 4-5 min -->

---

## What a decision model does instead

**Input:** state (text/JSON) + typed question(s) — **Choice** / **Noul** (yes-no) / **Score**

**Output:** probabilities over the declared options — **no tokens generated**

**Latency:** single forward pass, no decode loop
- 70–500 ms — vendor claim (Jev)
- 0.15–0.45 s — Jev hosted, measured by independent studies
- ~35–60 ms — open models, self-hosted on one GPU

<!--
[G p.168] vendor; [A p.11] independent hosted medians; Laya 33–40 ms on T4 [B p.41];
SII-Decision-4B ~60 ms on H100 [C p.62]. Hosted numbers include the network hop.
-->

---

<!-- _class: dense -->

## Isn't that just a classifier?

Structurally, yes: closed output space, one pass, a softmax. The difference is the **interface**.

| | Classical classifier (LogReg / GBM / fine-tuned BERT) | Decision model |
|---|---|---|
| Label set | Fixed at training time | **Declared per request**, read from descriptions |
| Input | Features, or text for *one* task | Any text/JSON state + instructions |
| New task | Retrain | Write a new question schema |
| Questions per call | One | Many, **isolated**, over one shared state |
| Output used as | Argmax label | The **distribution** — threshold, route, defer |

**A classifier whose label set is an argument.**

*Honest caveat: with enough labels and a stable taxonomy, a task-specific classifier is often still the most accurate.*

<!--
Three defining properties from the evidence audit [A p.2]. Caveat: [I p.183], [A p.13 item B3].
Lineage: NLI zero-shot classifiers, GLiClass [C p.49].
-->

---

## Part 3 — Architecture, vs. traditional LLM

**Traditional decoder-only LLM**

input tokens → transformer stack → next-token distribution
→ sample → feed back in → **repeat**

The output *is* a token sequence. "Deciding" is coaxed out of it via prompting.

<!-- 9-10 min for this section (now 13-15 with the added slides) -->

---

## Decision model architecture

state + typed question schema
→ pretrained transformer backbone *(encoder or decoder)*
→ **readout**: option-label logits (decoder) or option-marker / pooled vectors (encoder)
→ **one parallel pass**
→ probabilities over the declared options

No decode loop — there was never a token sequence to decode.

---

## Why it's fast

1. **No autoregressive decoding** — one prefill pass; latency scales with *input* length, not output
2. **State encoded once, shared by every question** — reuse the state's KV cache across questions
   *(prefix sharing: 8.9× faster per question at 32 questions/image)*
3. **Smaller models** — Laya 421M; Softplan NDM ~110M trained parameters
4. **No retries** — output is valid by construction, no "re-prompt because the JSON broke"

*Watch out:* some wins are serving design, not the model — caching shrank one reported gap to <1 ms; vs. **one-token** constrained generation the readout is only ~1.03× faster.

<!--
[C p.61] latency depends only on backbone + input length. [I p.188, p.190] shared state / exact KV reuse.
[A p.5] Visual Jev 8.9×. [B p.40], [I p.189] sizes. [A p.11] caching; [A p.9] Zhou et al. 1.03×.
-->

---

## The detail worth sitting with

**A decision model is usually not a new architecture family.**

It's built on an existing pretrained backbone — **encoder or decoder**:

- **Laya** — ModernBERT-large encoder (421M), Apache-2.0
- **SII-Decision** — Qwen3.5-4B / Qwen3.8-27B decoders
- **SearchJev** — Qwen3.5-0.8B / 4B decoders

In decoders, **keeping** the pretrained LM head beats swapping in a new one.
The innovation lives in the readout, request layout, and training objective.

<!--
[A p.4-5], [B p.40], [C p.61] (new head lowered accuracy by several points),
[A p.8] Yu & Yao: typed head .761 = LM head .761. [K p.255].
-->

---

<!-- _class: dense -->

## Encoder or decoder? Both.

Jev: **undisclosed** — TypeSafe only says it's *"neither small nor an LLM."*

| | Encoder-based | Decoder-based |
|---|---|---|
| Examples | Laya (ModernBERT), Open-Jev (DeBERTa-v3), Softplan NDM | SII-Decision, SearchJev, JevLite (Qwen) |
| Readout | Marker token or pooled vector per option → head | Next-token softmax restricted to option labels |
| Strengths | Bidirectional, cheap, invariances **by construction**, easy to self-host | Inherits LLM knowledge, **zero new params**, prefix cache |
| Weaknesses | Short context (Laya: 512 tok), weaker on knowledge tasks | Larger; option-order bias unless mitigated |
| JEVal accuracy | Laya 26.3% | SII-Decision-27B **79.2%** |

**BERT-based?** Some *are*. What they add over a BERT classifier: request-time labels, many isolated questions per pass, calibration-oriented training.
*"Non-autoregressive" ≠ encoder — a decoder doing one prefill is non-autoregressive too.*

<!--
[A p.4-5], [B p.41], [C p.52], [I p.186] (why encoder), [C p.61] (why decoder).
Caveat: Laya vs SII gap mixes architecture with scale + training — "task training and backbone quality
matter more" [A p.6]. Robustness: DeBERTa pooling flips 19.5% vs ModernBERT marker 76.9% under
yes/no name swaps [A p.4].
-->

---

<!-- _class: dense -->

## Two reference designs

**(a) First-token decision readout** — decoder (SII-Decision, SearchJev)
```
"State … Instruction … (A) … (B) … (C) …  The correct answer is ("
                                        one prefill │ logits at this position
               softmax restricted to {A,B,C} label tokens → p(option)
  Choice = argmax · Noul = p(true) · Score = Σ v_k · p(k)   (expected level)
```

**(b) Prefix tree + block attention mask** — encoder (Softplan NDM)
```
State ─┬─ Q1 Choice ─┬─ O11     siblings share logical positions and
       │             └─ O12     attend only to themselves + ancestors
       ├─ Q2 Score ── levels    → one sequence, one encoder pass
       └─ Q3 Noul               → set head compares options (permutation-equivariant)
```
(b) guarantees **option-order invariance** and **question isolation** by construction — provable, testable.

<!--
[C p.60-61]: no new params — output-embedding rows of label tokens are the head; + Reasoning-to-Readout
Self-Distillation (model's own post-reasoning distribution → first answer token, no gold labels).
[I p.185-189]: Props 1-3; 6.4M new params on 110M body; explicit "none of these" for abstention.
-->

---

## Side by side

| | Traditional LLM | Decision model |
|---|---|---|
| Output | Token sequence | Typed value + probability |
| Inference | Autoregressive decode loop | Single parallel pass |
| Latency | Seconds, scales with output | 35 ms – 0.5 s, no "response length" |
| Backbone | Purpose-trained/fine-tuned decoder | Existing pretrained encoder or decoder + new readout |
| Consumer of output | A human reading prose | Code thresholding a probability |

---

## The landscape

- **Jev** (TypeSafe) — proprietary, API-only (`POST /v1/systemone`, `jev-latest`)
- **SII-Decision 4B / 27B** — open, Qwen decoders
- **Laya** — open, Apache-2.0, ModernBERT encoder, self-hostable, fine-tunable
- **SearchJev** — open, specialised for search-agent decisions
- **Torchcast Decision** — 4B/12B/27B family *(what today's demo runs)*

On **JEVal** (11k instances, 36 datasets): best open model **matches** Jev
— SII-Decision-27B **79.2%** vs Jev **78.4%** — most open models trail.

<!--
[C p.52]: Jev 78.38 / ECE .053; SII-27B 79.24 / ECE .015; Laya 26.3; Nimble-9b 73.9.
Torchcast numbers come from its own GitHub page, not from the corpus.
-->

---

## Part 4 — Training objective & logic

**Standard LLM training**

- Pretraining: next-token cross-entropy
- RLHF: reward = did a human prefer this *generated continuation*

Both stages optimize for producing good **text**.

<!-- 6-7 min (8-9 with the foundations slide) -->

---

## RLCD — Reinforcement Learning for Calibrated Decisions

TypeSafe *describes* the reward as **"was the probability honest"**, not
"did a human like this continuation" — the exact recipe is **not public**.

The public version: train against **proper scoring rules** (Brier, log-loss) —
losses whose unique optimum *is* the true probability.

If the model says 0.8 across 1,000 similar cases,
~800 of them should actually resolve true.

<!--
[A p.4], [I p.194]: RLCD undisclosed. [B p.27]: Brier — minimizer reports p=0.7 when event occurs 70%.
Laya documents Brier training explicitly [B p.41].
-->

---

<!-- _class: dense -->

## Where the ideas come from

Almost every component has an ancestor — the novelty is the **combination**.

| Component | Ancestor |
|---|---|
| Score labels by LM probability | Cloze / verbalizer classification (PET, LM-BFF), GPT-3 likelihood scoring |
| Guaranteed-valid output | Grammar-constrained decoding |
| Runtime labels | NLI zero-shot classifiers, GLiClass |
| Confidence-based escalation | Reject option, learning to defer, LLM cascades (FrugalGPT) |
| Calibration | Proper scoring rules, temperature scaling |

**Why RL and supervised meet:** with reward r = c − p, the expected REINFORCE gradient is
**exactly −½ ∇ Brier**. Known labels → plain proper-scoring loss, same optimum, no sampling variance.
A 0/1 accuracy reward (RLVR-style) → **overconfidence**.

<!--
[A p.3 Table 1]. Gradient identity: E[r_a ∇log p_a] = Σ (c_a − p_a) ∇p_a = −½ ∇ Σ (p_a − c_a)²  [I p.194].
RL still useful when you only observe the outcome of the action taken (bandit feedback).
-->

---

## Why this matters mechanically

- Generative output → consumed by a **human**, who can read between the lines of a hedge
- Decision output → consumed by **code** — a threshold, a router, an escalation rule

Code can't "read between the lines" of a miscalibrated number.
**Calibration has to be trained in directly.**

…and **verified per workload**: raw calibration varies by task — measure ECE/Brier on held-out data, recalibrate (temperature scaling), *then* set thresholds.

<!--
[A p.11 Finding 4]: crash coding slope 1.63, recalibration cut error 3.3×; pooled ECE 0.047 vs
median per-benchmark 0.168.
-->

---

## Part 5 — Use cases

- Fast guardrails/routers inside agent pipelines
- Moderation & transaction flagging, at volume
- A/B and multi-way routing
- Replacing brittle "LLM returns JSON" classification
- **Entity mapping/resolution** — match / no-match + calibrated probability
- **Decision-model-as-judge** — calibrated pass/fail instead of LLM-judge prose
- **Retrieval** — rerank or navigate documents (DecisionRAG: +13.8 R@1 over dense + 8B reranker)

<!--
4-5 min. DecisionRAG [E p.122].
-->

---

<!-- _class: dense -->

## Reality check — what the evidence says

From an audit of the first **28 papers** after Jev's release:

**Holds up**
- Latency & cost gains — tens to hundreds of times cheaper than frontier LLMs
- Accuracy near thinking LLMs **when the answer is in the input**

**Doesn't (yet)**
- **The typed readout itself adds no accuracy** vs. a plain label-probability baseline (only 5 of 26 papers tested it)
- **Type-safe ≠ correct** — swapping yes/no option names: AUC **.815 → .581**, zero type errors
- **Context can flip decisions** — optimized, answer-preserving text redirected 61% of correct answers
- **Computation & long horizons** — τ-bench success **66% vs 78%**; "whether to call a tool" only 52%

<!--
Audit = Tang & Zheng [A]. Findings 1-4 [A p.8-11]; name swap [A p.13]; JevOut 61.4% [A p.11];
τ-bench [C p.58]; BFCL whether-to-call 52.0% [J p.236]. Specialist domains (medicine/finance/law) weaker [C p.48].
-->

---

## The pattern that makes it work: confidence-gated cascade

```
request ─▶ decision model ─▶ p ≥ τ ? ──yes──▶ act automatically
                                  └──no───▶ strong LLM ─▶ still unsure? ─▶ human
```

- 8-way routing, keep only conf ≥ 0.90: **83.8% → 95.5%** accuracy at 70% coverage
- Judge cascade: **92.5 vs 93.1** for GPT-6 alone, at **57% of the cost**
- REFLEX agent: 95% success, **72.7% fewer** strong-model calls

**Limits:** when the fallback shares the same errors, escalation recovers little
(LLM judges repeated Jev's confident mistakes **96%** of the time) — and a cheap
*generative* cascade is a serious baseline. Freeze τ before testing.

<!--
[F p.158] routing; [A p.11] judge cascade; [J p.236] REFLEX; [A p.12] 96% correlated errors;
τ²-bench: generative cascade cheaper, no less accurate — authors withdrew superiority claim [J p.237].
-->

---

## Part 6 — Why it matters for us

We're risk management at a bank. Three direct threads:

**GenAI evaluation / RAG**
Full LLM-as-judge on every response is slow and expensive.
→ Decision-model-as-judge triages: auto-pass/fail the clear cases,
reserve full judge scrutiny for the ambiguous ones.

<!-- 6-8 min total for this section (8-10 with governance) -->

---

## Why it matters for us (cont.)

**Graph ML / entity resolution**
Entity resolution is already a typed-decision problem.
→ A decision model reads messy unstructured fields like an LLM,
but answers like the classifier our resolution pipeline expects —
a complementary fast-path signal alongside existing graph matching.

---

## Why it matters for us (cont.)

**Risk & compliance — specifically**
Low-latency, calibrated, auditable decisioning isn't a side benefit here —
it's the core job description.

Decision models: a technically legible bridge between LLM-adjacent
capabilities and the regulated, explainable scoring compliance already
requires of us.

---

<!-- _class: dense -->

## Governance: what model risk will ask

| Concern | Hosted Jev | Self-hosted open / narrow model |
|---|---|---|
| Explainability | Probability only, **no rationale** | Same — but compose decisions via a versioned rubric in code |
| Data residency | Single region, early-stage vendor | Stays on-prem |
| Reproducibility | ~1% of answers change on identical repeats | Pinned weights, testable invariances |
| Transparency | Architecture, data, reward undisclosed | Open architecture, documented training |
| Monitoring | Calibration drifts with prevalence / version | Same — monitor ECE per segment, recalibrate |

**Auditability via composition:** ask atomic questions, push their *distributions* through a frozen rubric → the decision logic is code, versioned and testable.

<!--
[F p.154, p.157]; nondeterminism 4/510, ≤1.33% [A p.4]; [I p.183, p.193] composition;
monitoring [A p.14]. Realistic route for regulated decisions: self-hosted open or narrow model.
-->

---

## Live demo

1. **Entity resolution** — two candidate records → match / no-match + probability
2. **Decision-model-as-judge** — same model, swapped schema → RAG pass/fail
3. **Contrast** — same entity-resolution call via a generic LLM prompted for JSON, timed side by side

*(Heavy setup pre-run before this session — live cells only.)*

---

## Live demo — pluggable providers

| | Open-source default (today) | Hosted alternative (drop-in later) |
|---|---|---|
| Decision model | Torchcast Decision 4B, local | TypeSafe Jev (`jev-latest`) via API |
| LLM contrast | Ollama, local (`qwen3.6`) | OpenAI or Claude via API |

Default path: no API keys, no internet — everything runs on-machine.
Switching providers is a one-line change in the notebook's Config cell.

---

## Discussion

Where in **our** stack could a decision model replace:

- an LLM-as-judge call?
- a hand-rolled classifier?

---

<!-- _class: dense -->

## Sources

**Papers & reports** (bundled in `corpus/DECISION_MODEL.pdf`)
- Tang & Zheng — *Typed Decision Models: An Early Evidence Audit and Evaluation Checklist* (arXiv:2609.32160)
- Duan et al. — *General Decision Models: Benchmarking and Insights Beyond Jev* (JEVal, SII-Decision; arXiv:2610.03935)
- Antunes et al. (Softplan) — *Narrow Decision Models: A System-1 Architecture…* (arXiv:2610.25553)
- Wan, Jünger et al. — *Retriever-Free Retrieval: Navigating Documents with Decision Models* (arXiv:2610.35868)
- Wu & Lim — *REFLEX with Jev for Efficient Selective Control in LLM Agents* (arXiv:2609.26532)
- Cao et al. — *SearchJev: A Calibrated System-1 Model for Search Agents* (arXiv:2610.05107)
- dos Santos — *Calibrated Decision Models for Autonomous Pentest Harnesses* (arXiv:2609.28940)
- Almeida — *Introducing System One Models & Jev*, TypeSafe blog · TypeSafe docs, *AI primer*

**Web**
- sanity.io/glossary · openrouter.ai/blog · befailproof.ai/jev · layer3labs.io/guides
- github.com/andyrewlee/awesome-system-one · systemonemodels.org/models
- github.com/Torchcast-AI/torchcast-decision-4b

---

# Appendix
### Backup Q&A — not presented

---

<!-- _class: backup -->

## Rapid-fire

| Q | A |
|---|---|
| What's "Noul"? | TypeSafe's yes/no primitive — returns P(yes) |
| Open weights for Jev? | None — API-only |
| Price? | $0.042 / 1M input tokens; output free |
| Max options? | 255 per question; beyond that, go hierarchical |
| Context length? | Jev: unpublished · Laya: 512 tok (EN), 1024 (multilingual) |
| Multilingual? | Jev documented as most accurate in English |
| Explains decisions? | No — probability, not a reason |
| Extra questions add latency? | Vendor: ~no (parallel). One study: 17 labels vs 1 → 1.15 s → 1.22 s |
| Best open model? | SII-Decision-27B — 79.2% on JEVal, ECE 0.015 |

---

<!-- _class: backup -->

## Can it reason? Can we build our own?

**Reasoning:** not at inference — no generated intermediate computation, by design.
*Reasoning-to-Readout Self-Distillation* (SII-Decision): distill the model's own post-reasoning
answer distribution into its first answer token — no gold labels needed.
→ Recovers most of thinking's gain on **weighing** options (intent, emotion, harm),
a quarter or less on **computation** (GPQA, MMLU-Pro). Decompose, or hand off to System 2.

**Build in-house:**
- Cheapest — first-token readout on an open model we already trust (+ self-distillation)
- Narrow — Softplan curriculum: format → read descriptions on *fictional* org charts → target domain;
  agreed with the production LLM classifier on **91%** of held-out docs
- Self-hosted & fine-tunable — Laya (Apache-2.0)

---

<!-- _class: backup -->

## How to benchmark one fairly

Compare against **three** baselines — not just a JSON-prompted LLM:
1. A **label-log-prob readout** on a comparable open model (isolates the interface)
2. A **cheap generative cascade** (if you escalate on confidence)
3. A **task-specific classifier** (if you have labels)

Plus: non-saturated tests · repeated runs with intervals · held-out calibration
(ECE **and** Brier, reliability diagram) · **frozen** thresholds tested on a new workload ·
latency with *and* without caching · cost per **correct** outcome.

*Many early "parity" results came from ceiling-level tests (100/100, 63/63).*
