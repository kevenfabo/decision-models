---
marp: true
theme: research-eng
paginate: true
size: 16:9
header: 'DECISION MODELS — EDITION 01'
footer: 'Team Engineering & Research Initiative · 2026-11-19'
---

# Decision Models
### Jev, System 1 Models, and Open Alternatives

Team Engineering & Research Initiative — Edition #1
2026-11-19

<!--
Welcome / framing: this is edition #1 of a new recurring team session.
Note up front: "Jev" = TypeSafe AI's System 1 model, not JEPA.

TOTAL ESTIMATE after the 2026-11-19 30-min restructure: ~28-32 min slide narration
+ live demo execution (~3 min) = realistic target ~30-35 min, down from ~65-70 min
across 32 slides. 14 presented slides now, 11 in the backup appendix (not presented,
preserves everything cut from the main flow for Q&A -- nothing was deleted).

What changed from the previous (65-70 min) version, to get here:
- Merged 6 slide-pairs/triples into single dense slides (why-exist, challenge+instead,
  why-it-matters-for-us, training, eval).
- Live demo: only entity resolution is run live now (the other two scenarios --
  decision-model-as-judge, LLM contrast -- are mentioned verbally and still fully
  present in demo/decision_model_demo.ipynb, just not executed in front of the room).
- Evaluation: 4 slides (context/data, metrics, results, conclusion) compressed to 1.
  Full detail stays in doc Section 7 and the eval notebook.
- Use cases folded into "why it matters for us" as short examples rather than a
  standalone slide; full list preserved in backup.
- "Why it's fast," "Reality check," "Why RL not just supervised," and the full
  use-case list moved to backup appendix.
If still tight on the day: cut the live demo to 0 scenarios (walk through the
notebook's existing output instead of re-running live) and/or cap Discussion at 2 min.
-->

---

## A quick terminology note

- **Jev** = TypeSafe AI's proprietary **System 1 model**, released 2026-09-15 by founder **Diogo Almeida** (ex-OpenAI, InstructGPT co-author) — named after the Jevons paradox, not JEPA
- Jev itself: closed, API-only, no published weights, no paper
- Today's demo runs on **Laya**, open-source and self-hosted — not Jev

<!--
One sentence, don't dwell. Full founder bio + why not Torchcast Decision: backup appendix.
-->

---

<!-- _class: dense -->

## Part 1 — Why decision models exist

**System 1 vs. System 2** (*Thinking, Fast and Slow*, Kahneman): System 1 is fast,
intuitive, automatic; System 2 is slow, effortful, deliberate.

Most of risk/eval work is **System 1 at scale**: "suspicious, yes or no?", "pass the
rubric?", "same entity?" — millions of bounded judgments needing a calibrated
confidence, not an essay.

**The mismatch:** LLMs are generative even when the task isn't. Ask "fraud, yes or
no" and you still get preamble → maybe a rationale → answer, decoded one token at a
time. **That mismatch is the gap decision models close.**

<!-- 4-5 min -->

---

<!-- _class: dense -->

## Part 2 — The challenge, and what a decision model does instead

Four problems with a generative LLM for decision tasks: **Latency** (sequential
decode) · **Cost** (per-token pricing for a yes/no) · **Calibration** ("80% sure" in
prose ≠ a trustworthy 0.8) · **Brittleness** (reliable JSON is its own maintenance burden)

**Decision model:** state (text/JSON) + typed question (**Choice** / **Noul** / **Score**)
→ probabilities over the declared options, **no tokens generated**. Single forward
pass: 70–500ms vendor claim, 0.15–0.45s hosted (independently measured), ~35–60ms self-hosted.

<!--
[G p.168] vendor; [A p.11] independent hosted medians; Laya 33-40ms on T4 [B p.41].
~4-5 min.
-->

---

## Part 3 — Architecture, vs. traditional LLM

**Traditional decoder LLM:** tokens → transformer → next-token distribution → sample
→ feed back in → repeat. The output *is* a token sequence.

**Decision model:** state + question schema → pretrained backbone (**encoder or
decoder**) → **readout** restricted to option tokens → one parallel pass →
probabilities. No decode loop — there was never a token sequence to decode.

<!-- ~3 min. "Why it's fast" detail list moved to backup. -->

---

<!-- _class: dense -->

## The "how," worked through one example: Qwen3.5-4B → a decision model

| Step | What happens |
|---|---|
| **1. Start** | Pretrained decoder LLM, untouched — same transformer, same weights |
| **2. Readout trick** | Prompt ends right before the answer; restrict the next-token distribution to option-label tokens. One prefill, no decode. Label tokens' output-embedding rows **are** the decision head — zero new params |
| **3. Post-train** | For the interface (options described at request time) + calibration (proper scoring rule, not right/wrong) |
| **4. Self-distillation** | Model reasons with thinking on → that answer distribution becomes the target → same model, reasoning off, learns to match it at the first token. No gold labels |

Verified for **SII-Decision** `[C p.60-62]`. Torchcast Decision 4B is built the same way on the same backbone family — but isn't in the audited literature, so treat its exact recipe as inferred, not confirmed.

<!-- Full version: doc/qa-prep.md Q6a. ~2-3 min. -->

---

## The landscape

- **Jev** (TypeSafe) — proprietary, API-only · **SII-Decision 4B/27B** — open, Qwen decoders
- **Laya** — open, Apache-2.0, ModernBERT encoder, self-hostable *(what we run today)*
- **Torchcast Decision** — 4B/12B/27B *(CUDA-only — not usable on this Apple Silicon machine)*

On **JEVal** (11k instances, 36 datasets): best open model **matches** Jev —
SII-Decision-27B **79.2%** vs Jev **78.4%** — most open models trail.

<!--
[C p.52]: Jev 78.38/ECE .053; SII-27B 79.24/ECE .015; Laya 26.3.
~2 min.
-->

---

<!-- _class: dense -->

## Usage patterns: every call is state + typed questions

Same `/v1/systemone` shape across **every** Jev-like model, any backbone:

| Pattern | Shape | Our example |
|---|---|---|
| **Noul** | yes/no, calibrated probability pair | Entity resolution; decision-model-as-judge |
| **Choice** | fixed option set, probabilities sum to 1 | News sentiment (negative/neutral/positive) |
| **Score** | ordinal scale, full probability distribution | *(not yet used)* — e.g. 1-5 risk severity |

One call can carry **up to 10 questions against the same state**, each answered **in isolation**.

<!-- [A p.2; F p.145]. ~2 min. -->

---

<!-- _class: dense -->

## Part 4 — Training objective & logic

Standard LLM training optimizes for good **text**: next-token cross-entropy, then
often RLHF (reward = did a human prefer this *generated continuation*).

**RLCD** trains for **honest probabilities** instead — a proper scoring rule
(Brier/log-loss), not right/wrong. If the model says 0.8 across 1,000 similar cases,
~800 should actually resolve true.

Why it matters mechanically: a human reader can discount a hedge; code thresholding
a number can't. Calibration must be **verified per workload** — a good *pooled* ECE
can hide bad calibration on any one task.

<!--
[A p.4; I p.194]: RLCD undisclosed, this is the public proper-scoring-rule framing.
[A p.11]: pooled ECE 0.047 vs median per-benchmark ECE 0.168.
"Why RL and not just supervised" (the Brier-gradient identity) moved to backup appendix.
~5-6 min.
-->

---

<!-- _class: dense -->

## Part 5/6 — Why it matters for us

Risk management at a bank — three direct threads:

- **GenAI eval/RAG** — decision-model-as-judge triages: auto-pass/fail the clear
  cases, reserve full LLM-judge scrutiny for the ambiguous ones
- **Graph ML / entity resolution** — reads messy fields like an LLM, answers like
  the classifier our pipeline expects; a fast-path signal alongside graph matching
- **Risk & compliance** — low-latency, calibrated, auditable decisioning is the
  core job description here, not a side benefit

Also fits: moderation/routing, replacing brittle "LLM returns JSON," retrieval reranking.

<!-- Full use-case list: backup appendix. ~5 min. -->

---

<!-- _class: dense -->

## Governance: what model risk will ask

| Concern | Hosted Jev | Self-hosted open / narrow model |
|---|---|---|
| Explainability | Probability only, **no rationale** | Same — compose via a versioned rubric in code |
| Data residency | Single region, early-stage vendor | Stays on-prem |
| Reproducibility | ~1% of answers change on identical repeats | Pinned weights, testable invariances |
| Monitoring | Calibration drifts with prevalence/version | Same — monitor ECE per segment, recalibrate |

**Robustness gap, stated plainly:** swapping which option name means "yes" moved one
study's AUC **.815 → .581**, zero type errors. **Type-safe ≠ correct.**

<!--
[F p.154, p.157]; nondeterminism [A p.4]; name-swap [A p.13]. Realistic route for
regulated decisions: self-hosted open or narrow model. ~3 min.
-->

---

## Live demo

**Live:** entity resolution — two candidate records → match/no-match + probability,
on **Laya** (ModernBERT, local, MPS). Torchcast Decision 4B was the original plan —
its own docs require a CUDA-only stack with no Apple Silicon path.

**In the notebook, not re-run live today:** decision-model-as-judge (same model,
swapped schema → RAG pass/fail), and the LLM contrast (same call via Ollama/qwen3.6,
timed side by side) — both fully working in `demo/decision_model_demo.ipynb`.

<!-- ~2 min narration + ~2-3 min live execution. -->

---

<!-- _class: dense -->

## Evaluation — context, data, metrics, results

**Question:** with an LLM already available, is a dedicated decision model worth
standing up? **Use case:** financial news sentiment. **Ground truth:** Financial
PhraseBank, `Sentences_AllAgree` (100% annotator agreement), 100 sampled sentences.
**Metrics:** Accuracy/F1 (right label?) · Brier (honest probability?) · Latency
p50/p95 (full distribution) · Cost/1k (both $0 here, self-hosted).

| | Accuracy | Macro F1 | Brier | Latency p50 | Latency p95 |
|---|---|---|---|---|---|
| **Laya** | **92%** | **0.911** | **0.118** | **43 ms** | **50 ms** |
| Ollama (qwen3.6) | 85% | 0.863 | 0.253 | 1077 ms | 1242 ms |

Laya wins every axis: more accurate, ~half the Brier score, **~25x faster**. Caveat:
100 samples on one task is a signal, not a certified benchmark. Next step: shadow-mode
pilot — measure on our own data before setting any threshold. Full methodology and
figures: `eval/decision_model_eval.ipynb`, doc Section 7.

<!-- ~4-5 min. -->

---

## Discussion

Where in **our** stack could a decision model replace:

- an LLM-as-judge call?
- a hand-rolled classifier?

*(Full source list: written doc. Cap at ~2-3 min if the demo ran long.)*

---

# Appendix
### Backup Q&A — not presented

---

<!-- _class: backup -->

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

<!-- _class: backup -->

## Why Laya, not Torchcast Decision 4B

Torchcast Decision's own docs: "Do not serve the weights with vLLM, llama.cpp, MLX
or a generic `transformers` causal-LM pipeline: they apply a causal mask and give
different probabilities." It mixes causal and non-causal (bidirectional) attention
layers and requires its CUDA-only stack (`flash-linear-attention`, `causal-conv1d`)
for correct results — no legitimate CPU/MPS path exists.

Laya is a standard ModernBERT encoder with explicitly documented MPS support
(`LAYA_DEVICE=mps`), so it's the one that actually runs, correctly, on this machine.

---

<!-- _class: backup -->

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

<!-- _class: backup -->

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

<!-- _class: backup -->

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

<!-- _class: backup -->

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

<!-- _class: backup -->

## Why RL, and not just supervised training?

With reward r = c − p, the expected REINFORCE gradient is **exactly −½ ∇ Brier**.
Known labels → plain proper-scoring loss reaches the same optimum, no sampling variance.
A 0/1 accuracy reward (RLVR-style) → **overconfidence** instead.

<!--
Gradient identity: E[r_a ∇log p_a] = Σ (c_a − p_a) ∇p_a = −½ ∇ Σ (p_a − c_a)²  [I p.194].
RL still useful when you only observe the outcome of the action taken (bandit feedback).
-->

---

<!-- _class: backup -->

## Where the ideas come from

Almost every component has an ancestor — the novelty is the **combination**.

| Component | Ancestor |
|---|---|
| Score labels by LM probability | Cloze / verbalizer classification (PET, LM-BFF), GPT-3 likelihood scoring |
| Guaranteed-valid output | Grammar-constrained decoding |
| Runtime labels | NLI zero-shot classifiers, GLiClass |
| Confidence-based escalation | Reject option, learning to defer, LLM cascades (FrugalGPT) |
| Calibration | Proper scoring rules, temperature scaling |

<!--
[A p.3 Table 1].
-->

---

<!-- _class: backup -->

## Full use-case list

- Fast guardrails/routers inside agent pipelines
- Moderation & transaction flagging, at volume
- A/B and multi-way routing
- Replacing brittle "LLM returns JSON" classification
- **Entity mapping/resolution** — match / no-match + calibrated probability
- **Decision-model-as-judge** — calibrated pass/fail instead of LLM-judge prose
- **Retrieval** — rerank or navigate documents (DecisionRAG: +13.8 R@1 over dense + 8B reranker)

<!-- DecisionRAG [E p.122]. -->

---

<!-- _class: backup -->

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

<!-- _class: backup -->

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

---

<!-- _class: backup -->

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
