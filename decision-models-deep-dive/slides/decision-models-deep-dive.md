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

**Input:** state (text/JSON) + typed question(s) (choice / yes-no / score)

**Output:** calibrated probabilities over the typed answer — **no tokens generated**

**Latency:** ~70–500ms, single forward pass, no decode loop

---

## Part 3 — Architecture, vs. traditional LLM

**Traditional decoder-only LLM**

input tokens → transformer stack → next-token distribution
→ sample → feed back in → **repeat**

The output *is* a token sequence. "Deciding" is coaxed out of it via prompting.

<!-- 9-10 min for this section -->

---

## Decision model architecture

state + typed question schema
→ shared transformer backbone *(same embeddings/attention as any LLM)*
→ classification/scoring **head** (replaces the LM head)
→ **one parallel pass**
→ calibrated probabilities over the typed answer space

No decode loop — there was never a token sequence to decode.

---

## The detail worth sitting with

**Usually an existing pretrained backbone — not a new architecture family.**
**But it can be an encoder *or* a decoder.**

| | Decoder-based | Encoder-based |
|---|---|---|
| Examples | SII-Decision (Qwen), SearchJev | Laya (ModernBERT, 421M), Open-Jev (DeBERTa) |
| How it reads the answer | Next-token softmax, restricted to option tokens | Marker/pooled token → small head |
| New parameters | **Zero** — LM head rows *are* the decision head | A small added head |
| Strength | Inherits backbone's world knowledge | Cheap, self-hostable, order-invariant by construction |

A dedicated typed head vs. just reading the ordinary LM head: **.761 vs .761**
in one matched study. The gain isn't the head — it's the training objective (Part 4).

---

## Side by side

| | Traditional LLM | Decision model |
|---|---|---|
| Output | Token sequence | Typed value + calibrated probability |
| Inference | Autoregressive decode loop | Single parallel pass |
| Latency | Seconds, scales with output | Hosted: 0.15-0.45s · Self-hosted: 30-60ms |
| Backbone | Purpose-trained/fine-tuned decoder | Usually pretrained — encoder *or* decoder |
| Consumer of output | A human reading prose | Code thresholding a probability |

---

## The landscape

- **Jev** — proprietary, API-only (`POST /v1/systemone`, `jev-latest`), architecture undisclosed
- **`xor`** (Juspay) — Apache-2.0, open weights, Hugging Face: `juspay/xor`
- **Torchcast Decision** — 4B/12B/27B family, mixed Apache-2.0 / CC-BY-NC

On the public JEVal benchmark: only **SII-Decision-27B** edges past Jev
(79.2% vs 78.4%) — and is better calibrated (ECE 0.015 vs 0.053).
Most open models score *below* Jev, some far below (Laya 26.3%).

**Fair claim:** best open model ≈ matches Jev, better calibrated.
**Not:** "open already wins." (`xor` / Torchcast aren't in the published literature yet.)

---

## Usage patterns: every call is state + typed questions

Same `/v1/systemone` shape across **every** Jev-like model, any backbone:

`state` (text/JSON) + one or more `questions`, each declaring an answer shape:

- **Noul** — yes/no, calibrated probability pair `{yes: .., no: ..}`
- **Choice** — fixed option set, probabilities sum to 1
- **Score** — ordinal scale, full probability distribution across levels

---

## Usage patterns — where we already used them

| Pattern | Our example |
|---|---|
| **Noul** | Entity resolution — "same entity: yes/no" |
| **Noul** | Decision-model-as-judge — "pass the rubric: yes/no" |
| **Choice** | News sentiment — negative / neutral / positive |
| **Score** | *(not yet used)* — e.g. 1–5 risk severity, full distribution |

One call can carry **up to 10 questions against the same state** —
e.g. "escalate: yes/no" + "which team: billing/technical/other" together.
Questions share the state but are answered **in isolation** from each other.

Score detail: returns the *expected* level, not just argmax — can land
between two levels. Trained so "high" vs. truth "critical" costs less
than predicting "none": distance along the scale matters.

<!-- ~2-3 min, folds into the existing architecture time budget -->

---

## Limitations — type-safe ≠ faithful

- **Option framing changes answers, not confidence.** Swapping which name means "yes" moved one model's AUC .815 → .581 — zero type errors, still validly "an option."
- **Calibration must be verified on your data.** Pooled ECE 0.047 hid a median *per-benchmark* ECE of 0.168. Wrong answers still carried mean confidence .778 in one study.
- **Context can flip a correct answer.** Answer-preserving adversarial additions redirected 61% of initially-correct decisions.
- **Hosted ≠ deterministic.** ~1% of answers changed on identical repeated hosted calls. Self-hosted + pinned weights can be made fully deterministic.

**Always valid ≠ always correct.**

<!-- ~3-4 min -->

---

## Part 4 — Training objective & logic

**Standard LLM training**

- Pretraining: next-token cross-entropy
- RLHF: reward = did a human prefer this *generated continuation*

Both stages optimize for producing good **text**.

<!-- 6-7 min -->

---

## RLCD — Reinforcement Learning for Calibrated Decisions

Reward ≠ "did a human like this continuation"
Reward = **"was the probability actually well-calibrated against outcomes"**

If the model says 0.8 confidence across 1,000 similar cases,
~800 of them should actually resolve true.

*(TypeSafe hasn't published Jev's exact recipe — "RLCD" is their framing.
What follows is the public proper-scoring-rule idea it's presented against.)*

---

## Why RL, when you could just train on labels?

Reward r = c − p (correct indicator minus predicted probability).

Expected RL gradient of r = **exactly half the negative gradient of the Brier score**
— the standard proper-scoring loss.

→ When you **know** the label: supervised training on a proper score
reaches the *same* optimum, with less variance than RL.
→ RL only earns its keep under **bandit feedback** — you see only the
outcome of the action you took, not the label for every option.

A plain 0/1 "was it correct" reward optimizes **accuracy**, not calibration
— and produces **overconfidence**, the opposite of what a threshold needs.

<!-- optional/appendix slide if time is tight -->

---

## Why this matters mechanically

- Generative output → consumed by a **human**, who can read between the lines of a hedge
- Decision output → consumed by **code** — a threshold, a router, an escalation rule

Code can't "read between the lines" of a miscalibrated number.
**Calibration has to be trained in directly.**

---

## Part 5 — Use cases

- Fast guardrails/routers inside agent pipelines
- Moderation & transaction flagging, at volume
- A/B and multi-way routing
- Replacing brittle "LLM returns JSON" classification
- **Entity mapping/resolution** — match / no-match + calibrated probability
  *(watch option cardinality: 77-label benchmark — 0.870 vs. 0.425 by model)*
- **Decision-model-as-judge** — calibrated pass/fail instead of LLM-judge prose
  *(as a reranker: BM25+decision model took R@1 33%→58%; LLM scoring tied 231/234 groups)*

<!-- 4-5 min -->

---

## Part 6 — Why it matters for us

We're risk management at a bank. Three direct threads:

**GenAI evaluation / RAG**
Full LLM-as-judge on every response is slow and expensive.
→ Decision-model-as-judge triages: auto-pass/fail the clear cases,
reserve full judge scrutiny for the ambiguous ones.

First pilot: **shadow mode** — run alongside the existing LLM-judge,
don't act on it yet, measure calibration on our labels, fix a threshold.

⚠ Caveat: an LLM fallback judge tends to **repeat the decision model's
own mistakes** on its most-confident wrong answers (96% in one study) —
escalation isn't automatically a safety net; measure the correlation.

<!-- 6-8 min total for this section -->

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
Low-latency, calibrated, auditable decisioning is the core job description
here, not a side benefit. But the fit cuts both ways:

- **No explanations** — a probability, not a reason. Auditability comes from
  decomposing into atomic questions + a versioned rubric in *our* code.
- **Hosted Jev is a governance-relevant black box** — undisclosed internals,
  single region, ~1% nondeterminism. **Self-hosted** is the realistic route
  for anything touching a regulated decision.
- **Calibration drifts** — monitor per segment, not pooled; recalibrate on a schedule.
- **Human review budgets work, but must be measured** — 0.8% review reached
  90% precision in one case; per-field budgets ranged up to 72%.

**Typed, thresholdable output is a better fit for model-risk governance
than free-text LLM output — if evaluated with the same rigor as any
new scoring model.**

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

## Sources

**Primary (arXiv):**
- Tang & Zheng — *Typed Decision Models: An Early Evidence Audit*, 2609.32160
- Duan, Lin, Wang, Xiang et al. — *General Decision Models (JEVal)*, 2610.03935
- Antunes et al. — *Narrow Decision Models*, 2610.25553
- dos Santos · Wu & Lim · Cao et al. · Wan, Jünger et al. — see written doc for full list

**Vendor/secondary:**
- sanity.io/glossary · openrouter.ai/blog/insights — *What is Jev?*
- github.com/andyrewlee/awesome-system-one
- huggingface.co/juspay/xor · github.com/Torchcast-AI/torchcast-decision-4b
