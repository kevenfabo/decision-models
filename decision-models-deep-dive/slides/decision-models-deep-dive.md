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

**A decision model is usually not a new architecture family.**

It's a fine-tune (often head-swapped) on an existing open LLM backbone:

- `xor` (Juspay) — post-trained **Qwen3.6-35B-A3B**
- Torchcast Decision — **Qwen3.5-4B / Gemma-4-12B / Qwen3.8-27B**

The innovation lives in the output stage + training objective — not a rethink of the transformer. Your existing LLM mental model transfers directly.

---

## Side by side

| | Traditional LLM | Decision model |
|---|---|---|
| Output | Token sequence | Typed value + calibrated probability |
| Inference | Autoregressive decode loop | Single parallel pass |
| Latency | Seconds, scales with output | 70–500ms, no "response length" |
| Backbone | Purpose-trained/fine-tuned decoder | Usually a fine-tune/head-swap of one |
| Consumer of output | A human reading prose | Code thresholding a probability |

---

## The landscape

- **Jev** — proprietary, API-only (`POST /v1/systemone`, `jev-latest`)
- **`xor`** (Juspay) — Apache-2.0, open weights, Hugging Face: `juspay/xor`
- **Torchcast Decision** — 4B/12B/27B family, mixed Apache-2.0 / CC-BY-NC

Open alternatives already **out-benchmark Jev** on community leaderboards.

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
- **Decision-model-as-judge** — calibrated pass/fail instead of LLM-judge prose

<!-- 4-5 min -->

---

## Part 6 — Why it matters for us

We're risk management at a bank. Three direct threads:

**GenAI evaluation / RAG**
Full LLM-as-judge on every response is slow and expensive.
→ Decision-model-as-judge triages: auto-pass/fail the clear cases,
reserve full judge scrutiny for the ambiguous ones.

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
Low-latency, calibrated, auditable decisioning isn't a side benefit here —
it's the core job description.

Decision models: a technically legible bridge between LLM-adjacent
capabilities and the regulated, explainable scoring compliance already
requires of us.

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

- sanity.io/glossary — *What is Jev?*
- openrouter.ai/blog/insights — *What Is Jev?*
- befailproof.ai/jev — *What is TypeSafe AI?*
- github.com/andyrewlee/awesome-system-one
- layer3labs.io/guides — *Jev, Laya, and Open Alternatives*
- huggingface.co/juspay/xor · github.com/juspay/neurolink
- github.com/Torchcast-AI/torchcast-decision-4b
- systemonemodels.org/models
