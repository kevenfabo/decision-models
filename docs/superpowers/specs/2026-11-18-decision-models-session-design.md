# Design: Decision Models Deep-Dive Session (Team Eng/Research Initiative, Edition #1)

**Date:** 2026-11-18
**Status:** Approved for implementation planning

## 1. Overview

**Goal:** Give the data science team a shared mental model of "decision models" (System 1 models such as TypeSafe AI's Jev) — why they exist, how they differ architecturally and in training from generative LLMs, and where they could plausibly plug into risk-management AI work (graph ML, GraphRAG, GenAI evaluation, RAG).

**Context:** This is the first edition of a new recurring team engineering/research initiative. The team sits in the risk management department of a bank; the presenter's own work spans graph ML, GraphRAG, GenAI evaluation, and RAG. This spec covers only the content and structure of this first session — series cadence, topic selection, and rotating presenters are out of scope and will be addressed separately.

**Audience:** Mixed-background data science team — some strong ML/research depth, others more applied/engineering-focused. Content should motivate before going deep into mechanics, with brief refreshers where needed rather than a full LLM-101 preamble.

**Format:** Live session, 45-60 minutes total: ~35-40 minutes of conceptual/architecture/training content, ~10-15 minutes for a live demo plus Q&A/discussion.

**Terminology note:** "Jev" refers to TypeSafe AI's proprietary System 1 decision model (released 2026-09-15), not JEPA. Jev is not open source; the open-source ecosystem referenced throughout (e.g., Torchcast Decision, Juspay's `xor`) consists of independent alternatives, not forks of Jev.

## 2. Deliverables

1. **Slide deck** — Markdown-based (Marp or reveal.js format), covering the conceptual narrative end to end.
2. **Live demo notebook** — Jupyter notebook, anchored on entity resolution and decision-model-as-judge scenarios (see Section 4).
3. **Written deep-dive doc** — standalone Markdown document covering the same material as the deck in prose form, as a leave-behind for anyone who missed the session or wants more depth.

All three are version-controlled files living in this repository (see Section 5 for file structure).

## 3. Content Outline

| # | Segment | Time | Key content |
|---|---|---|---|
| 1 | Why decision models exist | 4-5 min | Kahneman's System 1/2 distinction (*Thinking, Fast and Slow*). The gap: organizations make huge volumes of fast, bounded, calibrated choices (approve/deny, route, flag) — a System 1 task. LLMs, even when forced into a classification prompt, are generative and token-by-token, carrying System 2-shaped cost for a System 1 job. |
| 2 | The challenge it solves | 4-5 min | Latency, cost, calibration, hallucinated rationale, brittle structured-output parsing. Decision models generate no tokens, return typed output (choice/score/probability) with a calibrated probability in roughly 70-500ms — contrasted against "ask an LLM to emit JSON," which is slower, prompt-brittle, and often miscalibrated. |
| 3 | Architecture & variants, vs. traditional LLM architecture | 9-10 min | Typed input (state + question schema) → shared encoder/backbone → classification/scoring head, single parallel pass, no autoregressive decode loop. Contrast with decoder-only LLMs: next-token prediction via a generative decode loop. Key insight to land: most "decision models" are a head-swap/fine-tune on an existing LLM backbone rather than a wholly new architecture family. Landscape: Jev (proprietary) vs. open alternatives (Torchcast Decision, Juspay's `xor` built on a Qwen3.6-35B-A3B backbone). |
| 4 | Training objective & logic | 6-7 min | RLCD — Reinforcement Learning for Calibrated Decisions. Reward is shaped around calibration, not just accuracy, because the output probability is consumed by downstream automated systems rather than read as prose by a human. Contrast with standard LLM training: next-token cross-entropy pretraining plus RLHF (preference optimization over generated text) — a different reward signal shape optimizing a different thing. |
| 5 | Use cases | 4-5 min | Fast guardrails/routers in agent pipelines, content/transaction moderation, A/B routing, structured classification replacing brittle "LLM returns JSON" patterns, score-based gating, **entity mapping/resolution** (typed match/no-match decisions between candidate entity records), and **decision-model-as-judge** (calibrated pass/fail or score in place of an LLM-as-judge prose rationale in eval pipelines). |
| 6 | Why it matters for us | 6-8 min | Concrete ties to the team's actual work: (a) GenAI evaluation/RAG — calibrated decision models as a cheap triage layer ahead of full LLM-as-judge evaluation; (b) graph ML/entity resolution — decision models as a complementary fast-path scorer/matcher reading unstructured case text or record pairs, the same typed-decision paradigm risk teams already use; (c) risk/compliance specifically — low-latency, calibrated, auditable decisioning is the department's core function, making this a natural technical bridge between LLM-adjacent capabilities and the regulated scoring requirements risk/compliance already impose. |

Segments 1-6 total 33-40 min, fitting the ~35-40 min content budget from Section 1. The closing demo and discussion/Q&A (see Section 4) share the remaining 10-15 min window — they are not additional time on top of it.

## 4. Demo Design

**Primary scenario — entity resolution.** State = two candidate entity records (e.g., customer/counterparty records with name/address/ID variants). Question = "same entity: yes/no." The open-source decision model returns a typed match decision with a calibrated probability in well under a second.

**Secondary scenario — decision-model-as-judge.** Reusing the same loaded model, swap the question schema to "does this RAG answer satisfy the rubric: pass/fail," with state = `{question, retrieved context, generated answer}`. This demonstrates the same typed-decision pattern applied to evaluation, without re-running heavy setup.

**Contrast moment.** For the entity-resolution scenario, also run the same decision by prompting a general-purpose LLM for JSON output, and time both paths side by side — making the latency/calibration gap tangible live rather than asserted on a slide.

**Model choice.** Not locked in by this spec — selecting between candidates (Juspay's `xor`, Torchcast Decision, or another open alternative) is an implementation-phase task. Selection criteria: runnable locally or via a free tier within a notebook, clearly exhibits typed input → calibrated output behavior, reasonably fast to set up.

**Live execution.** Heavy setup (model loading, auth, dependency installation) is pre-run before the session starts. The live portion re-runs only the input → output → timing comparison cells, taking roughly 7-8 minutes, leaving 3-7 minutes for open discussion ("where in our stack could a decision model replace an LLM-as-judge call or a hand-rolled classifier?") within the shared 10-15 minute closing window.

## 5. File Structure

```
decision-models-deep-dive/
  slides/decision-models-deep-dive.md     (Marp-format deck)
  doc/decision-models-deep-dive.md         (written deep-dive doc)
  demo/decision_model_demo.ipynb           (entity resolution + judge-eval demo)
docs/superpowers/specs/2026-11-18-decision-models-session-design.md  (this spec)
```

## 6. Success Criteria

- Team can explain, in their own words, what a decision model is and how it differs from an LLM call, in both architecture and training objective.
- Team can name at least one place in the GenAI evaluation/RAG, entity resolution, or risk-scoring workflow where a decision model could plausibly be tried.
- Live demo runs without failure and makes the latency/calibration contrast visible, not just asserted.
- Deck, doc, and notebook are complete, versioned, and reusable for teammates who missed the session.
