# Q&A Prep: Decision Models

*Companion to the Edition #1 deck. Grounded in `corpus/DECISION_MODEL.pdf`.*

**How to use this.** Each question has three parts. **Say this** is a 20–40 second answer. **Go deeper** is for when an MLE or MLRE follows up. **Receipts** are the numbers and sources. Citations look like `[A p.11]`, meaning source A on page 11 **of the bundled corpus PDF** (not the original paper's page numbering).

---

## Sources (what's in the corpus PDF)

| Key | Source | PDF pages |
|---|---|---|
| **A** | Tang & Zheng (Northeastern), *Typed Decision Models: An Early Evidence Audit and Evaluation Checklist*, arXiv:2609.32160 — reviews 28 papers from the first 9 days after Jev's release | 1–25 |
| **B** | dos Santos, *Calibrated Decision Models for Autonomous Penetration-Testing Harnesses: JEV and Laya as System One Decision Layers*, arXiv:2609.28940 | 26–46 |
| **C** | Duan, Lin, Wang, Xiang et al. (Shanghai Innovation Institute / Fudan), *General Decision Models: Benchmarking and Insights Beyond Jev* (JEVal benchmark, SII-Decision models), arXiv:2610.03935 | 47–105 |
| **D** | Substack blog, *Can decision models improve RAG retrieval?* (a hands-on test) | 106–121 |
| **E** | Wan, Jünger, …, Weinberger (Cornell / ASAPP), *Retriever-Free Retrieval: Navigating Documents with Decision Models* (DecisionRAG), arXiv:2610.35868 | 122–142 |
| **F** | Dahani (AY Automate), *What Can You Build With Jev*, blog, 19 Sep 2026 | 143–164 |
| **G** | Diogo Almeida, *Introducing System One Models & Jev*, TypeSafe launch post, 15 Sep 2026 | 165–176 |
| **H** | TypeSafe docs, *AI primer* (RLHF / RLVR / RLCD) | 177–181 |
| **I** | Antunes et al. (Softplan), *Narrow Decision Models: A System-1 Architecture for Calibrated, Typed Decisions*, arXiv:2610.25553 | 182–208 (repeated 209–235) |
| **J** | Wu & Lim (NTU), *REFLEX with Jev for Efficient Selective Control in LLM Agents*, arXiv:2609.26532 | 236–254 |
| **K** | Cao et al. (Huawei / Leiden), *SearchJev: A Calibrated System-1 Model for Search Agents*, arXiv:2610.05107 | 255–274 |

---

## ⚠ Fix before presenting: places where the deck and the corpus disagree

A sharp MLRE will catch these. Either fix the slide or be ready to qualify it.

1. **"A decision model is usually a fine-tune (often head-swapped) of an existing decoder LLM."** This is only half right.
   - Open decision models come as **both encoders and decoders**. Laya is built on ModernBERT-large, Open-Jev on DeBERTa-v3-large, and the Softplan NDM on a ModernBERT-family encoder. SII-Decision, SearchJev, JevLite and PixelJev are Qwen decoders `[A p.4–5; I p.183, p.186]`.
   - In the decoder case, the best results come from **keeping** the pretrained LM head and reading it only at the option-label tokens, not from swapping it out. Replacing it with a freshly initialized decision head *lowered* accuracy by several points `[C p.61]`. With matched backbone, data and budget, a dedicated typed head scored the same as the ordinary LM-head readout (.761 vs .761) `[A p.4, p.8]`.
   - Better wording: *"Usually built on an existing pretrained backbone, encoder or decoder. The new parts are the readout, the request layout, and a calibration-oriented training objective."*
2. **"Open alternatives already out-benchmark Jev."** On JEVal (11k instances, 36 datasets), only **SII-Decision-27B** edges past Jev: 79.24% vs 78.38% accuracy. Most open decision models score below Jev, and some far below: Laya 26.3%, NanoJev 25.3% `[C p.52]`. Better wording: *"The best open model now matches Jev on accuracy and is better calibrated."*
3. **`xor` and Torchcast Decision** do not appear anywhere in the corpus. That's fine, but it means you have no paper to back them, only web sources. If you're challenged, pivot to models the corpus does cover: SII-Decision, Laya, SearchJev.
4. **RLCD as "reward = was the probability calibrated against outcomes."** That's TypeSafe's framing. The actual RLCD recipe is **not public** `[A p.4; I p.194]`. Say *"TypeSafe describes it as…"* and then give the public version: training against proper scoring rules (Q4, Q20).
5. **"Calibrated" as a property you get for free.** Independent studies find raw calibration **inconsistent across workloads** `[A p.11, Finding 4]`. Present calibration as a training *goal* that you must *verify on your own data*.
6. **"Can't hallucinate."** The output can't be malformed, but it can still be wrong. Swapping the yes/no names across rubrics moved Jev's AUC from .815 to .581 **with zero type errors** `[A p.13]`. The blog's line for this is "Treating type safety as truth" is a mistake `[F p.154]`.

---

## Six numbers worth memorizing

| | |
|---|---|
| **78.4% vs 79.2%** | Jev vs SII-Decision-27B accuracy on JEVal; ECE 0.053 vs 0.015 `[C p.52]` |
| **0.15–0.45 s** | Typical independently measured latency for one hosted Jev request (vendor claims 70–500 ms) `[A p.11; G p.168]` |
| **~35 ms / ~60 ms** | Laya on a T4 GPU / SII-Decision-4B on an H100: self-hosted, no network hop `[B p.41; C p.62]` |
| **.815 → .581** | Jev AUC after swapping the yes/no option names across rubrics: the robustness warning `[A p.13]` |
| **83.8% → 95.5%** | 8-way routing accuracy when you keep only answers with confidence ≥ 0.90 (70% coverage) `[F p.158]` |
| **5 of 26** | Corpus papers that include a label-log-prob baseline, the one comparison that isolates the interface `[A p.13]` |

---

# Part 1: Your questions

## Q1. What's the difference between a decision model and a classical classifier?

**Say this:** *"Structurally, a decision model is a classifier: closed output space, one forward pass, a softmax over options. The difference is the interface. A classical classifier's labels are fixed at training time and it takes engineered features. A decision model takes free text or JSON as its 'state', and the **label set is an argument you pass at request time**, with a natural-language description of each option. You can ask several isolated questions about the same state in one call, and the probability distribution is the product, not just the argmax."*

**Go deeper.** The evidence audit pins down three defining properties `[A p.2]`:
1. **Closed support at request time.** The output covers exactly the options you supplied, but *"unlike a conventional classifier, the label set need not be fixed at training time."*
2. **Multiple isolated questions per forward pass.** Each question sees the state but not the other answers.
3. **The distribution is the product.** Downstream code thresholds, ranks, routes or defers on it.

| | Classical classifier (LogReg / GBM / fine-tuned BERT head) | Decision model |
|---|---|---|
| Label set | Fixed at training time | Declared per request, read from descriptions |
| Input | Features or text for *one* task | Any text/JSON state, plus instructions |
| New task | Retrain | Write a new question schema (zero-shot) |
| Questions per call | One | Many, isolated from each other, sharing one encoded state |
| Training | Task-specific supervised | General multi-task decision training + calibration objective |

**The honest caveat to volunteer:** if you have enough labels for a task, a task-specific classifier is often still the best model. *"A classifier fine-tuned on each task is the most accurate model on almost every labeled benchmark"* `[I p.183]`. Fine-tuned encoders were *"not significantly worse than an open TDM in ranking… and were faster"* `[A p.13, B3]`. A decision model earns its place when labels are scarce, the taxonomy changes, or you need many different decisions from one model.

**Lineage.** It descends directly from NLI-based zero-shot classifiers (Yin et al. 2019), the Universal Discriminator and GLiClass, which already scored labels supplied at inference time `[C p.49; A p.3]`.

---

## Q2. What makes decision models fast?

**Say this:** *"There's no decode loop. An LLM spends one forward pass per output token. A decision model does one prefill over the input and reads the answer probabilities straight off that pass, so latency depends on input length, not output length. On top of that, many questions share one encoding of the state."*

**Go deeper.** Four sources of speed, roughly in order of importance:
1. **No autoregressive generation.** You pay for one prefill pass and zero decode steps. The decoder readout *"needs one prefill pass and no decoding, its latency depends only on the backbone and the input length"* `[C p.61]`.
2. **The state is encoded once and shared across questions.** In the NDM block-mask design, *"the state is encoded once and attended by every question"*. Because the state attends only to itself, its KV cache can be reused *exactly* across slices or tree-descent steps `[I p.188, p.190]`. Visual Jev's shared-prefix batching cut amortized time per question **8.9×** relative to serial execution at 32 questions per image `[A p.5]`. TypeSafe's docs tell users to send every question in one request because *"extra questions typically add no latency"* `[F p.145]`.
3. **Smaller models.** Laya has 421M parameters; NDM trains a 110M body plus 6.4M new parameters `[B p.40; I p.189]`.
4. **No retries or parsing.** The output is valid by construction, so there's no "re-prompt because the JSON broke" loop.

**Receipts:**
- Vendor: 70–500 ms end-to-end, *"40×–200× faster"* `[G p.168]`. The headline **193.6× faster / 444.6× cheaper** comes from TypeSafe's own workflow evals, and the vendor says it expects these *"are on the higher end of real world gains"* `[G p.172]`.
- Independent: typical hosted medians are 0.15–0.45 s `[A p.11]`. A blogger measured a median of 0.33 s and a max of 1.42 s over 791 calls through OpenRouter `[F p.158]`.
- Self-hosted: Laya takes 32.8–39.5 ms on a T4, and 10 batched questions take 72 ms total `[B p.41]`. SII-Decision-4B takes about 60 ms on an H100 `[C p.62]`. SearchJev is **5.2–5.3×** faster than a same-size Qwen3.5 emitting JSON `[K p.255]`.

**Watch out:**
- Some reported speedups come from serving design, not the readout. *"Repeated-request caching shrinks the median paired latency difference to under a millisecond"* in one edge study `[A p.11]`.
- The same prefix-sharing trick works for *any* label-probability readout `[A p.5]`.
- Against **one-token constrained generation**, the readout is only **1.03×** faster `[A p.9, Zhou et al.]`. The big win is over *free-form* generation.

---

## Q3. Who created Jev, and what's their background?

**Say this:** *"Jev is from TypeSafe AI, founded by **Diogo Almeida**. He was at OpenAI, where he worked on the instruction-following methods behind ChatGPT, and he's a co-author of the InstructGPT paper. He left convinced that chat models were missing something for automation, spent two years in stealth, and launched Jev on 15 September 2026 with a $40M seed round."*

**Go deeper:**
- In his own words: *"At OpenAI, I helped build the methods that made language models useful at following instructions and talking with people. That work ended up as the research behind ChatGPT."* His driving question for four years: *"Models have been superhuman at chat for years, so where is all the automation?"* `[G p.166]`
- TypeSafe's docs describe RLHF as *"co-invented by Diogo Almeida, cofounder of TypeSafe"* `[H p.178]`. He appears in the author lists of InstructGPT (Ouyang et al., 2022) and the GPT-4 technical report `[A p.19; C p.64]`.
- After *"two years in stealth"*, TypeSafe launched with *"a new model architecture, parallel sampler… and training method we call RLCD"* `[G p.166]`. The $40M seed figure is from `[F p.144]`.
- **The name.** Jev is named after **William Stanley Jevons**, of the Jevons paradox: as efficiency gains cut the cost of a resource, demand for it rises. *"Every order of magnitude drop in the cost of intelligence unlocks orders of magnitude more use cases."* "System One" comes from Kahneman's *Thinking, Fast and Slow* `[G p.175]`.
- Company thesis: *"Machine Native Intelligence"* and *"Building prod, not God."* They expect automation to be *"99% machine-to-machine interactions and 1% human"* `[H p.177]`.

**Watch out:** "co-invented RLHF" is the vendor's framing. RLHF as a technique predates InstructGPT (for example Christiano et al., 2017). The defensible version is: *"He was part of the OpenAI team that applied RLHF to make InstructGPT/ChatGPT. He's an author on that paper."*

---

## Q4. What are the theoretical foundations of decision models?

**Say this:** *"There are four main roots. First, **label-probability classification**: reading an LM's probability over a fixed set of answer tokens. Second, **proper scoring rules**: losses like log-loss and Brier whose optimum is the true probability, which is what 'trained to be calibrated' means. Third, **selective classification and learning-to-defer**: act when confident, escalate otherwise. Fourth, the **System 1 / System 2** framing from Kahneman, which motivates the split. Very little of it is new. What's new is putting these pieces together behind one general-purpose interface."*

**Go deeper.** The audit maps each part of the model to its ancestor `[A p.3, Table 1]`:

| Component | Ancestor | What decision models change |
|---|---|---|
| Score a label set by LM probability | Cloze/verbalizer classification (PET, LM-BFF), GPT-3 likelihood scoring | Options declared per request, not fixed verbalizers |
| Guaranteed valid output | Grammar-constrained decoding | Return a renormalized distribution, no token generation |
| Fine-tuned label-token head | Yes/no rerankers (monoT5, Qwen3-Reranker), generative verifiers | Any option set, many questions per call |
| Labels supplied at runtime | NLI zero-shot classifiers, GLiClass | Presented as a general decision interface |
| Rubric-conditioned judging | Prometheus, G-Eval | A rubric attached to each option |
| Confidence-based escalation | Reject option (Chow), LLM cascades (FrugalGPT) | A cheap first stage with a native probability |
| Calibration training | RL with proper-scoring rewards | Calibration is part of the product objective (RLCD) |

**The core math, in three layers:**
1. **Proper scoring rules.** A loss is *strictly proper* when its unique minimum is the true distribution (Gneiting & Raftery 2007). The two common ones:
   - **Brier score**: BS = (1/N) Σ (pᵢ − oᵢ)². A model minimizing it is pushed to output p = 0.7 when the event happens 70% of the time `[B p.27]`.
   - **Log-loss.** NDM trains every primitive on log-loss plus a quadratic term. For ordinal **Score** questions it adds the **ranked probability score**, which penalizes probability mass far from the true level more than mass next to it `[I p.190]`.
2. **Why RL and supervised training meet here.** This is the elegant part, and MLREs love it. Take the open RLCD-style reward r_a = c_a − p_a, where c_a ∈ {0,1} says whether option a was correct. The expected REINFORCE gradient is then **exactly half the negative gradient of the Brier score** `[I p.194]`:

   𝔼ₐ∼ₚ[rₐ ∇log pₐ] = Σₐ (cₐ − pₐ) ∇pₐ = −½ ∇ Σₐ (pₐ − cₐ)²

   So when labels are known, plain supervised training on proper scores reaches the same optimum **without the sampling variance**. RL only earns its keep when you observe only the outcome of the action you took (bandit feedback). A plain 0/1 reward, as in verifiable-reward RL, optimizes **accuracy** instead, and *"the resulting model is expected to be overconfident"* `[I p.194]`.
3. **Post-hoc calibration and selective prediction.** Temperature scaling (Guo et al. 2017) fits one temperature per primitive on held-out data `[I p.191; K p.259]`. Selective classification trades coverage for risk. Conformal risk control and learn-then-test can turn the empirical threshold into a finite-sample guarantee, but only on real, exchangeable calibration data `[I p.191]`.

**Other foundations that show up:**
- **Deep Sets / permutation equivariance**, behind NDM's set head that compares options `[I p.188]`.
- **Bradley–Terry pairwise preference models**, behind reward-model heads `[A p.3]`.
- **Dual-process theory**: Kahneman, and *"intuition grows out of practiced deliberation"* (Kahneman & Klein 2009). This is the motivation for distilling reasoning into a single-pass readout `[C p.48]`.

---

## Q5. Are they encoder or decoder architectures?

**Say this:** *"Both, and we don't know which Jev is. TypeSafe hasn't published the architecture. It only says Jev is 'neither small nor an LLM'. The open models split into two camps. Encoder ones, like Laya on ModernBERT, are small and fast and can guarantee option-order invariance by construction. Decoder ones, like SII-Decision on Qwen, read the first-answer-token distribution of an LLM and inherit its knowledge. Right now the decoders win on accuracy."*

**Go deeper:**

| | Encoder-based | Decoder-based |
|---|---|---|
| Examples | **Laya** (ModernBERT-large, 421M), **Open-Jev** (DeBERTa-v3-large), **NDM-Block** (ModernBERT-family, multilingual) | **SII-Decision-4B/27B** (Qwen3.5-4B, Qwen3.8-27B), **SearchJev** (Qwen3.5-0.8B/4B), **JevLite** (Qwen3-4B LoRA), **PixelJev**, **Visual Jev** |
| How it reads the decision | A marker token per option, or a pooled option representation, fed to a head | The next-token softmax at the answer position, restricted to the option label tokens |
| Strengths | Bidirectional within a block; cheap; block masks give **invariance by construction**; easy to self-host | Inherits LLM world knowledge; **zero new parameters** needed; prefix KV-caching |
| Weaknesses | Short context (Laya: 512 tokens English); weaker on knowledge-heavy tasks | Larger; option-order and position bias unless you mitigate it |
| JEVal accuracy | Laya 26.3% | SII-Decision-27B 79.2% |

Sources: `[A p.4–5; B p.40–41; C p.52; I p.186; K p.259]`.

- **Why Softplan chose an encoder** `[I p.186]`: *"the contract never generates text; within a block, reading in both directions is what the task requires; and an encoder of this size costs several times less per token to train than a decoder of comparable quality."*
- **Why the SII team chose a decoder** `[C p.61]`: *"an LLM already contains a decision model."* Even untrained, the first-token readout of Qwen3.8-27B *"comes within a few points of Jev."*

**The trap:** "non-autoregressive" does **not** mean "encoder". A decoder that does one prefill and reads one position is also non-autoregressive at the output. The pentest paper calls System One models *"non-autoregressive"* without implying encoder-only `[B p.28]`.

**Watch out:** don't over-read the Laya-vs-SII gap. It mixes up *architecture* with *scale and training*. The audit's conclusion is that *"task training and backbone quality matter more for accuracy than the choice between a typed head and a conventional label-probability readout"* `[A p.6]`.

---

## Q6. Any difference with BERT-based models?

**Say this:** *"Some decision models **are** BERT-based. Laya is ModernBERT and Open-Jev is DeBERTa. So the question is really: what does a decision model add on top of a fine-tuned BERT classifier? Three things: an open label set described in natural language at request time, several isolated questions over one shared state, and a training objective aimed at calibrated probabilities rather than just accuracy."*

**Go deeper:**
- **A classic BERT classifier** puts a [CLS] head over N fixed labels. A new label means retraining.
- **A BERT zero-shot NLI classifier** scores "premise = text, hypothesis = label description" one pair at a time. That's the **pairwise cross-encoder** pattern, and its cost grows with the number of options.
- **A BERT-based decision model** (NDM-Block) packs the state, every question and every option into **one sequence** under a block attention mask, so it's one pass instead of Σ K_i passes `[I p.190, Table 1]`:

| | NDM-Block | NDM-Pair (cross-encoder) | Decoder scoring |
|---|---|---|---|
| Forward passes per request | 1 | Σᵢ Kᵢ | Σᵢ Kᵢ (with prefix cache: 1 shared + short per-option passes) |
| Options compared to each other | yes (set head) | no | no |
| Option-order invariance | by construction | by construction | no |
| Question isolation | by construction | by construction | depends on the prompt |

  At equal compute, the block design reached a **lower loss than the pairwise cross-encoder** `[I p.182]`.

**The readout design matters for robustness.** Sun et al. tested both BERT-family readouts. A DeBERTa model that **pools over the full option text** flipped about 4× less often under yes/no name swaps than a ModernBERT model that scores **a single marker token**: 19.5% vs 76.9% `[A p.4]`.

**BERT-era weaknesses that carry over:** short context (Laya: 512 tokens English, 1024 multilingual `[B p.41]`) and weaker world knowledge. That's why the decoder-based models lead on knowledge-heavy benchmarks.

---

## Q6a. High level, how do you actually go from a base LLM (e.g. Qwen3.5-4B) to a decision model like Torchcast Decision?

**Say this:** *"It's less 'new architecture' and more 'new training recipe plus a trick for reading the output,' layered on an unmodified base model. You take the pretrained LLM as-is, change how you read its output instead of what it is, then post-train it specifically to be good and calibrated at that interface -- often by distilling the model's own slow reasoning into a single fast readout step."*

**Go deeper**, as a four-step walkthrough (verified against SII-Decision, the closest documented analog -- Torchcast itself isn't in the corpus, so treat its specific recipe as inferred, not confirmed):

1. **Start from the plain pretrained LLM, untouched.** No new layers, no swapped-out head at this stage. Same transformer, same weights.
2. **The readout trick -- change how you read the output, not the model.** Format the prompt so it ends right before the answer (`"...Options: (A)... (B)... (C)... The correct answer is ("`), do one forward pass, no generation loop, and restrict the next-token distribution to just the option-label tokens. That distribution *is* the typed decision. The output-embedding rows for "A"/"B"/"C" are literally the decision head -- zero new parameters needed for this part alone.
3. **Post-train for the interface and for calibration, not just accuracy.** Raw zero-shot readout like step 2 is a documented *baseline*, not what makes a good decision model `[A p.8]`. Training does two things: teaches the request format (state + a question schema with options described at request time, across many tasks, not fixed at training time), and optimizes for calibration via a proper scoring rule instead of plain right/wrong -- the RLCD idea (see Q20).
4. **The specific trick SII-Decision uses -- Reasoning-to-Readout Self-Distillation `[C p.60-62]`:** let the model reason with full chain-of-thought, take the answer distribution it lands on *at the end* of that reasoning as the target, then train the same model (reasoning off) to match that distribution at the very first answer token. No gold labels needed. It's literally training System 1 to imitate what System 2 would have concluded -- the exact metaphor the session opens with.

**Receipts:**
- Replacing the LM head with a freshly-initialized decision head *lowered* accuracy by several points in controlled tests; a dedicated typed head scored the same as just reading the ordinary LM head (.761 vs .761) `[C p.61; A p.8]`. The gain is the training objective, not architectural surgery.
- Distillation recovers most of thinking's gain on label-selection/weighing tasks, but a quarter or less on exact-computation tasks (GPQA, MMLU-Pro) -- it transfers the weighing of options, not intermediate computation `[C p.63]`.

**Watch out:** Torchcast Decision is described by its own docs as "post-trained for typed decisions" and almost certainly follows this general shape, but it doesn't appear in the audited literature `[A p.3]` -- don't present its specific recipe as verified the way SII-Decision's is.

---

## Q7. What are typical architecture design examples?

**Say this:** *"There are five model-level designs and three system-level patterns. At the model level, the most common is the **first-token readout** on a decoder LLM. The most principled is Softplan's **prefix-tree encoder with a block attention mask**, which gives you invariances by construction. At the system level, the patterns are confidence-gated cascades, hierarchical tree descent, and composing decisions through a rubric."*

### Model-level designs

**(a) First-token decision readout (decoder).** Used by SII-Decision, SearchJev, JevLite, PixelJev `[C p.60–61; K p.259]`.
```
"State: …  Instruction: …  Options: (A) … (B) … (C) …  The correct answer is ("
                                                                       │ one prefill
                                    next-token logits at this position ▼
                     softmax restricted to {A, B, C} label tokens → p(option)
   Choice = argmax   ·   Noul = p(true)   ·   Score = Σ_k v_k · p(k)  (expected level)
```
- No new parameters: the output-embedding rows of the label tokens *are* the decision head.
- Labels are kept to single tokens (A…Z, AA…), which supports up to 255 options.
- The trained version adds **Reasoning-to-Readout Self-Distillation**. The model reasons with thinking on, its readout distribution at the *end* of the reasoning chain becomes the teacher, and the same model without reasoning learns to match it at the first answer token. A KL anchor to the original model limits forgetting. No gold labels are needed `[C p.61–62]`.

**(b) Marker-token encoder (Laya).** A mask marker sits at each option, the options share positions and can't see each other, and the marker embeddings are scored. Built on ModernBERT-large `[A p.4–5; I p.188]`.

**(c) Pooled-option encoder (Open-Jev, DeBERTa-v3-large).** Takes the mean representation over each option's name and definition. More robust to option-name swaps than (b) `[A p.4]`.

**(d) NDM-Block (Softplan): the reference "principled" design** `[I p.185–189]`.
```
prefix tree:   State ─┬─ Q1 (Choice) ─┬─ O11
                      │               └─ O12      siblings share logical positions,
                      ├─ Q2 (Score)  ─── levels   attend only to self + ancestors
                      └─ Q3 (Noul + yes/no criteria)
packed into ONE sequence → bidirectional encoder under a block attention mask
→ one vector per marker → heads:
   • Set head: a small transformer over one question's options (no positions)
                → compares candidates, stays permutation-equivariant
   • Choice/Score: relative logits → softmax
   • Adequacy: an absolute fitness logit per option → abstention that doesn't depend on siblings
   • Noul: [h_Q; h_S] → sigmoid
```
- **Guaranteed properties (with proof sketches):**
  - Prop. 1, option-permutation invariance;
  - Prop. 2, question isolation;
  - Prop. 3, slicing and paths give identical outputs. This is why the state's KV cache can be reused *exactly*.
- **Size:** 6.4M new parameters on top of a 110M trained body and a frozen 197M embedding table.
- **Explicit "none of these" option.** This is how the model abstains, and it also gives a tree descent its stop rule.

**(e) Answer-slot scoring against output embeddings** (this-that-model-1.0). A 1.88B hybrid linear-attention decoder whose answer-slot hidden state is scored against the output-embedding rows of single-token labels. Trained with cross-entropy plus Brier `[A p.5]`.

### System-level patterns

1. **Confidence-gated cascade (the dominant pattern).** Accept the decision model's answer above a threshold τ and escalate the rest to an LLM or a human.
   - REFLEX: 95% task success with 72.7% fewer strong-model calls `[J p.236]`.
   - SearchJev dual-system agent: BrowseComp-Plus accuracy 45% → 54% with a 3.7–4.7× speedup `[K p.255]`.
   - The pentest harness uses a 5-level cascade: deterministic validator → Noul → Choice + Score → LLM → human `[B p.40]`.
2. **Hierarchical descent.** Route through a tree with one Choice per node plus "none of these", using greedy or beam search.
   - Org-chart routing in NDM `[I p.191–192]`.
   - **DecisionRAG** navigates topics → docs → passages with no first-stage retriever. It beats dense retrieval plus an 8B reranker by +13.8 R@1 on MultiDoc2Dial `[E p.122]`.
3. **Composed decisions.** Ask atomic questions, then push their **full distributions** through a frozen, versioned rubric in code:
   P(u = k) = Σ P(a) P(t) P(σ) · 1[R(a, t, σ, d) = k]
   - Auditable, and consistent across answers by construction `[I p.193]`.
   - **Highly relevant for a bank** (see Q23).

---

# Part 2: What a good MLE or MLRE will probably ask you

## Evidence and evaluation

### Q8. Is the typed readout itself actually better, or is it just a good model behind a nice API?

**Say this:** *"So far there's no evidence that the readout by itself adds accuracy. The few studies that compared against a label-log-probability baseline on the same backbone found parity. The real gains are latency, cost, guaranteed-valid output, and the multi-question interface."*

**Receipts `[A p.8, Finding 1]`:**
- **Ren et al.:** label-token readout of a generative fine-tune AUROC .950 vs a typed ensemble .974, not significant.
- **Yu & Yao:** typed head .761 = LM head .761.
- **Zhou et al.:** with weights fixed, restricted option-logit readout **matches greedy generation exactly whenever the generation is valid**. Its only gain comes from fixing invalid generations.
- **Only 5 of 26** applicable papers even included that baseline `[A p.13]`.

**What would settle it** `[A p.14]`: matched-cost open backbones under three conditions: (1) no task training, (2) task fine-tuning with ordinary label probabilities, (3) calibration-oriented decision training.

### Q9. How calibrated is Jev, really? How would you verify it on our data?

**Say this:** *"Better than most LLMs' verbalized confidence, but inconsistent across workloads. It usually ranks examples well, but raw probabilities can be off for a given task. So I'd treat it like any scoring model: measure ECE, Brier and reliability diagrams on held-out data, recalibrate with temperature scaling, and fix thresholds before testing."*

**Receipts:**
- **JEVal:** Jev ECE 0.053, versus 0.10–0.25 for generative LLMs. SII-Decision-27B gets 0.015 `[C p.52]`.
- **Crash-narrative coding:** prevalence overstated (4.8% vs 2.7%), calibration slope 1.63; held-out recalibration cut calibration error **3.3×**. The two-decimal output grid limits resolution for labels rarer than 1% `[A p.11]`.
- **Pooled vs per-benchmark:** pooled ECE 0.047 but **median per-benchmark ECE 0.168** `[A p.11]`. Pooling hides miscalibration.
- **High confidence ≠ correct:** wrong BFCL tool calls had mean confidence .778 `[A p.11]`. Decision models *"can often identify the most likely outcome while substantially overstating its probability"* `[C p.47]`.

**What to say about metrics:** report ECE **and** Brier (Brier mixes calibration with resolution), plus a reliability diagram, accuracy on the highest-confidence slice, and results before and after recalibration (checklist C1/C2 `[A p.13]`).

### Q10. Where do they fail?

**Say this:** *"Anything that needs intermediate computation or specialist knowledge, and long-horizon control where small errors compound. They're strongest when the answer can be read straight off the evidence in the input."*

**Receipts:**
- **Specialist domains** (medicine, finance, law): they fall behind thinking LLMs `[C p.48]`.
- **Exact computation:** on GPQA and MMLU-Pro, distillation recovers *"a quarter of [thinking's] gain or less"*. *"A single forward pass does not reproduce that computation even when trained on its outcome"* `[C p.63]`.
- **Long-horizon agents (τ-bench):** Jev + LLM succeeds on **66.1% vs 77.6%** for the baseline, even though median episode time drops from 267 s to 161 s. Single-step errors compound into the slow tail `[C p.58]`.
- **Whether to act at all:** BFCL function *selection* is 98.4% accurate, but deciding *whether to call any function* is only **52.0%** `[J p.236]`.
- **High cardinality:** Banking77 (77 labels): Jev 0.870, Laya 0.425 `[B p.40]`.

### Q11. How robust are they? (option names, ordering, adversarial context)

**Say this:** *"This is the weakest point. A decision model can follow the option **name** instead of the rubric attached to it, and answer-preserving context can flip its decisions. Type safety guarantees a valid answer, not a faithful one."*

**Receipts:**
- **Option-name polarity** (Sun et al., *"Type-Safe Is Not Error-Free"*): swapping yes/no names across rubrics moved AUC .815 → .581 with zero type errors `[A p.13]`. NDM puts it starkly: attaching yes/no to the opposite rubrics *"changes about half of the hosted model's answers"* `[I p.183]`.
- **Context flipping (JevOut):** optimized answer-preserving additions redirect **61.4%** of initially correct decisions; a neutral addition redirects 2.2% `[A p.11]`.
- **Prompt injection:** score-guided injection raises attack success from 1.8% to 3.5% `[A p.11]`.
- **Changing the primitive:** asking via Score instead of Choice moved graded accuracy by up to 10.5 points `[A p.13]`.

**Mitigations to mention:**
- Use neutral option names and test with swapped names (checklist T3).
- Permute option order during training (SearchJev does this `[K p.259]`).
- Use an architecture with invariance by construction (NDM-Block).

### Q12. Is it deterministic and reproducible?

**Say this:** *"Hosted Jev isn't fully deterministic: about 1% of answers change on identical repeats. A self-hosted model with pinned weights can be. For a model-risk function that settles it: pin versions and log everything."*

**Receipts:**
- **Repeat-run changes:** 4 of 510 labels changed on an identical repeat, and at most 1.33% of decisions changed between two identical runs on 300 questions `[A p.4]`.
- **Probability grid:** probabilities sit on a **two-decimal grid** from 0.01 to 1.00; no zero was observed in 2.4M answers `[A p.4]`.
- **Version coverage:** every hosted study used essentially one version (jev-1.13), so stability across versions is unknown `[A p.6]`.
- **Self-hosted:** with pinned weights, the invariances hold exactly in exact arithmetic and up to rounding in floating point. NDM sets the tolerance in advance and tests against it `[I p.189]`.

### Q13. How would you benchmark one fairly?

**Say this:** *"Against three baselines, not just a JSON-prompted LLM. Those are: a label-log-prob readout on a comparable open model, a cheap generative cascade, and a task-specific classifier if we have labels. Plus tests that aren't saturated, repeated runs, and held-out calibration."*

**The audit's minimal reporting card** `[A p.14]`:
- exact model version and run date;
- test difficulty or spread between models;
- one ordinary-LM readout baseline;
- one cheap generative baseline;
- intervals from repeated runs;
- held-out calibration;
- for any cascade, a **frozen** threshold tested on a new workload.

**Why the vendor's own comparison is weak:** the reference answer is the average of two frontier models, the workflows were written by TypeSafe's own team, and the LLMs were run through TypeSafe's wrapper with OpenRouter routing. The vendor lists these caveats itself `[G p.170, p.172]`. Many reported "parity" results come from **ceiling-level tests** (100/100, 63/63) `[A p.8]`.

## Systems and engineering

### Q14. Does confidence-based escalation actually work?

**Say this:** *"Often, yes. It's the most common deployment pattern and the savings are real. But it fails when the fallback model makes the same mistakes, and a cheap generative cascade is a serious competitor."*

**Receipts:**
- **Judge cascade:** 92.5 vs GPT-6's 93.1 **at 57% of the fee** `[A p.11]`.
- **Annotation cascades:** match the LLM alone at 27–56% of its cost `[A p.11]`.
- **Correlated errors:** on Jev's 12 most confident errors per panel, LLM judges **repeat Jev's wrong answer 96% of the time**, so escalation recovers at most about 1.5 points `[A p.12]`.
- **Generative cascade comparison:** on the external τ²-bench, a cheap generative cascade was cheaper and no less accurate, and the REFLEX authors **withdrew their superiority claim** `[A p.12; J p.237]`.
- **Thresholds must be frozen:** temperatures fitted on small pilot sets transfer unevenly `[A p.11]`.

### Q15. How do they handle many options or hierarchies?

**Say this:** *"Jev caps options at 255. Above that you go hierarchical: a tree of Choice questions, each with a 'none of these' option, searched greedily or with a beam. That's exactly how DecisionRAG retrieves without a vector index."*

**Receipts:**
- **Jev's two-stage mode:** 255-option cap; for high-cardinality choices it *"scores independently then makes an explicit choice"* `[G p.174]`. This hasn't been evaluated independently `[A p.6]`.
- **Path scoring:** product (joint probability), geometric mean (depth-insensitive), or minimum (weakest link). The aggregate doubles as the routing confidence `[I p.192–193]`.
- **DecisionRAG:** retrieval takes 0.7–1.8 s per query over 2.68M Wikipedia paragraphs, with **34–145× lower latency** than an LLM generating the same navigation decisions token by token. Judging candidates in a shared context adds up to +7.7 points over judging them independently `[E p.122–123]`.

### Q16. Can they reason? What about chain-of-thought?

**Say this:** *"Not at inference. There's no generated intermediate computation, by design. You can distill reasoning into the readout, and that works for *weighing* evidence but not for multi-step computation. Anything that needs computation should be decomposed into atomic questions or sent to System 2."*

**Receipts:** Reasoning-to-Readout Self-Distillation recovers most of thinking's gain on label-selection tasks (CLINC150, GoEmotions, AgentHarm) but a quarter or less on GPQA and MMLU-Pro `[C p.63]`. In the authors' words: *"Distillation transfers the weighing of options, not intermediate computation."*

### Q17. Are the latency claims real?

**Say this:** *"Directionally yes, but always ask what's included. Hosted Jev's 0.15–0.45 s includes the network. Self-hosted open models run at 30–60 ms. And some reported wins disappear once the LLM baseline uses caching or one-token constrained decoding."*

**What to ask of any latency claim** (checklist C4 `[A p.13]`):
- with and without caching;
- serving location;
- concurrency;
- **cost per correct outcome**, not cost per call.

Only 3 of 20 papers reported latency both with and without caching. Note that TypeSafe served from the US West Coast at launch `[G p.169]` and is single-region `[F p.154]`.

### Q18. Can we build or fine-tune one in-house?

**Say this:** *"Yes, and there are published recipes. The cheapest is a first-token readout on an open model we already trust, optionally self-distilled from its own reasoning, with no labels needed. For one narrow, high-volume decision with in-house data, Softplan's narrow decision model approach is the blueprint: a small encoder, a curriculum, and proper-scoring training."*

**Recipes:**
- **SII-Decision** `[C p.61–62]`:
  - First-token readout, then Reasoning-to-Readout Self-Distillation.
  - Training data is *any* question with options, with no gold labels (39,279 questions for the 27B).
  - Gains plateau beyond that, because self-generated questions are too easy for the model.
- **NDM curriculum** `[I p.194]`:
  - Stage 0 trains the heads with the backbone frozen.
  - Stage 1 teaches the decision format on public data.
  - Stage 2a teaches *reading* option descriptions on **fictional org charts**, so the model can't memorize them.
  - Stage 2b moves to the target domain, keeping some nodes held out.
  - A first instance agreed with the production LLM classifier on **91%** of held-out documents `[I p.182]`.
- **Laya** `[B p.41]`: Apache-2.0, fine-tunable with proper-scoring RL, self-hosted.
- **SearchJev's SLCD** `[K p.259]`: soft labels plus proper scoring losses plus a temperature per output type. Soft labels beat one-hot labels on held-out accuracy.

**Narrow vs general** `[I p.183]`: narrow buys data residency, your own taxonomy and language, version control, and inspectable guarantees. It costs one training pipeline per domain and less coverage.

### Q19. Why not just fine-tune a BERT or train a GBM?

**Say this:** *"If we have labels and a stable taxonomy, we probably should. That's still the strongest baseline. A decision model wins when labels are scarce, the taxonomy changes often, the input is messy free text, or we need dozens of different decisions without maintaining dozens of models."* (This repeats the Q1 caveat. Lean into it: it shows you're not selling.)

### Q20. How does RLCD differ from RLHF and RLVR? And why use RL at all?

**Say this:** *"They optimize different things. RLHF optimizes human preference, which rewards confident-sounding answers and sycophancy. RLVR optimizes verifiable correctness, which is accuracy, so you get overconfidence. RLCD optimizes a proper scoring rule, so honest probabilities are the optimum. And if you know the labels, supervised proper-scoring loss reaches the same optimum with lower variance. RL is only needed when you see just the outcome of the action you took."*

**Receipts:**
- **The three objectives** `[G p.167; H p.178–180]`:
  - RLHF → *"human preference"*;
  - RLVR → *"outputs that can be programmatically verified"*;
  - RLCD → *"answers with epistemically honest probabilities."*
- **TypeSafe's critique of RLHF** `[H p.180]`: *"mode dropping"*, a milder form of GAN-style mode collapse.
- **The gradient identity** `[I p.194]`: see Q4.
- **Laya's variant** `[I p.194]`: adds Gaussian noise to the logits and rewards the perturbed distribution with a proper score.
- **Proposed RLHV extension** `[B p.28]`: use deterministic validators as ground truth in a closed loop. It's a proposal, not implemented.

### Q21. How does the Score (ordinal) primitive work?

**Say this:** *"It's a distribution over ordered levels. The model returns the expected level, not the argmax, so it can land between two levels. Training adds the ranked probability score, so predicting 'high' when the truth is 'critical' costs less than predicting 'none'."* `[B p.29; C p.61; I p.190]`

### Q22. What about RAG and retrieval?

**Say this:** *"It's one of the most promising uses, and it's directly relevant to our RAG evaluation. As a reranker on top of BM25, a decision model gave big wins in a small test. And a general LLM asked to score passages in text mostly returned ties."*

**Receipts:**
- **Small hands-on test:** BM25 + decision model R@1 went from 33% to 58% `[D p.119]`.
- **Ties from a general LLM:** Qwen returned tied scores in **231 of 234** candidate groups, versus **0** ties for the decision model `[D p.118]`.
- **DecisionRAG at scale:** see Q15 `[E]`.

## Bank and risk-specific

### Q23. How does this fit model risk management and governance?

**Say this:** *"Better than an LLM in some ways and worse in others. It's better because outputs are typed scores we can monitor, threshold and validate like any scoring model, and decision logic can be composed from atomic answers through a versioned rubric in code. It's worse because hosted Jev is a black box: undisclosed architecture and training data, a single region, an early-stage vendor, about 1% nondeterminism, and no explanations. For regulated decisions, a self-hosted open or narrow model is the realistic route."*

**Points to have ready:**
- **Explainability:** *"Does Jev explain its decisions? No. It returns a probability, not a reason"* `[F p.157]`. You get auditability through decomposition and composition, with the rubric as versioned code `[I p.193]`, and by logging inputs, outputs and thresholds.
- **Vendor and data residency:** hosted in a single region by an early-stage vendor, which *"raises security, data-residency and service-level questions"* `[F p.154]`. Self-hosted Laya or an NDM keeps data on-premises `[B p.41; I p.183]`.
- **Testable guarantees:** NDM's evaluation protocol *"accepts or rejects the mechanism before held-out data is opened"* `[I p.184]`. That maps neatly onto independent model validation.
- **Ongoing monitoring:** calibration drifts when prevalence, workload or model version changes `[A p.14]`. Monitor ECE per segment and recalibrate.
- **Human review budgets:** in crash-narrative coding, a review budget of 0.8% of flagged records reached 90% precision overall, but per-variable budgets went up to 72% `[A p.11]`.

### Q24. What would you pilot first?

**Say this:** *"A shadow-mode triage layer for RAG evaluation. The decision model scores every response, we compare it with our current LLM-as-judge, measure calibration on our labels, fix a threshold on a selection set, and only then let it auto-pass or auto-fail the confident cases. Low risk, easy to measure, and it tests the cascade question directly."* Hit the known pitfalls out loud: correlated errors with the fallback judge `[A p.12]`, and confidence being less informative on adversarial or reference-free inputs `[A p.6]`.

### Q25. The closer: "Is this a new model class, or a rebrand of label-probability classification?"

**Say this:** *"Mechanically it's mostly a repackaging, and the evidence audit says so. Every component has a clear ancestor, and the typed readout hasn't shown an accuracy advantage by itself. But the packaging matters for engineering: valid output by construction, many isolated questions over one state, a calibration-oriented objective, and latency low enough to put a model inside an if-statement. That's why I'd evaluate it, not adopt it on faith."*

`[A p.3]`: *"These systems show that a non-generative decision interface can be useful without implying that the underlying mechanism is novel."*

---

## Rapid-fire: short questions, short answers

| Q | A |
|---|---|
| What's "Noul"? | TypeSafe's name for the yes/no primitive. Returns P(yes) `[A p.2; F p.145]` |
| Is Jev just a smaller LLM? | TypeSafe says it's *"neither small nor an LLM"*. Unverifiable, because the internals are closed `[A p.4]` |
| Open weights for Jev? | None. API-only `[F p.158]` |
| Price? | $0.042 per 1M input tokens; output free `[G p.168]` |
| Max options? | 255 per question `[G p.174]` |
| Context length? | Jev: not published. Laya: 512 tokens English, 1024 multilingual. SII: backbone limits `[B p.41]` |
| Multilingual? | Jev is documented as most accurate in English `[I p.183]`. JEVal is bilingual EN/ZH. NDM is trained on Portuguese legal text |
| Can it see images? | Jev: structured state only at launch `[G p.173]`. Open: Visual Jev and PixelJev on Qwen-VL `[A p.5]` |
| Does adding questions add latency? | Vendor: typically no, because they run in parallel `[F p.145]`. One study: 17 labels vs 1 raised median latency from 1.15 s to 1.22 s `[A p.5]` |
| Best open model today? | SII-Decision-27B on JEVal: 79.24% accuracy, ECE 0.015 `[C p.52]` |
