# Drift-Forget: experiment protocol (v0, draft)

## 1. Status

This is a **living design document, not a frozen pre-registration** (decision D4). Hypotheses, outcomes and
tests are written down before any data exist. Every later change is recorded in the change log (§10) with
its date and reason. A paper written from this study must say plainly that the protocol was not formally
frozen. Items marked **TBD (Dn)** are still open and are decided at the increment that needs them.

## 2. Question and hypotheses

**Question.** In pretrained π0 trained continually on LIBERO, how do weight drift and feature drift between
checkpoints relate to forgetting of earlier tasks?

**Background.** Liu et al. (arXiv 2603.03818) report that π0 with experience replay (ER) barely forgets
(LIBERO-Object NBT −0.007 at M=1000), while sequential fine-tuning forgets heavily (NBT 0.696). Their paper
does not explain this.

**Hypotheses.**

- **H1 (attenuation).** Drift shrinks along the chain weights → old-task features → old-task action
  velocity → loss → success. The shrinkage is stronger for larger replay buffers M.
- **H2 (relevant drift).** Sensitivity-weighted weight drift (W_eff) and feature drift predict the drop in
  old-task success better than raw weight drift (W_raw).
- **H3 (locus).** Drift in the action-side modules (action expert and projections) predicts forgetting
  better than drift in the vision encoder or the VLM. Component swaps that put the old modules back in
  should agree.
- **H4 (replay).** Old-task samples that were replayed drift less than old-task samples that were not.
  The gap grows with M.
- **H5 (schedule).** The relationships in H2 still hold when the LR schedule restarts at every task, so
  they are not an artefact of the paper's single decaying schedule.

## 3. Setup (matches the paper and the vendored code)

All code is at the vendored commits (see `VENDORED.md`). Facts marked *(code)* were read in
`continual-openpi`.

| Item | Setting |
|---|---|
| Model | π0 base weights. Config `pi0_libero_low_mem_finetune-libero_object_sequential`: vision encoder fully trained, VLM (`gemma_2b_lora`) and action expert (`gemma_300m_lora`) LoRA, freeze mode `lora`. |
| Optimizer | AdamW, weight decay 1e-10, gradient clip 1.0, no EMA *(code)*. |
| Batch | 8 (`--batch-size 8`; the config default is 32). With ER, 8 current + 8 replay from task 1 on *(code)*. |
| LR schedule, paper | One cosine over the whole run: warmup 1000 steps, peak 2.5e-5, floor 2.5e-6 reached at step 30,000, never reset between tasks *(code)*. |
| LR schedule, restart | Warmup and cosine restart at each task. Shape TBD (D15). |
| Suites | LIBERO-Object (10 tasks). LIBERO-10 is in Tier 4. |
| Task order | The paper's Appendix B.2 order, verified to match the fork's `_LIBERO_OBJECT_SEQ` / `_LIBERO_10_SEQ` and LIBERO's eval indices (`task_order_index=0`). Object: alphabet soup, bbq sauce, butter, chocolate pudding, cream cheese, ketchup, milk, orange juice, salad dressing, tomato sauce. |
| Data | OpenVLA-filtered `modified_libero_rlds` (no-noops), converted to LeRobot format with the fork's converter. |
| Steps per task | 10,000 (100,000 per run). The fork's ER resume requires task ends at `…9999` *(code)*. |
| Replay | Fixed buffer of M transitions per earlier task, recorded in each checkpoint (`buffer_lst`). |
| Normalization | Per-task norm stats. The policy server takes `--task_idx` *(code)*. |
| Simulator eval | Each task-end checkpoint is evaluated on tasks 0..k, 50 trials per task, init states fixed by episode index, seed 7, max 280 steps for Object, replan every 5 steps *(code)*. Mesa EGL rendering on CPU. |
| Seeds | 5 per condition: values TBD (D10); the count may change after the pilot. |

## 4. Conditions and priority tiers

Each run trains all 10 tasks for 100,000 steps. The budget gate runs tiers **in order** and drops whole tiers
from the bottom. It never drops seeds inside a tier.

| Tier | Condition ID | Replay M | LR schedule | Batch | Runs |
|---|---|---|---|---|---|
| 1 | `seq_M0` | 0 (`--cl ""`) | paper | 8 | 5 |
| 1 | `er_M10` | 10 | paper | 8+8 | 5 |
| 1 | `er_M100` | 100 | paper | 8+8 | 5 |
| 1 | `er_M1000` | 1000 | paper | 8+8 | 5 |
| 2 | `seq_M0_rs`, `er_M10_rs`, `er_M100_rs`, `er_M1000_rs` | 0/10/100/1000 | restart | as Tier 1 | 20 |
| 3 | `seq_M0_b16` | 0 | paper | 16 | 5 |
| 4 | Tiers 1–3 on LIBERO-10 (all 9 conditions, IDs suffixed `_l10`) | as Tiers 1–3 | as Tiers 1–3 | as Tiers 1–3 | 45 |

Total across all tiers: 90 runs. LIBERO-10 episodes allow up to 520 steps (Object allows 280) *(code)*, so
simulator eval costs more per trial in Tier 4.

Tier 3 tests whether `seq_M0` differs from ER because it has no replay or because it sees half as many
samples per step. A pilot (1 seed, Tier 1, settings TBD) comes before Tier 1. Its data are reported
separately and are not pooled with the main runs.

## 5. Outcomes

**Primary (D6): simulator success.**
- The success matrix c_{k,τ}: success on task k after training through task τ (τ ≥ k).
- NBT_k = mean over τ>k of (c_kk − c_kτ); NBT = mean over k (paper §4).
- Normalized NBT: each term divided by c_kk, excluding tasks with c_kk = 0 (paper App. A.2).
- Average SR.
- Outcomes are stored per episode. Because init states are identical across checkpoints, comparisons
  between checkpoints are paired.

**Secondary.**
- `L_off`: old-task flow-matching loss at fixed (noise, time) draws, relative to its value at the end of
  task k.
- `L_act`: error of sampled action chunks against the demos.
- Knowledge-transfer curves and recovery steps as in the paper (recovery in scope: TBD, D22).
- Mid-task simulator evals: TBD (D21).

## 6. Drift measures

Each measure is computed per run × snapshot × old task k × module × probe group.

- **Modules (TBD, D18).** Vision, VLM LoRA, action-expert LoRA, and projections, using the module regexes
  from `scripts/component_swap.py`. LoRA drift is computed on the merged update s·BA.
- **W_raw.** ‖θ_t − θ_ref‖ / ‖θ_ref‖. The reference is either the end of task k or the previous snapshot.
- **W_eff.** Σᵢ Fᵢ Δθᵢ², where F is the diagonal empirical Fisher of task k at the end of task k.
- **W_par.** The share of ‖Δθ‖² that lies in the span of task-k per-sample gradients (small modules only).
- **Feature drift.** Measured at SigLIP tokens, the VLM output (`prefix_out`), the action expert output
  (`suffix_out`) and the velocity v_t. Pooling TBD (D17). Metrics: 1 − CKA (linear and debiased),
  Procrustes residual, held-out ridge residual, and the user's mappings (TBD, D5, D19).
- **Theory quantities (scope TBD, D20).**
  - The exact split ΔL_k = E‖Δv‖² + 2E⟨Δv, v_ref − u⟩.
  - Linearized drift J_mΔθ_m.
  - Quadratic prediction gᵀΔθ + ½ΔθᵀHΔθ.
  - Loss along the straight path from θ_ref to θ_t.
- **Probes (TBD, D16).** A fixed set per task, tagged `replayed` or `trained_not_replayed` from `buffer_lst`.
- **Snapshots (D13, D14).**
  - Contents: trainable parameters in fp32, plus the pretrained start `init`.
  - When: steps 49, 99, 249, 499, 999, 1999, 3999, 5999 and 7999 after each task starts.
  - Task ends: read from the full checkpoints, which are kept at task ends only.
  - Retention: intermediate snapshots are deleted only after a run's analysis and evals are checked.

## 7. Analysis plan

| Hypothesis | Main test | Outcome |
|---|---|---|
| H1 | Attenuation ratio at each link of the chain. Trend in M from a mixed model with M as a fixed effect. | Drift ratios |
| H2 | Leave-one-run-out cross-validated predictive performance for SR drop: W_eff and feature drift against W_raw. Paired cluster bootstrap over runs. | SR (episode-level logistic mixed model) |
| H3 | Same comparison by module, then whether each module's predicted share matches its recovered share in component swaps. | SR, swap recovery |
| H4 | Replayed vs. non-replayed drift, paired within (run, snapshot, task), with an M interaction. | Feature drift |
| H5 | H2's comparisons repeated in Tier 2, with a test of the schedule × drift interaction. | SR |

Rules that apply to every test:

- **Dependence.** Observations are nested in run → condition and crossed with task. All models include
  random intercepts for run and task. Snapshots are never counted as independent seeds.
- **Time confound.** Covariates for steps since the end of task k and the current LR. Comparisons across M
  at matched steps. First differences between consecutive snapshots.
- **Noise floors.** Split-half probe resampling, and drift between seeds at the same checkpoint.
- **Multiplicity.** Holm correction within each hypothesis family.
- **Significance level and exact test choices.** TBD (D23).

## 8. Validity gates and stop rule

These run on the MIG slice before main runs. Details are in the plan.

- **G1:** environment check.
- **G2:** smoke test and timing.
- **G3:** the snapshot hook doesn't change training.
- **G4:** the analysis passes its self-consistency checks.
- **G5 (validity anchor):** Tier 1 `seq_M0` and `er_M1000` NBT on Object fall within seed noise of 0.696 and
  −0.007 (paper Table 2). **If G5 fails, analysis stops until the gap is explained.**

## 9. Known confounds and deviations from the paper

- **Suites.** Object only unless Tier 4 runs; with Tier 4, Object and LIBERO-10. The paper uses four suites for
  its main table.
- **Batch.** ER sees twice as many samples per step as sequential fine-tuning. Tier 3 addresses this.
- **LR schedule.** Under the paper's schedule, tasks 4–10 train at the floor LR, so later tasks may drift less
  just because of the schedule. Tier 2 and H5 address this.
- **Seeds.** The paper does not give its seed count. Ours is 5 (pending the pilot).
- **Pre-registration.** The protocol is not formally frozen (§1).

## 10. Change log

| Date | Version | Change | Reason |
|---|---|---|---|
| 2026-10-24 | v0 | First draft from decisions D4, D6, D7, D8, D9, D10, D15 (conditions only) and the tier rule. | — |
| 2026-10-24 | v0 | Tier 4 changed from Tier 1 on LIBERO-10 (20 runs) to all 9 conditions on LIBERO-10 (45 runs). | The user chose a full replication on the second suite, so H5 and the batch control are tested there too. |
| 2026-10-24 | v0 | Task order filled in, and its open item removed. | Paper App. B.2, the fork's task lists, and LIBERO's eval order were checked and agree. |
| 2026-10-25 | v0 | Snapshot schedule, precision and retention fixed (D13, D14). Full checkpoints at task ends only. | Storage: full checkpoints every 2000 steps would take about 600 GB per run. |

## Open items

- **D5, D19:** what the user's feature mappings compute and how they are fitted.
- **D10:** seed values.
- **Replay-buffer seeding (increment 6).** `create_deterministic_buffer` draws the buffer with numpy's unseeded
  global generator, so a rerun with the same seed replays different samples. The buffer actually used is
  recorded either way.
- **D15:** shape of the restart schedule.
- **D16:** probe sizes.
- **D17:** feature pooling.
- **D18:** module partition.
- **D20:** theory scope.
- **D21:** mid-task simulator evals.
- **D22:** recovery fine-tuning.
- **D23:** significance level and exact tests.
- **Pilot settings.**
