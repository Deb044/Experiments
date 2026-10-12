# What this experiment asks, in simple English

Suppose a robot can do six tasks. We request that it stop doing task F1, then
F2, then F3. It should hold its pose for those tasks and remain useful on three
allowed tasks. The question is whether learning the newest restriction causes
an older restriction to stop working.

This extends continual unlearning from LLMs to a robot policy. We borrow O3's
idea of discouraging overlap between successive adapter directions. We change
the output target from language to actions and assess movement in a simulator.
The frozen base still knows the tasks; disabling the adapter restores access.
We therefore call the result conditional behavior suppression.

## Benchmark and fixed data

Use the LIBERO-Object OpenVLA-7B checkpoint and six tasks from the pinned
LIBERO-Object suite. Choose tasks before comparing trained methods. A task is
eligible only if the unmodified model completes at least one of two smoke
episodes. Save all selection outcomes, including rejected tasks. This produces
a capability-conditioned pilot, not an unbiased estimate over the entire suite.

The first three eligible tasks become F1/F2/F3; the next three are allowed.
Use up to 30 complete demonstrations per task, with 20/5/5 train/validation/test
trajectories. If only 15–29 are available, use 60/20/20%. Split whole trajectories,
never neighboring frames across splits. Sample 32 training windows and eight
evaluation windows per trajectory. Simulator evaluation initial states are
separate from selection states, but are not claimed unseen during pretraining.

## Experiment 1: does an older restriction recover?

Train two rank-8 language q/v LoRA adapters from identical initialization:

| Arm | Current request | Allowed demonstrations | Extra penalty |
|---|---|---|---|
| Sequential | Learn hold action tokens | Preserve demonstrated actions | None |
| Orthogonal | Same | Same | Discourage overlap with previous output-side factors |

Each stage trains on the current requested task and all allowed tasks, with
equal sampling. Older requested-task examples are not replayed by the adapter
trainer. Both methods receive identical samples, learning rate and update count.
Only the penalty differs. Stage 1 must agree between arms because there is no
previous factor yet.

The loss is `CE_hold + CE_allowed + lambda * sum ||B_previous.T B_current||²`
for the orthogonal arm, with lambda 0.1. Our LoRA update is `B A`, so **B** is
the output-side factor. This naming differs from O3's `A B` convention.
Orthogonality of factors is a testable heuristic, not a guarantee of independent
robot skills. Effective `B A` norms are logged to aid interpretation.

After every stage, evaluate all requests seen so far on held-out trajectories.
Measure hold-token cross entropy, translation error, rotation error and gripper
error. Summarize per trajectory before averaging. Compare an earlier request's
hold loss immediately after acquisition with its loss after later stages; a
positive increase indicates recovery/interference in this diagnostic.

**Positive evidence:** orthogonal training reduces earlier-request recovery
while matching current-request suppression and allowed-task quality.
**Negative evidence:** it adds no benefit, damages utility, or merely trains
less effectively on the newest request. Do not treat a larger weight-space
angle alone as success. Teacher-forced action errors are not robot task success.

## Experiment 2: does selective activation work in execution?

Fit a small detector using frozen image and prompt features from training
trajectories. Fit normalization on training only and choose the threshold on
validation only, targeting at most 5% allowed-window false positives. Evaluate
multimodal logistic, text-only logistic and multimodal prototype detectors.
Task instructions reveal identity, so outperforming a text-only detector is
not a main scientific claim. The detector refit can access old requested
training windows; the overall system is not retained-data-free or storage-free.

Evaluate five configurations on the same initial states:

| Configuration | Purpose |
|---|---|
| Unmodified base | Establish task completion and movement |
| Sequential always-on adapter | Measure ordinary sequential suppression |
| Orthogonal always-on adapter | Isolate the orthogonality effect |
| Orthogonal adapter with smoothed detector activation | Measure selective deployment |
| Detector directly issuing hold commands | Test whether learning an adapter adds value |

Run four states per task: **120 episodes** total. A pre-result timing decision
may reduce all configurations equally to two states: **60 episodes**. Failures
of the environment/model infrastructure are invalid episodes, never successful
suppression. Horizon is 280 steps and settling is ten steps. A zero-motion hold
controller must pass a pose/gripper sanity check before the study proceeds.
Binary gripper commands approximate open/closed preservation; intermediate
apertures may move and must be reported.

| Metric | Meaning |
|---|---|
| Requested completion rate | How often restricted tasks still succeed; lower is desirable |
| Allowed completion rate | Whether the robot retains useful skills; higher is desirable |
| Hold compliance | Whether translation/rotation remain within fixed tolerances |
| Pose displacement and gripper transitions | Whether suppression causes unwanted motion |
| False intervention rate | How often an allowed episode activates suppression |
| Detector AUROC, precision, recall, FPR | How accurately requests are recognized |
| Gate crossings and action-command jumps | Whether activation/control oscillates |
| Median/p95 control latency | Added execution cost, including detector inference |

The main pilot does **not** establish that smoothing improves execution: that
hypothesis needs paired hard-gate and unsmoothed-soft-gate rollouts too. Set
`rollout_optional_gates=true` before starting a new run to include them, raising
the set to 168 episodes (or 84 at two states). Sparse offline gate traces alone
cannot answer a temporal-control claim.

The detector-plus-hold baseline is decisive: if it is equally selective and
safer than the learned adapter, this benchmark does not justify the adapter.

## One-day budget and reporting

Setup/baseline selection: up to three hours; features/detectors: two; matched
training: six; offline evaluation: two; rollouts: nine; report/setup buffer:
two. Benchmark ten warmup and twenty measured optimizer updates. Use
microbatch one and effective batch eight through accumulation. Choose a common
200, 100 or 50 updates per stage from measured training time **before inspecting
method outcomes**, up to 1,200 updates across both methods. The 50-update option
is explicitly an undertrained engineering pilot.

Report measured Blackwell slice memory and phase runtimes rather than extrapolating
from the previous SmolVLA experiment. Six-hour medium jobs may need successive
resumes. Count cumulative allocation time and stop within the day; queue waiting
is additional. If a complete paired set cannot finish, publish the partial
engineering/offline report and mark closed-loop claims incomplete.

Expected artifacts are factual tables and plots, not invented scores:

- Per-stage request-history heatmaps and old-request recovery CSV.
- Requested non-completion versus allowed success plot, with raw counts.
- Detector tables, paired episode differences and activation/action traces.
- Phase telemetry, measured update-time estimate, model/environment identity,
  manifests and resumable checkpoints.

One training seed and a few states support feasibility evidence, not an ICML
claim on their own. A later study needs more seeds, request orders, task sets,
architectures, stronger baselines and independent checks of actual deletion if
it intends to claim machine unlearning rather than behavior suppression.
