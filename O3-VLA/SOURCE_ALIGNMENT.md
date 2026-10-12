# Upstream alignment and deliberately modified components

Reference code inspected at immutable revisions:

- [O3](https://github.com/GCYZSL/O3-LLM-UNLEARNING/tree/839afea7fc09001433ef30bd6670a6d6804a9045): sequential shared-adapter training and output-factor orthogonality. Our code is independently implemented, not a complete reproduction.
- [OpenVLA](https://github.com/openvla/openvla/tree/c8f03f48af692657d3060c19588038c7220e9af9): action tokenizer, model forward/generation and LIBERO evaluation utilities.
- [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO/tree/8f1084e3132a39270c3a13ebe37270a43ece2a01): suite ordering, demonstrations, environments and initial-state files.
- [LIBERO-Object checkpoint](https://huggingface.co/openvla/openvla-7b-finetuned-libero-object/tree/287d6cfdf12d07b1449505f66d9bf3550257e9b3).
- [Remote model code](https://huggingface.co/openvla/openvla-7b/tree/47a0ec7fc4ec123775a391911046cf33cf9ed83f): the checkpoint's auto-map refers to this base repository. Model and code revisions are pinned separately.

## Native interfaces

OpenVLA uses a fused vision backbone, language backbone and seven autoregressive
action tokens. Training masks prompt tokens, inserts the expected terminal
29871 prompt token, and uses the last seven causal predictions after native
image-token insertion. Generation uses native `predict_action`; gate scale is
fixed for the entire generation of one control step. This checkpoint does not
consume robot proprioception as a policy input; gripper state is used only to
construct hold targets and direct-hold control.

Action binning follows 256 edges on [-1,1] and checkpoint q01/q99 normalization.
The final gripper coordinate is unnormalized. LIBERO commands use -1=open and
+1=close, whereas native OpenVLA actions use 1=open and 0=close. Explicit
conversion applies on both training and generation. Evaluations use rotated
agentview images, JPEG conversion, Lanczos resize and the official 90%-area
center crop; the HDF5 reader assumes raw LIBERO orientation and action convention.
Do not feed already transformed RLDS records into it. The pinned LIBERO
`scripts/create_dataset.py` saves observations **after** stepping the corresponding
action. We pair observation j with action j+1 and discard the last observation.
This causal alignment is explicit rather than treating the historical HDF5
indices as pre-action observations. Converted datasets may have a different
convention and need a separate audited reader.

Vision and projector/base parameters remain frozen. Only language q/v adapters
train. The previous output-side factor is detached and the penalty is a sum,
not an orthogonal projection or a constraint on the effective policy function.
This rank-limited adapter is not assumed to support unlimited requests.

## Departures from an exact O3 reproduction

The forgetting objective becomes replacement with hold action tokens. Allowed
demonstrations explicitly regularize utility. A lightweight frozen-feature
detector replaces O3's contrastive/entropy detector. Detector refits use past
requested training data. Temporal smoothing is added for control, and native
closed-loop rollouts assess task completion and movement. No certified deletion,
retained-data-free training, factorization-invariant orthogonality, or guaranteed
action interpolation is claimed.

## Verification boundary

CPU tests and the synthetic end-to-end run check scientific bookkeeping,
adapter gradients/gates, trajectory splitting, calibration, resume and report
artifacts. They do not execute the 7B CUDA model or EGL simulator. Native
`prepare` and `benchmark` are required on the actual Blackwell allocation;
record the environment report and any setup failures. Do not present synthetic
figures as real results or infer performance from a passing mock.
