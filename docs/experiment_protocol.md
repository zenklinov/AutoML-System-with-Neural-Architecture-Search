# Approved CIFAR-10 Experiment Protocol

This is the executable experiment contract. Implementing and testing it does **not** authorize a pilot, candidate search, confirmation run, final retraining, or official-test evaluation. The final budget remains deliberately unlocked (`protocol.final_budget_locked: false`).

## Question and comparison

Primary research question: under an identical architecture space, fixed training recipe, CIFAR-10 split, ASHA policy, resource allocation, and trial budget, how do Random Search and seeded Optuna/TPE compare in search efficiency and in the validation performance of discovered configurable CNN architectures?

Secondary questions are whether early validation ranks remain useful at later epochs, whether ASHA appears aggressive, ineffective, or architecture-biased, how parameter count relates to validation performance, and how stable shortlisted architectures are across new training seeds. These are descriptive questions, not general claims about AutoML or Bayesian optimization.

The fixed reference CNN is reported beside both strategies, not searched. Validation accuracy is primary and trainable parameter count is mandatory. MACs/FLOPs are deferred until an operator-counting convention is approved.

This protocol does not assume that Optuna/TPE will outperform Random Search. Legitimate outcomes are TPE better, Random better, similar performance, or results too variable for a strong conclusion.

## Data and preprocessing

Search, calibration, and confirmation load only the official 50,000-example CIFAR-10 training partition. Seed 2026 creates a stratified 45,000/5,000 split: exactly 4,500 training and 500 validation examples per class, fixed across candidates and strategies.

Training uses random 32×32 crop with four-pixel padding, horizontal flip with probability 0.5, tensor conversion, and normalization. Validation/test use tensor conversion and the same normalization without augmentation. Population statistics were computed once from raw pixels in only the designated 45,000-example training subset, scaled by 1/255:

- mean: `[0.49151359443082787, 0.48231470971200985, 0.44681383799700436]`
- std: `[0.24705092701492043, 0.24347971893365994, 0.2615095063426509]`

They are frozen. Validation and test pixels did not contribute. Final retraining uses all 50,000 official training examples, starts from scratch, reuses the constants, and does not load the test partition.

## Architecture space and baseline

Only architecture is searched. Depth is uniformly selected from 2, 3, and 4 blocks. Each active block independently chooses kernel `{3,5}`, channels `{16,32,64}`, activation `{ReLU,SiLU}`, and residual `{true,false}`. Inactive blocks are not sampled. The effective space is `24² + 24³ + 24⁴ = 346,176`; the depth-first prior is intentionally nonuniform over individual architectures.

The fixed reference baseline has three blocks: `(3,32,ReLU,residual)`, `(3,64,ReLU,residual)`, `(3,64,ReLU,residual)`. It has 68,458 trainable parameters and uses the identical fixed recipe and seed sets.

## Fixed recipe and seeds

Cross-entropy; Adam; learning rate 0.001; weight decay 0.0001; batch size 64; no learning-rate scheduler; validation every epoch; deterministic algorithms requested. Optimizer/data/training values are not search dimensions.

- split: 2026
- candidate-search strategy seeds: 2026, 2027, 2028
- common within-search training seed: 4242
- confirmation seeds: 3101, 3102, 3103
- final-training seeds: 4101–4105

`search-matrix` runs every candidate-search seed. `search` runs only the explicit `search.seed`.

## TPE and ASHA

TPE and random search share the conditional space and ASHA settings. TPE explicitly uses 10 startup trials, 24 EI candidates, `multivariate=false`, and `constant_liar=false`. ASHA uses `training_iteration`, maximizes `validation_accuracy`, has `max_t=12`, and reduction factor 2. The checked-in provisional config has grace period 12 and concurrency 1, so calibration evidence is collected before enabling useful pruning.

Calibration uses the fixed diverse panel in `configs/asha_calibration_panel.yaml`, complete learning curves, no ASHA, and no official test. Settings may change only in a documented amendment based on those curves—not candidate or test outcomes.

The scheduler-calibration template, Random comparative-pilot template, and TPE comparative-pilot template are checked in separately. Their roughly 12-epoch ceiling, grace period around four for comparative pilots, concurrency one, and small trial counts are provisional. The two comparative templates differ in treatment only: independent Random sampling versus adaptive seeded TPE sampling. Repository SHA, conditional prior, split, training recipe, ASHA, trial count, allocation, concurrency, objective, and output schema must otherwise match.

Pruning rate is a diagnostic, never a pass/fail threshold. Report completed, pruned, and failed trials, epochs consumed, pruning epochs, and histories. Do not require an arbitrary percentage to be pruned.

## Stages and leakage controls

1. `calibrate-asha` collects full validation curves and imports no scheduler.
2. `search-matrix` runs the two strategies over their declared seeds; ASHA sees validation only.
3. `confirm` accepts canonical results, ignores pruned/failed trials, and retrains completed shortlisted architectures from scratch across confirmation seeds. It does not invoke ASHA.
4. `final-train` is disabled until `final_budget_locked` is explicitly true in a separately approved amendment. It trains the selected architecture and baseline on all training data across final seeds without evaluation.
5. `evaluate-locked-test` is the only module/command loading `train=False`. It requires already-trained final checkpoints and `--acknowledge-locked-test`; no search, pruning, selection, or training occurs there.

The test set must not influence normalization, augmentation, calibration, early stopping, checkpoint selection, architecture selection, or reranking.

Confirmation deduplicates identical architectures. It retains one Random-selected and one TPE-selected architecture rather than discarding a strategy winner solely because the other has a higher confirmation mean. The primary ranking is mean confirmation validation accuracy. When means differ by no more than 0.001 absolute accuracy, fewer parameters wins; the stable architecture ID is the deterministic final fallback. One fixed search-training seed reduces within-search noise but cannot establish intrinsic architectural superiority, which is why independent confirmation seeds are required.

Final retraining discards all search/confirmation weights. Both strategy-specific winners and the fixed reference baseline are trained from scratch for every final seed. The isolated test stage evaluates every pre-specified final model once per final seed and cannot change configurations, seeds, architectures, stopping, or selection.

## Outputs and decision rules

Search artifacts contain terminal trial records, per-epoch histories, lifecycle timestamps, pruning iteration, epochs consumed, parameter count, checkpoint references, and aggregate wall-clock/CPU/GPU accounting. Search and confirmation contain no test metrics.

Compare strategy distributions of best validation accuracy across three independent strategy seeds. Report every seed, mean, standard deviation, trial/epoch consumption, wall time, CPU/GPU hours, and parameters. Confirmation reports every architecture × seed plus mean/std. Ties are broken by lower parameter count, then stable architecture ID. No quality/efficiency claim is permitted before runs exist.

The descriptive analysis will show raw search-seed results, mean, standard deviation, range, and paired Random-minus-TPE differences. Confirmation and final results will show raw seed values, mean, standard deviation, and range. Any later confidence interval must state the small-sample limitation. T-tests, ANOVA, and p-values are not preplanned.

## Pilot and final-budget decision

The calibration pilot examines complete curves and early-versus-late ranks. The comparative pilot measures runtime, learning behavior, failure modes, and the amount of genuinely adaptive TPE behavior. Because TPE has 10 startup trials, a tiny trial budget provides little adaptive evidence. A 20-trial-per-seed lower-bound may be considered if compute permits, but neither 20×3 nor any “recommended/extended” tier is locked. Final `max_t`, grace period, trial count, and which candidate search seeds to run are chosen only in a documented amendment after pilot evidence. Final budget status is **DO NOT LOCK YET**.

## Failure and invalidation rules

The entire affected comparison is invalid if official test data is accessed early or if the strategies use different search spaces, priors, or training recipes. A paired search seed must be rerun after a code/config/environment change, unequal resource allocation, missing manifest/result data, corrupted adaptive history, or unexplained failures that influence TPE observations. Failed trials are recorded and never silently dropped. An isolated final run may be rerun only when it did not change any selection decision and the failure and rerun are documented.

## Provenance and retention

Every run records the Git SHA and dirty state, frozen resolved config, named seeds, command/stage, Python and package versions, operating system, CPU/GPU information, allocation, start/end state, and artifact references. Search retains canonical terminal records and compact per-epoch histories outside Ray temporary directories, plus raw Ray state/checkpoints. Calibration, confirmation, final training, and locked-test evaluation retain manifests and their stage summaries/checkpoints. Generated datasets, checkpoints, and runtime results remain ignored and are not committed by this implementation task.

## Interpretation and claim boundaries

Results apply only to this conditional CNN space, CIFAR-10 protocol, implementation, budgets, seeds, resources, and software/hardware boundary. Parameter count is explanatory, not proof of efficiency; latency and monetary cost are outside the primary experiment. No current evidence permits claims that TPE is better than Random, Random is better than TPE, ASHA saves a specific percentage, NAS beats the baseline, NAS produces smaller models, a selected architecture is statistically superior, or the system is distributed, cost-aware, production-ready, or state of the art.

## Commands (not authorization to run)

```bash
automl-nas validate-config --config configs/cifar10.yaml
automl-nas calibrate-asha --config configs/cifar10.yaml --panel configs/asha_calibration_panel.yaml
automl-nas search-matrix --config configs/cifar10.yaml
automl-nas confirm --config configs/cifar10.yaml --results <tpe.json> <random.json>
automl-nas final-train --config <approved-locked-config.yaml> --confirmation <confirmation.json>
automl-nas evaluate-locked-test --config <locked-config.yaml> --final-models <final_models.json> --acknowledge-locked-test
```

The manifest records software/hardware. Deterministic algorithms are requested, but bit-for-bit equivalence across CUDA devices, drivers, or libraries is not claimed.
