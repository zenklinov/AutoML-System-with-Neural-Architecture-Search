# Approved CIFAR-10 Experiment Protocol

This is the executable experiment contract. The 2026-09-22 pilot amendment freezes a
recommended final budget in dedicated `configs/final_*.yaml` files. That freeze is not
authorization to execute the final search, confirmation, retraining, or official-test
evaluation. The older general and pilot templates remain deliberately unlocked.

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

The calibration pilot examines complete curves and early-versus-late ranks. The comparative
pilot measures runtime, learning behavior, failure modes, and the amount of genuinely adaptive
TPE behavior. The 2026-09-22 amendment below records the resulting frozen recommendation.
Pilot measurements remain diagnostic and are not performance claims.

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

## Resource-policy amendment — 2026-09-20

The initial one-CPU calibration was stopped as `ABORTED — ProtocolFeasibilityStop` after
the smallest 8,074-parameter panel architecture required 2,622.47 seconds for 12 epochs.
That incomplete CPU run is feasibility evidence only and must not be combined with GPU
calibration or search results.

Calibration, Random pilot, and TPE pilot now use one NVIDIA GPU per trial, one trial at a
time, and the separately pinned `requirements.cuda.lock` environment. Their training
recipe, split, transforms, architecture space, seeds, and strategy fairness rules are
unchanged. CPU installation, smoke tests, and GitHub Actions remain supported through
`requirements.lock`; CUDA is not a general project dependency. Fresh GPU calibration is
required before selecting a pilot grace period. The final budget remains unlocked.

### GPU calibration decision

Run `asha-calibration-20260920T111018Z-fece8078` completed all six fixed architectures
from Git commit `dcbd734317e4c52021879038e27548bb50690ebd` on the RTX 4060 Laptop GPU.
All final top-three architectures were already in the top half at every epoch; epoch-4
Spearman rank correlation with epoch 12 was 0.943. The panel showed no late-learning
architecture at risk of missing the final top half. The comparative pilot therefore keeps
`grace_period_epochs: 4`, `reduction_factor: 2`, `max_t: 12`, and concurrency one. Its
paired diagnostic budget is 16 trials per strategy at search seed 2026, providing six
post-startup TPE suggestions. These are pilot settings, not final-budget decisions.

## Final-budget amendment — 2026-09-22

The paired GPU pilot completed from clean Git commit
`dfe1b0f54f3e43eab6c24641be67df0f4fbdba35` on the same RTX 4060 Laptop GPU,
with one GPU per trial and concurrency one. Random run
`provisional-random-pilot-20260922T065949Z-6ecf900d` completed 7 and pruned 9 of
16 trials, consumed 132 epochs, had no failures, and recorded 1.151 GPU-hours.
TPE run `provisional-tpe-pilot-20260922T081203Z-c648b8d9` completed 9 and
pruned 7 of 16 trials, consumed 144 epochs, had no failures, and recorded 0.917
GPU-hours. The two resolved configs matched except for strategy and run label.

TPE trials 1–10 were startup proposals and exactly matched the first ten seeded
Random architectures and terminal scores. Trials 11–16 were adaptive proposals:
four completed, two were pruned at epoch 4, and none failed. Their median parameter
count was 38,986 versus 78,138 for startup proposals. This is evidence that adaptive
sampling occurred, not evidence that TPE is superior.

The strongest completed pilot trajectories had not uniformly plateaued by epoch 12.
Across completed trials, the median epoch-8-to-12 gain was 0.0074 for Random and
0.0168 for TPE; the strongest adaptive TPE candidate gained 0.0332 over those four
epochs. Validation noise was also visible, so these deltas are descriptive. The final
search ceiling is therefore 20 epochs: long enough to observe later separation without
jumping to an unsupported 30-epoch budget.

Grace period remains 4 with reduction factor 2. Calibration showed epoch-4 Spearman
rank correlation 0.943 with epoch 12 and retained all final top-half architectures in
the current top half. Pilot pruning occurred only at epochs 4 and 8, with no failures,
and did not show an obvious systematic penalty against larger/deeper candidates.

### Compute-tier comparison

| Tier | Trials / strategy / seed | Search seeds | Total search trials | Max epochs | Projected search GPU-hours |
|---|---:|---:|---:|---:|---:|
| Minimum Credible | 16 | 2 | 64 | 20 | 6.89 |
| Recommended | 20 | 3 | 120 | 20 | 12.93 |
| Extended | 30 | 3 | 180 | 20 | 19.39 |

The projection scales the measured combined 2.068 pilot GPU-hours by total trial count
and the 20/12 epoch ratio. It is a planning estimate, not a runtime guarantee: ASHA
rungs, architecture mix, and laptop thermal state make actual cost nonlinear.

The frozen final tier is **Recommended**:

- Random and TPE each receive 20 trials for each of search seeds 2026, 2027, and 2028.
- Both use `max_t: 20`, `max_epochs: 20`, grace period 4, reduction factor 2,
  one GPU per trial, and concurrency one.
- TPE keeps 10 startup trials, giving 10 adaptive proposals per search seed.
- Confirmation keeps seeds 3101–3103.
- Final training keeps seeds 4101–4105 for both strategy winners and the fixed baseline.
- The official test remains inaccessible until trained final checkpoints exist and a
  separate `--acknowledge-locked-test` invocation is explicitly authorized.

The frozen stage configs are `final_random_search.yaml`, `final_tpe_search.yaml`,
`final_confirmation.yaml`, `final_training.yaml`, and `final_locked_test.yaml`. They
must not be executed as part of this pilot task. Evidence is in
`results/pilot/pilot_analysis.json`, `pilot_provenance.json`, and
`pilot_budget_recommendation.json`.
