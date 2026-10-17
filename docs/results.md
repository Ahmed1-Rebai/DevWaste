# Experimental Results

> **Part 1 uses synthetic data from DevWaste's own simulator. Part 2 uses
> real Docker metrics from one container with deliberately injected
> incidents. Neither is evidence of accuracy on real production
> workloads.**

# Part 1: Synthetic data

## Setup

Reproduce with:

```bash
python scripts/run_experiment.py --out experiments
```

| Item | Value |
|---|---|
| Training data | One `normal` run, 3000 samples, simulator seed 1000 |
| Test data | 3 independent runs per scenario (and 3 held-out `normal` runs), 1000 samples each, each with its own seed |
| Model | LSTM AE, 1 layer, hidden 16, window 10, stride 1 |
| Training | Adam lr 1e-3, batch 32, ≤50 epochs, early stopping (patience 10) on normal validation loss |
| Split | Chronological 80/20 per series *before* windowing; scaler fit on the training split only |
| Threshold | 95th percentile of reconstruction error on normal validation windows |
| Training seeds | 0, 1, 2 |
| Aggregation | Mean ± std over 9 (training seed × test run) evaluations |
| Unit of evaluation | Window (a window is anomalous if any of its samples is labelled anomalous) |
| Hardware | AMD Ryzen 5 220 laptop CPU (12 threads), PyTorch 2.14.1+cpu, Python 3.13 |

Anomalous-window prevalence in the test sets (mean over runs): cpu-spike
15.5%, memory-growth 47.0%, network-spike 16.4%, oscillating 50.5%,
normal 0%.

Training takes 5.0 s per model on average. Batched inference takes about
0.002 ms per window.

## Detectors

| Name | Description |
|---|---|
| **LSTM AE** | Reconstruction error of the LSTM Autoencoder. |
| **zscore** | Largest absolute z-score in the window, using the same training scaler as the LSTM. Threshold: same 95th-percentile validation calibration. |
| **rolling** | Largest past-only rolling z-score (previous 10 samples, per series) in the window. Threshold: same 95th-percentile validation calibration. |
| **threshold** | Fixed rules `cpu_percent > 80` or `memory_percent > 80`. **Caveat:** these rules overlap with how the cpu-spike labels are generated, so it is advantaged on that scenario. |

## Results

| Scenario | Detector | Precision | Recall | F1 | FPR | ROC-AUC |
|---|---|---|---|---|---|---|
| cpu-spike | LSTM AE | 0.826 ± 0.069 | 0.975 ± 0.008 | 0.893 ± 0.041 | 0.039 ± 0.017 | 0.995 ± 0.003 |
| cpu-spike | zscore | 0.946 ± 0.042 | 0.961 ± 0.009 | 0.953 ± 0.017 | 0.010 ± 0.008 | 0.990 ± 0.008 |
| cpu-spike | threshold | 1.000 ± 0.000 | 0.996 ± 0.003 | 0.998 ± 0.002 | 0.000 ± 0.000 | n/a |
| cpu-spike | rolling | 0.707 ± 0.032 | 0.650 ± 0.010 | 0.677 ± 0.015 | 0.050 ± 0.009 | 0.803 ± 0.034 |
| memory-growth | LSTM AE | 0.952 ± 0.027 | 0.648 ± 0.096 | 0.768 ± 0.078 | 0.027 ± 0.011 | 0.896 ± 0.059 |
| memory-growth | zscore | 0.946 ± 0.027 | 0.532 ± 0.083 | 0.678 ± 0.076 | 0.025 ± 0.009 | 0.919 ± 0.043 |
| memory-growth | threshold | 1.000 ± 0.000 | 0.165 ± 0.075 | 0.275 ± 0.110 | 0.000 ± 0.000 | n/a |
| memory-growth | rolling | 0.436 ± 0.199 | 0.029 ± 0.010 | 0.055 ± 0.019 | 0.043 ± 0.030 | 0.488 ± 0.052 |
| network-spike | LSTM AE | 0.842 ± 0.068 | 0.985 ± 0.006 | 0.907 ± 0.041 | 0.037 ± 0.019 | 0.990 ± 0.001 |
| network-spike | zscore | 0.782 ± 0.077 | 0.791 ± 0.043 | 0.782 ± 0.025 | 0.046 ± 0.023 | 0.966 ± 0.016 |
| network-spike | threshold | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | n/a |
| network-spike | rolling | 0.757 ± 0.075 | 0.718 ± 0.090 | 0.735 ± 0.076 | 0.045 ± 0.014 | 0.849 ± 0.037 |
| oscillating | LSTM AE | 0.967 ± 0.025 | 0.995 ± 0.004 | 0.981 ± 0.013 | 0.035 ± 0.026 | 0.995 ± 0.006 |
| oscillating | zscore | 0.939 ± 0.049 | 0.603 ± 0.019 | 0.735 ± 0.029 | 0.041 ± 0.033 | 0.953 ± 0.033 |
| oscillating | threshold | 1.000 ± 0.000 | 0.681 ± 0.009 | 0.810 ± 0.006 | 0.000 ± 0.000 | n/a |
| oscillating | rolling | 0.459 ± 0.155 | 0.027 ± 0.009 | 0.050 ± 0.018 | 0.033 ± 0.012 | 0.451 ± 0.013 |
| normal | LSTM AE | – | – | – | 0.035 ± 0.010 | n/a |
| normal | zscore | – | – | – | 0.037 ± 0.005 | n/a |
| normal | threshold | – | – | – | 0.000 ± 0.000 | n/a |
| normal | rolling | – | – | – | 0.031 ± 0.005 | n/a |

## Interpretation

- **Where the LSTM helps:** temporal patterns. On `oscillating`, F1 is
  0.981 vs 0.735 for z-score, because each value stays in the normal range
  but the sequence shape does not. It also leads on `network-spike` (0.907
  vs 0.782) and `memory-growth` (0.768 vs 0.678). Gradual growth is the
  hardest scenario for every detector: the early leak phase is labelled
  anomalous but looks almost normal, so recall stays at about 0.65 even
  for the LSTM. The memory-growth ROC-AUC of the LSTM (0.896) is slightly
  below z-score (0.919), so its F1 advantage comes from where the
  threshold falls rather than from better ranking overall.
- **Where it does not:** isolated CPU spikes. The per-value z-score is
  better (F1 0.953 vs 0.893), with fewer false positives.
- **Rolling statistics** adapt to slow change by design, so they miss
  memory growth and sustained oscillation almost entirely (F1 ≈ 0.05).
  They work moderately on short spikes.
- **Fixed thresholds** look strong only where the rule matches the label
  rule (`cpu-spike`). They miss network anomalies entirely and catch only
  the end of memory growth.
- **False-positive rate on clean data** is 3.1–3.7% for all calibrated
  detectors, close to but below the 5% implied by the 95th-percentile
  calibration.
- **Run-to-run variation matters.** An earlier version of this experiment
  used a single test run per scenario and reported memory-growth F1 of
  0.919 for the LSTM. With 3 test runs it is 0.768 ± 0.078. Single-run
  numbers were optimistic.

## Limitations of this experiment

- The simulator generates every scenario, including the "normal" process.
  Real workloads have far more varied patterns, and anomalies are not
  injected with clean labels.
- One training run and three test runs per scenario. The std mixes
  training-seed and test-run variation.
- Window-level labels with stride 1 make neighbouring windows highly
  correlated. Metrics count windows, not distinct incidents.
- Hyperparameters were left at their defaults (no tuning on test data).

# Part 2: Controlled incidents in a real container

## Setup

Reproduce with (requires Docker and the `python:3.11-slim` image; about 11 minutes):

```bash
python scripts/real_incident_experiment.py --out experiments/real
```

- **Container:** one unprivileged container (`python:3.11-slim`, 1 CPU, 512 MB, no network), created and removed by the script.
- **Workload:** a light normal load, with small variable CPU work every 0.5 s and occasional 32 KB writes.
- **Phases:** 300 s normal (training), then an incident phase (testing). Incidents are separated by 30 s of normal load:
  - CPU burn, 30 s
  - Memory leak, about 4 MB/s for 60 s
  - Disk-write burst, about 20 MB/s for 20 s
  - CPU on/off oscillation, 4 s/4 s for 64 s
  - CPU burn, 15 s
- **Metrics:** Docker's streaming stats API, about 1 sample/s, with the same parsing as `devwaste collect`.
- **Labels:** the workload logs each incident's exact start and end.
- **Training:** default hyperparameters (window 10 ≈ 10 s), 3 training seeds. Detectors are evaluated on the incident phase.
- **Data size:** about 300 training samples and 373 test samples. 65% of test windows touch an incident, leaving only 127 normal test windows.

## Run 1: diagnosis

The first run had a **false-positive rate of about 60%** for both the LSTM and z-score (LSTM F1 0.862, FPR 0.596).

The cause: container memory was almost constant during training (std 0.01 percentage points). After the disk-write burst, page cache kept memory about 0.8 pp higher for the rest of the run. Divided by a std of 0.01, that harmless shift looked like an ~80σ outlier, so every later normal window was flagged.

The fix (commit "Floor per-feature scale…") sets a minimum scale per feature: 1 pp for CPU and memory, and 0.1 log-units for byte rates. The floors do not bind on the synthetic data; Part 1 results were unchanged when re-run.

Because the fix was designed after seeing run 1, re-scoring run 1 is not an unbiased estimate. That re-score gave LSTM FPR 0.276 and z-score 0.102. **Run 2 below is a fresh, independent run made after the fix.**

## Run 2: results (fresh run, after the fix)

Mean ± std over 3 training seeds. Window-level metrics on the incident phase.

| Detector | Precision | Recall | F1 | FPR | ROC-AUC |
|---|---|---|---|---|---|
| LSTM AE | 0.889 ± 0.013 | 0.979 ± 0.000 | 0.932 ± 0.007 | 0.228 ± 0.029 | 0.986 ± 0.001 |
| zscore | 0.951 ± 0.000 | 0.983 ± 0.000 | **0.967** ± 0.000 | 0.094 ± 0.000 | 0.980 ± 0.000 |
| threshold | 0.992 ± 0.000 | 0.553 ± 0.000 | 0.710 ± 0.000 | 0.008 ± 0.000 | n/a |
| rolling | 0.000 | 0.000 | 0.000 | 0.000 | 0.523 |

**Where the false positives occur** (seed 0):

- 26 of the LSTM's 33 false-positive windows start within 15 s after an incident ends. These are recovery transitions, mostly attributed to block writes and CPU.
- Only 2 of z-score's 12 false positives fall in that period.

Excluding windows that start within 15 s after an incident, the FPR is 0.132 for the LSTM and 0.189 for z-score. This is a sensitivity analysis; the strict numbers in the table are the primary result.

## Interpretation

- On these incidents, the simple **z-score baseline is at least as good as the LSTM** (F1 0.967 vs 0.932). Every incident here is a large level change: CPU near 100%, memory up to about 50%, or disk-write rates 1000× normal. A per-value detector handles all of these. The test included no incident whose values stay within the normal range but have an unusual shape, which is the case where the LSTM led on synthetic data.
- The LSTM's extra false positives come from **recovery transitions**: the shape of the return to normal is unlike anything in its training data. Training on data that includes past recoveries, or allowing a short recovery margin, would address this.
- The fixed threshold catches only the CPU incidents, giving recall 0.55.
- The rolling baseline's validation-calibrated threshold flagged nothing in run 2. Rolling statistics are unstable with about 1 s samples and short normal periods.
- **Main practical lesson:** preprocessing assumptions break on real containers. Near-constant metrics and lasting page-cache shifts would not show up in synthetic data. The scale floor is now part of the default pipeline.

## Limitations of Part 2

- One container, one workload, and two runs. Run 2 is the only run not used for design decisions.
- Only 127 normal test windows, so FPR estimates are coarse: each false-positive window is worth about 0.8 percentage points.
- The incidents were injected by us and are large and clean. Real incidents are rarer, subtler, and not labelled.
- Networking was disabled, so the network features are all zero.
