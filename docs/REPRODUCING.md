# Reproducing the analysis and experiments

All commands below run from the package root. Analysis outputs are separate from the frozen report and recorded data.

## 1. Rebuild the recorded results on CPU

```sh
python -m pip install -r requirements-analysis.txt
python -m src.verify
python -m src.analyze --output reproduced/analysis
python -m src.figures --output reproduced/figures
```

Verification checks file hashes, the complete CSV inventory, and evaluation, gradient, and perturbation grids. The analysis produces:

- `summary.json`: all final outcomes, success criteria, early internal measurements, and gradient peaks.
- `coupling.json`: all phase-wise local-response counts and the Figure 4 example.
- `plateau.json`: prompt polarization, stabilized aggregate metrics, and gradient decay in the L15 plateau.

The plotting command rebuilds all eight report figures as PNG files. It uses no model checkpoints or GPU. The report's PDF and HTML remain unchanged. Font rendering may vary between operating systems; numerical curves come from the same measurements.

## 2. Check the mathematical implementation

Install a compatible PyTorch build to run the model checks. The recorded experiments used Python 3.12.14, PyTorch 2.11.0 with CUDA 12.8, and an NVIDIA GeForce RTX 4090 on a rented Linux instance. `requirements-training.txt` pins the PyTorch release; select the CUDA build appropriate for the execution environment when installing it.

```sh
python -m pip install -r requirements-training.txt
python -m unittest discover -s tests -v
python -m src.analyze_oracle --output reproduced/analysis/oracle.json
```

The tests compare finite-state recursion and reward gradients with explicit trajectory enumeration, check transition factorization and the centered adjoint, and check task construction. The oracle analysis evaluates the actual frozen head on all 9,216 state-operation pairs and estimates full-path capacity under perfect operation selection. It does not train a policy.

## 3. Train a fresh run

The supplied head reproduces the experimental starting model. A new output directory is required; training refuses to overwrite an existing run.

```sh
python -m src.train --config configs/mixed5_15_45_seed101.json --head model/transition_head.pt --output runs/mixed5_15_45_seed101 --device cuda
```

Choose another file in `configs/` for fixed-L45 training, seeds 102 or 103, or the five-level mixture. Every configuration uses 40,000 updates, evaluation every 50 updates, Q snapshots every 50 updates, and recovery checkpoints every 2,000 updates. The training objective is length-normalized REINFORCE with an EMA baseline and entropy bonus. Adam uses a constant learning rate of 0.01; there is no cosine learning-rate schedule.

All scientific settings match the retained runs. Package configurations use descriptive labels and a ten-day wall-time guard so the original instance-specific limit does not interrupt a slower replay. The completed step count, not the wall-time guard, defines the experiment. Hardware and software differences can change a fresh stochastic trajectory even with the same seed.

Training writes `train.jsonl`, `eval.jsonl`, the run configuration, environment metadata, Q snapshots, and recovery checkpoints. These are outputs of the new run, not part of the distributed presentation package.

## 4. Measure exact gradients and local responses

After the mixed run completes:

```sh
python -m src.diagnose_markov --run runs/mixed5_15_45_seed101 --head model/transition_head.pt --output runs/gradient_mixed_seed101 --interval 100 --gradients-only --device cuda --max-seconds 864000
python -m src.diagnose_markov --run runs/mixed5_15_45_seed101 --head model/transition_head.pt --output runs/transfer_mixed_seed101 --interval 2000 --epsilon 0.01 0.005 --device cuda --max-seconds 864000
```

Defaults match the original protocol: lengths 5/15/45, three fixed sets of 16 prompts per length, and float64 measurement. The first command records the exact reward-gradient norm every 100 updates. The second measures source lengths 5 and 15 against target length 45. Each positive and negative displacement starts from the same stored Q; perturbations never accumulate or alter training.

Run the gradient command against each fixed-L45 run as well. For the five-level run, use `src.diagnose_ratio` with the same command options. Its defaults evaluate lengths 5/10/15/20/40/45 and test sources 5/10/15/20/40 against target 45. To extend a fixed-L45 run to those six lengths, use `src.diagnose_ratio --gradients-only` on its saved snapshots.

Diagnostics write `measurements.jsonl` and a manifest containing the measured checkpoint grid and completion status. All gradients use the same frozen transition head as the corresponding training run; a head-hash check enforces this. A time-limit or failed status is not a completed experiment.

## 5. Implementation map

| Module | Responsibility |
|---|---|
| `src/model.py` | Synthetic tasks, attention policy, and frozen atomic transition head |
| `src/train.py` | Training objective, evaluation, RNG separation, and state recording |
| `src/markov.py` | Exact finite-state terminal probabilities and centered-gradient recursion |
| `src/diagnose_markov.py` | Three-length gradients and paired finite perturbations |
| `src/diagnose_ratio.py` | Configurable-length version for the five-level control |
| `src/pretrain.py` | Atomic-head pretraining |
| `src/analyze*.py` | Numerical summaries, plateau analysis, and oracle-capacity check |
| `src/plot_*.py` | Figure construction from recorded measurements |

Pretraining is optional because the original weights are already included. To reproduce its procedure independently:

```sh
python -m src.pretrain --output runs/new_head --d 96 --units-per-class 384 --steps 6000 --batch-size 1024 --lr 0.003 --seed 17 --initialization random --negative-slope 0.01 --eval-every 500 --target-probability 1.0 --device cuda
```

Use `model/transition_head.pt` for comparisons with this report. A newly pretrained head defines a new experimental starting point.
