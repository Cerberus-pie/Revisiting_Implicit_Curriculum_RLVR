# Measurements and figures

## Experimental conditions

| Condition | Training lengths | Evaluation lengths | Seeds |
|---|---|---|---|
| Fixed | 45 | 45 | 101, 102, 103 |
| Three-level mixture | 5, 15, 45 | 5, 15, 45 | 101, 102, 103 |
| Five-level mixture | 5, 10, 20, 40, 45 | 5, 10, 15, 20, 40, 45 | 101 |

Mixtures sample the listed training lengths uniformly. L is the number of operations in a task. The state space has 96 values; the operation-identifier space has 64 positions. All runs share the same frozen transition head and train only the attention parameter matrix Q for 40,000 updates.

The five-level condition is an added control, motivated by a ratio of two between successive lengths and capped at 45. Its final 40-to-45 interval is truncated. L15 is an evaluation-only length in this condition. It is not a reproduction of a published five-level experiment.

## Archive layout

`data/measurements.zip` contains CSV files under two prefixes:

- `runs/<condition>_seed<seed>/eval.csv`: one row per checkpoint and evaluated length, every 50 updates from 0 through 40,000. Each row aggregates 15,360 prompts.
- `runs/<condition>_seed101/train.csv`: one row per actual update for the fixed and three-level runs, used in Figure 5 and the noncanonical-path analysis.
- `diagnostics/<measurement>/measurements.csv`: exact reward gradients or finite perturbation results. A small adjacent `manifest.json` identifies the run.

Dense gradients have 401 checkpoints, three fixed prompt sets, and 16 prompts per length per set. Transfer measurements have 21 checkpoints, every 2,000 updates, and both perturbation sizes 0.01 and 0.005. Transfer files also retain the gradients measured at those checkpoints.

The seed-101 fixed-run gradient recording was added through an otherwise identical same-seed rerun. Its 801 evaluation checkpoints matched the initial run. The six-length fixed-control extension reused that rerun's Q snapshots. The report describes this provenance in Appendix A. Measuring L5 or L15 at a fixed-L45 checkpoint evaluates the current model on those tasks; it does not imply it trained on those lengths.

## Main fields

| Field | Meaning |
|---|---|
| `step` | Number of completed training updates |
| `length` | Evaluated task length |
| `greedy_success` | Fraction with the correct final state under greedy state selection |
| `greedy_trajectory_correct` | Fraction whose entire greedy state sequence is canonical |
| `sampled_success` | Terminal success under sampled state selection |
| `attention_hit` | Fraction of steps whose highest-attention operation is the target operation |
| `attention_mass` | Mean attention weight assigned to the target operation |
| `correct_transition_prob` | Mean probability of applying the target operation correctly from the current sampled state |
| `reward_gradient_norm` | Frobenius norm of the gradient of mean terminal probability divided by task length |
| `per_prompt_probabilities` | JSON array of the 16 exact terminal probabilities for the fixed prompt set |
| `repeat` | Prompt-set index: 0, 1, 2 correspond to A, B, C |
| `source`, `target` | Source gradient length and target reward length in a perturbation |
| `gradient_cosine` | Directional alignment between source and target reward gradients |
| `positive_change` | Target terminal probability after a positive unit-gradient step minus its unperturbed value |
| `negative_change` | Target terminal probability after the matching negative step minus its unperturbed value |
| `finite_difference_resolved` | Whether the symmetric difference meets the numerical derivative tolerance |
| `gradient_norm` in training CSV | Norm of the realized training loss gradient, including the training objective's terms |
| `parameter_update_norm` | Norm of the actual Adam parameter update |

Attention mass and local transition probability average over prompts and operation steps. The latter is evaluated from the sampled state, which need not be the canonical state. Neither quantity is a full-trajectory success probability.

An exact reward gradient integrates out sampled state trajectories for a fixed prompt set. It is exact for that set, not a population expectation over all possible prompts. It excludes the learning rate, Adam state, and entropy bonus. The cosine measures alignment in parameter space, not synchronization of two time series.

## Figure-to-data map

| Figure | Question | Data | Builder |
|---|---|---|---|
| 1 | Does mixed training improve terminal performance? | Seed-101 fixed and mixed evaluations | `plot_terminal.py` |
| 2 | What changes internally before terminal success rises? | Seed-101 L45 attention, transition, and sampled-success evaluations | `plot_internal.py` |
| 3 | How do reward-gradient strengths evolve under the two conditions? | Seed-101 mixed and fixed gradient measurements | `plot_gradients.py` |
| 4 | Do easier-task ascent directions also help L45? | Seed-101 mixed perturbations, epsilon 0.01 | `plot_gradients.py` |
| 5 | Do apparently smooth evaluation curves conceal update-level fluctuations? | Both seed-101 training CSV files | `plot_updates.py` |
| 6 | Does the incomplete relay recur across seeds? | All three seeds' evaluations and mixed gradients | `plot_seeds.py` |
| 7 | Does operation selection stabilize in the partial solution? | Seed-101 attention-hit evaluations | `plot_terminal.py` |
| 8 | What changes with a denser length mixture? | Fixed, three-level, and five-level seed-101 evaluations | `plot_mixture.py` |

`python -m src.figures` invokes these builders together. The eight plots are not averaged across training seeds. Gradient bands show the range across the three prompt sets; these are not confidence intervals across training runs. All measured points in the stated display windows are drawn. The Figure 2 zoom and the coupling-summary phase windows were chosen after observing the trajectory.

## Integrity and provenance

`data/provenance.json` maps each CSV to its source archive and member hash, lists every retained column and row count, and records implementation adaptations. CSV bytes are unchanged. Source archives are identified by neutral aliases and hashes rather than machine locations.

`MANIFEST.json` hashes every distributed file. The checkpoint was re-serialized to remove launch and timing metadata; its parameter tensors were checked for bitwise equality with the experimental head. Tensor hashes and the original checkpoint hash are retained in the provenance record.
