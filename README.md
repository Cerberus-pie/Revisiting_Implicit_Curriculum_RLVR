# Revisiting Implicit Curriculum in RLVR

**Reproduction and local analysis of synthetic state tracking**  
Yankai Ding · Report v21

A fixed mixture of easy and hard tasks can bootstrap learning on harder tasks. This study independently implements the synthetic experiment of Huang et al. and examines that process through attention, local transition probabilities, exact reward gradients, and cross-length perturbations.

## Read the report

- [PDF report](report/report.pdf) — six main pages and five appendices.
- [HTML report](report/report.html) — the same report, with all figures embedded; opens offline.
- [Data and figure guide](docs/DATA.md) — what each measurement means and which files support each figure.
- [Reproduction guide](docs/REPRODUCING.md) — rebuild the analysis, inspect the implementation, or rerun training.

## What the experiments show

Mixed training enables success on more hard tasks. In the main run, L45 terminal success rises from 1.03% under fixed training to 11.65% under mixed training. Another seed reaches 66.17%; one remains near chance. The results show hard-task bootstrapping with substantial variation, at lower final accuracies than the near-one levels reported across lengths in the original experiment.

Our local measurements reveal improving attention and local execution before terminal success rises appreciably. Reward-gradient norms peak in length order; later, L15 ascent directions help L45 more often even as the L15 signal weakens. Sharpening attention, polarized prompt outcomes, and shrinking gradients characterize saturation within the main run's partially learned policy. How the learned transition network and actual optimizer updates produce these different learning states remains open.

Seed 101 is the main example. Seeds 102 and 103 and an additional five-level mixture provide the comparisons in the appendices and data.

## Rebuild from the recorded measurements

From this directory, using Python 3.12 or newer:

```sh
python -m pip install -r requirements-analysis.txt
python -m src.verify
python -m src.analyze --output reproduced/analysis
python -m src.figures --output reproduced/figures
```

These commands run on CPU and require no model download, GPU, account, or network connection after installation. They validate the package, recompute the principal numerical summaries, and rebuild all eight report figures from recorded data. Curves use the original measurement grid without smoothing or added samples.

The supplied PDF and HTML are the v21 documents. Rebuilt figures and analysis are written separately under `reproduced/`.

## Package structure

```text
report/       Frozen PDF and self-contained HTML
data/         Recorded measurements, source hashes, and provenance
model/        The frozen transition head used in the experiments
configs/      Seven training configurations and head pretraining settings
src/          Model, training, exact diagnostics, analysis, and figure code
tests/        Enumeration, gradient, and task-construction checks
docs/         Data dictionary and reproduction instructions
MANIFEST.json SHA256 inventory of the distributed files
```

The measurement archive contains evaluations and local diagnostics for every reported condition, plus per-update seed-101 measurements used to examine training fluctuations and the plateau. The frozen head is included so a fresh training run uses the same atomic model. Full optimizer histories and policy snapshots are not needed to reconstruct the report and are not included; fresh training writes new snapshots for its own local measurements.

## Implementation and attribution

The implementation is independent of the original authors' code. The original experiment supplies the reproduction target; the trajectory-level measurements, reward-gradient tracking, and finite local perturbations are the additional analyses developed for this study. See Sections 3–5 and the implementation appendix for their definitions.

Source paper: Yu Huang et al., *On the Emergence of Implicit Curriculum in RLVR Learning Dynamics*, ICML 2026, PMLR 306:47891–47938. [Conference paper](https://proceedings.mlr.press/v306/huang26bk.html) · [Experimental exposition, Section 7.1](https://arxiv.org/html/2602.14872v3#S7.SS1).

OpenAI Codex assisted extensively with experimental design, code, execution, analysis, figures, and writing. The author directed the study and is responsible for the final report.
