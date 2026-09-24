# 🧠 NeuroSeg-U: Robust Brain Tumor Segmentation with U-Net

> **Research-oriented medical imaging project for PhD/RA applications**

NeuroSeg-U is a reproducible research framework for **brain tumor segmentation from multi-modal MRI** using a strong U-Net baseline and research extensions. The project is designed to move beyond a classroom U-Net implementation by including patient-level splitting, leakage checks, robust preprocessing, composite losses, uncertainty estimation, ablations, calibration, statistical reporting, and comparison against a modern self-configuring segmentation baseline.

## Research question

**How can a U-Net-based segmentation system remain accurate and reliable when MRI intensity distributions, tumor sizes, and acquisition conditions vary across patients?**

### Hypotheses
1. Dice + boundary-aware loss improves small/irregular tumor boundary segmentation over Dice-only training.
2. Modality dropout improves robustness when one MRI sequence is degraded or unavailable.
3. Test-time augmentation and uncertainty maps identify clinically ambiguous regions and correlate with segmentation error.
4. A carefully controlled 2D/2.5D U-Net provides a useful, reproducible baseline against which stronger 3D methods can be compared.

##  Research-orientation

This repo treats segmentation as an **experimental study**, not merely a model demo:

- patient-level train/validation/test splitting
- reproducible seeds and experiment configs
- NIfTI-aware preprocessing
- multi-modal MRI support (T1, T1ce, T2, FLAIR)
- residual/attention U-Net variants
- Dice + BCE + boundary loss
- deep-supervision-ready model interface
- Dice, IoU, sensitivity, specificity, HD95 and ASSD
- calibration and uncertainty analysis
- robustness experiments with modality dropout
- ablation study templates
- bootstrap confidence intervals
- experiment manifests and checkpoint metadata
- automated tests and CI
- model card + research limitations

## Dataset

The intended primary benchmark is **BraTS**, a major benchmark family for brain-tumor segmentation. Use the official challenge/data access route and follow its license/usage terms. Do **not** commit patient data, NIfTI volumes, predictions, or private clinical information to GitHub.

## Architecture

```text
MRI modalities
 T1 ─┐
 T1c ├──> preprocessing ──> 2.5D input ──> U-Net encoder
 T2 ─┤                                      │
 FLAIR┘                                      ▼
                                  bottleneck + attention
                                              │
                                              ▼
                                      U-Net decoder
                                              │
                                  tumor probability map
                                              │
                         ┌────────────────────┴──────────────────┐
                         ▼                                       ▼
                    segmentation                         uncertainty map
```

## Research pipeline

```text
Raw NIfTI
  ↓
Dataset audit → orientation/shape checks → missing-modality checks
  ↓
Brain/tumor-aware normalization
  ↓
Patient-level split (no slice leakage)
  ↓
2D/2.5D U-Net baseline
  ↓
Composite loss + augmentation + modality dropout
  ↓
Validation model selection
  ↓
Held-out test evaluation
  ↓
Bootstrap CIs + calibration + uncertainty
  ↓
Ablation / robustness study
  ↓
Research report
```

## Quick start

```bash
git clone https://github.com/showravj2-create/Nuroserg_U_r
cd Nuroserg_U_r
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

### BraTS 2020-ready workflow

This repo now supports real BraTS 2020 MRI datasets, legacy lab-style naming, and Colab/cloud workflows.

- Run dataset validation: `python scripts/check_brats_dataset.py --data-root /path/to/BraTS2020`
- Train on BraTS: `python scripts/train.py --config configs/brats_2p5d.yaml`
- Colab-friendly runner: `python colab_run.py --dataset-root /content/BraTS2020 --config configs/brats_2p5d.yaml`
- V2 project snapshot: see [v2/README.md](v2/README.md)

Run the unit tests:

```bash
pytest -q
```

Train a smoke-test model using synthetic data:

```bash
python scripts/train.py --config configs/smoke.yaml
```

Train on a prepared dataset:

```bash
python scripts/train.py --config configs/brats_2p5d.yaml
```

Evaluate:

```bash
python scripts/evaluate.py --config configs/brats_2p5d.yaml --checkpoint runs/best.pt
```

## Suggested experiments

| ID | Experiment | Purpose |
|---|---|---|
| E0 | vanilla U-Net + Dice | baseline |
| E1 | Dice + BCE | stabilize foreground/background learning |
| E2 | Dice + BCE + boundary | improve contours |
| E3 | E2 + modality dropout | missing/degraded modality robustness |
| E4 | E3 + test-time augmentation | uncertainty/robustness |
| E5 | attention/residual U-Net | architecture ablation |
| E6 | 2D vs 2.5D | context ablation |
| E7 | U-Net vs nnU-Net | strong external baseline |

## Evaluation

Primary metric:

\[
Dice = \frac{2|P \cap G|}{|P| + |G|}
\]

Also report:

- IoU / Jaccard
- sensitivity / recall
- specificity
- precision
- HD95
- ASSD
- lesion-wise detection rate
- calibration error
- bootstrap 95% confidence intervals

**Important:** never claim clinical utility from a retrospective benchmark alone. This repository is a research prototype, not a diagnostic medical device.

## Reproducibility checklist

- [ ] Record dataset release/version.
- [ ] Record preprocessing parameters.
- [ ] Split by patient, never by individual slices.
- [ ] Fix random seeds.
- [ ] Save configuration with every run.
- [ ] Save git commit hash.
- [ ] Report hardware and software versions.
- [ ] Report all failed/aborted experiments.
- [ ] Keep the final test set untouched until model selection is complete.
- [ ] Report confidence intervals and not only a single Dice score.

## Research extension roadmap

### Phase 1 — credible baseline
U-Net + Dice/BCE, correct patient-level split, reproducible preprocessing.

### Phase 2 — research contribution
Boundary-aware loss + modality dropout + attention/residual blocks.

### Phase 3 — reliability
Monte-Carlo dropout / TTA uncertainty, calibration and failure-case analysis.

### Phase 4 — benchmark study
Compare with nnU-Net under a carefully controlled evaluation protocol.

### Phase 5 — paper-quality analysis
Bootstrap confidence intervals, statistical comparisons, subgroup analysis by tumor size, and an error taxonomy.

## Ethical and medical disclaimer

This project is for research and educational use. Segmentation outputs can be wrong and must not be used for diagnosis or treatment decisions. Public datasets may have different acquisition protocols, demographics, and annotation practices than a real clinical population.

## Core references

- Ronneberger, Fischer & Brox. **U-Net: Convolutional Networks for Biomedical Image Segmentation** (2015).
- Isensee et al. **nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation** (Nature Methods, 2021).
- BraTS challenge resources and official evaluation tooling.

## Author

**Showrav Das** — BSc (Hons) Mathematics | Aspiring ML/Medical Imaging Researcher
