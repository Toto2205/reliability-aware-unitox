# Reliability-Aware Machine Learning for Drug Toxicity Prediction

**Research project | M.Tech Data Science & Artificial Intelligence**

This project investigates whether the reliability of LLM-curated drug-toxicity labels can be estimated and used to improve molecular toxicity prediction. It builds on **UniTox**, the NeurIPS 2024 dataset and benchmark created by applying GPT-4o to FDA drug labels.

> **Research status:** The dataset audit, molecular baseline, custom scaffold-separated evaluation, preliminary DILIrank comparison, and out-of-fold diagnostics have been developed. The complete reliability score and reliability-weighted training experiment are **not yet implemented or validated**. Results below are preliminary, not evidence that reliability-aware training improves performance.

## Table of contents

- [Background](#background)
- [Research question](#research-question)
- [Approach](#approach)
- [Current preliminary findings](#current-preliminary-findings)
- [Repository contents](#repository-contents)
- [Setup](#setup)
- [Dataset setup](#dataset-setup)
- [Running the notebook](#running-the-notebook)
- [Reproducibility and leakage prevention](#reproducibility-and-leakage-prevention)
- [Planned experiments](#planned-experiments)
- [Limitations and responsible use](#limitations-and-responsible-use)
- [References](#references)

## Background

UniTox provides toxicity summaries and binary and ternary toxicity ratings for eight toxicity categories, curated from FDA drug labels with an LLM-assisted pipeline. The original study evaluated the curated labels against external datasets and clinician review, and benchmarked molecular prediction models on a small-molecule subset.

Our work studies a related but distinct question: **even when a dataset is useful overall, might individual labels differ in how strongly they are supported by independent evidence?** If so, estimating label reliability could help a downstream model avoid treating every training label as equally trustworthy.

This project does **not** recreate the original UniTox curation pipeline or claim to improve UniTox yet. It audits and models the released labels and investigates a reliability-aware extension.

## Research question

> Can we estimate the reliability of LLM-curated drug-toxicity labels and use those estimates to improve prediction on previously unseen molecular scaffolds?

For a training example \(i\), let \(y_i\) be its UniTox label, \(\hat y_i\) the model output, and \(R_i\) an estimated reliability weight. The proposed objective is:

\[
\mathcal{L}_{\mathrm{weighted}}
= \frac{1}{N}\sum_{i=1}^{N} R_i\,\mathrm{CE}(y_i,\hat y_i).
\]

This is the research objective, **not yet a completed implementation**. For a scikit-learn estimator such as Random Forest, sample weighting may be implemented through its supported `sample_weight` fit argument; the exact estimator and weighting procedure must be specified in the experiment.

## Approach

The proposed reliability score explores three potential evidence sources:

| Signal | Intended meaning | Current status |
|---|---|---|
| \(A_i\): external-reference agreement | Agreement between a UniTox label and an independently curated reference, when an eligible match exists | Preliminary DILIrank comparison completed; reference is imperfect and coverage is limited |
| \(M_i\): model agreement | Agreement between the UniTox label and a prediction generated without training on that example | Out-of-fold diagnostic generated; usefulness as a reliability signal is not established |
| \(S_i\): molecular-similarity consistency | Consistency between a drug's label and labels of structurally similar drugs | Proposed; not implemented in the current progress report |

An initial candidate formulation is \(R_i=w_1A_i+w_2M_i+w_3S_i\). The components, scaling, missing-evidence policy, weights, and any normalization remain to be determined empirically. We will compare individual signals and combinations rather than assume that every component helps.

### Current workflow

1. Audit the UniTox small-molecule benchmark and target labels.
2. Recompute Morgan fingerprints and compare them with the supplied feature array.
3. Train baseline Random Forest classifiers.
4. Construct scaffold-separated folds using Bemis–Murcko scaffolds.
5. Match eligible drugs to DILIrank using chemical identifiers, including InChIKeys.
6. Generate scaffold-held-out out-of-fold predictions and diagnose model agreement.
7. Implement reliability signals and weighted training only after defining a leakage-safe evaluation protocol.
8. Compare baseline and weighted models on identical held-out folds; perform ablations and sensitivity analyses.

## Current preliminary findings

These values are from the project progress report dated **9 October 2026**. They should be reproduced from the relevant notebooks and data before being treated as final reported results.

### Dataset audit

- 1,349 drugs in the small-molecule liver-toxicity benchmark used in this analysis.
- 613 negative and 736 positive liver-toxicity labels.
- No missing target values, no missing SMILES, and no duplicate SMILES strings were reported in the audited table.
- Morgan fingerprints: 2,048 bits, radius 2; exact equality with the supplied UniTox Morgan array was reported.

### Baseline prediction

| Evaluation | Preliminary ROC-AUC |
|---|---:|
| Liver toxicity, random 10-fold CV | 0.7698 ± 0.0410 |
| Liver toxicity, confident-label subset (878 drugs) | 0.8306 ± 0.0365 |
| Liver toxicity, custom scaffold-separated CV | 0.6971 ± 0.0693 |
| Custom 10-fold scaffold OOF predictions, pooled | 0.7204 |

The confident-only evaluation uses a different population, so the higher score does not by itself prove that filtering labels caused an improvement. The custom scaffold split is project-defined and is **not claimed to be an exact reproduction** of the original UniTox scaffold-balanced split.

### DILIrank comparison

For 116 exact chemical matches in the clear `No` and `Most` DILIrank categories, with `Less` excluded:

- Raw agreement: **86.21%**
- Sensitivity: **96.36%**
- Specificity: **77.05%**

This is agreement within a selected subset, not proof that UniTox labels are 86.21% reliable across all drugs. DILIrank is a useful external reference, not infallible ground truth, and excluding `Less` limits generalization.

### Model-agreement diagnostic

- The scaffold-held-out model agreed with UniTox for 907/1,349 drugs (67.23%).
- Among the 116 clear-category external matches, the association between model agreement and external-reference agreement was weak and not statistically significant (Fisher's exact \(p=0.5787\); Cohen's \(\kappa=0.0419\)).

Therefore, model agreement should not be assumed to be a valid reliability signal. Its incremental value needs to be tested, and it may be removed if ablations do not support it.

### What has not been demonstrated

**No result currently establishes that reliability-weighted training outperforms ordinary training.** The reliability score, weighted-model experiment, ablations, and final controlled evaluation remain future work.

## Repository contents

A suggested GitHub structure is:

```text
reliability-aware-unitox/
├── README.md
├── requirements.txt
├── notebooks/
│   └── 01_UniTox_Dataset_Audit.ipynb
├── data/
│   ├── UniTox-Small-Molecule-Benchmark.csv
│   ├── UniTox-Small-Molecule-Benchmark-morgan.npz
│   ├── UniTox-Small-Molecule-Benchmark-rdkit.npz
│   └── README.md
├── outputs/
│   └── .gitkeep
└── .gitignore
```

The names above are a suggested organization. Adjust them to match the files actually committed. Keep large datasets and generated artifacts out of Git when their license, size, or distribution terms make that appropriate. If data are not committed, provide clear instructions for obtaining them from the official UniTox source.

## Setup

Python **3.10 or 3.11** is a sensible starting point for this scientific Python stack. RDKit installation can vary by operating system; if pip installation fails, use the official RDKit conda-forge installation instructions.

### Option A: virtual environment and pip

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd <YOUR-REPOSITORY-DIRECTORY>

python -m venv .venv
```

Activate the environment:

**Windows PowerShell**
```powershell
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux**
```bash
source .venv/bin/activate
```

Install dependencies and register the notebook kernel:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name unitox-reliability --display-name "Python (UniTox Reliability)"
```

If RDKit fails to install with pip, create a conda environment instead:

```bash
conda create -n unitox-reliability -c conda-forge python=3.11 rdkit numpy pandas scipy scikit-learn matplotlib jupyter ipykernel
conda activate unitox-reliability
```

## Dataset setup

Obtain UniTox data and code from the official project sources listed below. Respect the original dataset's terms and cite the paper.

The current audit notebook initially defines a Windows-specific project path:

```python
PROJECT_ROOT = r"D:\pro dataset\UniTox-main"
```

**Update this path before running the notebook.** Prefer a repository-relative path, for example:

```python
from pathlib import Path

PROJECT_ROOT = Path.cwd()  # if the notebook is launched from the UniTox repository root
DATA_DIR = PROJECT_ROOT / "Data"

CSV_PATH = DATA_DIR / "UniTox-Small-Molecule-Benchmark.csv"
MORGAN_PATH = DATA_DIR / "UniTox-Small-Molecule-Benchmark-morgan.npz"
RDKIT_PATH = DATA_DIR / "UniTox-Small-Molecule-Benchmark-rdkit.npz"
```

If your notebook is stored in a `notebooks/` subdirectory, set `PROJECT_ROOT` to the parent directory that actually contains `Data/`. Check the filenames against the downloaded release; do not assume every release uses identical paths.

DILIrank is used for an external comparison in the analysis. Obtain it from its authoritative source, document the version/date, and keep the matching and category-selection procedure reproducible. Do not upload restricted or redistributed data without checking its terms.

## Running the notebook

1. Launch Jupyter Lab or Jupyter Notebook from the project environment.
2. Select the `Python (UniTox Reliability)` kernel.
3. Update dataset paths and verify that all required input files exist.
4. Run `01_UniTox_Dataset_Audit.ipynb` from top to bottom.
5. Check the dataset audit, fingerprint equality, baseline scores, scaffold-fold validation, and external-match counts before interpreting downstream results.

The notebook currently includes data auditing, fingerprint checks, baseline evaluation, scaffold fold creation, DILIrank matching, and out-of-fold diagnostics. Notebook outputs may depend on the exact data version, package versions, split construction, and execution order.

## Reproducibility and leakage prevention

Reliability estimation must not use information from the outer held-out test fold. In particular:

- For each outer scaffold split, calculate training-example reliability weights without using the held-out fold's labels or external annotations.
- Do not use the same external annotations both to construct training weights and to claim independent success on those same evaluation examples.
- Exclude a molecule's own label when calculating its neighborhood-based similarity signal.
- Use identical outer folds, test examples, model settings, and metrics when comparing ordinary and reliability-weighted training.
- Tune the reliability method only within training data; reserve final test folds for evaluation.
- Report ROC-AUC, PR-AUC, calibration where appropriate, uncertainty estimates, and per-fold results.
- Run ablations for baseline training, confident-only training, each individual signal, and combinations of signals.
- Document ambiguous chemical identity matches, duplicate identifiers, and any exclusions.

Scaffold separation reduces the risk of overly optimistic evaluation caused by close structural analogues appearing in both training and test sets. It does not by itself eliminate every form of leakage.

## Planned experiments

- [ ] Finalize and freeze the evaluation protocol and random seeds.
- [ ] Investigate duplicate chemical identifiers and document resolution rules.
- [ ] Implement \(S_i\) using Morgan-fingerprint Tanimoto similarity and local label consistency.
- [ ] Evaluate whether each reliability signal is associated with independently assessed label quality.
- [ ] Define a leakage-safe reliability score and missing-evidence policy.
- [ ] Train the first sample-weighted model.
- [ ] Compare against ordinary training on the same scaffold-held-out folds.
- [ ] Run component ablations and weight/neighborhood sensitivity tests.
- [ ] Report PR-AUC, calibration, uncertainty, and robustness.
- [ ] Extend to other toxicity categories after the liver-toxicity experiment is stable.

## Limitations and responsible use

- UniTox labels were generated with LLM assistance and may contain errors or ambiguity.
- External reference datasets have their own coverage, matching, and classification limitations.
- A model's disagreement with a label can reflect structural novelty or poor generalization rather than an incorrect label.
- Similarity-based signals can amplify biases in the dataset if not evaluated carefully.
- Results are research findings, not medical advice. Do not use these models to make clinical decisions or assess the safety of a drug for an individual.

## References

1. Silberg, J. et al. **“UniTox: Leveraging LLMs to Curate a Unified Dataset of Drug-Induced Toxicity from FDA Labels.”** *Advances in Neural Information Processing Systems 37 (NeurIPS 2024), Datasets and Benchmarks Track.*  
   Paper: https://papers.nips.cc/paper_files/paper/2024/file/16659e412de3965fa195ddb9f2c4b356-Paper-Datasets_and_Benchmarks_Track.pdf  
   Official code: https://github.com/jsilbergDS/UniTox  
   UniTox website/data: https://zou-group.github.io/UniTox-website/

2. Chen, M. et al. **“DILIrank: The largest reference drug list ranked by the risk for developing drug-induced liver injury in humans.”** *Drug Discovery Today* (2016). Cite the version of DILIrank actually used in your experiment.

## Citation

If you use UniTox, cite the original paper above. This repository studies a proposed reliability-aware extension and should be cited separately only once a citable manuscript or release is available.

---

**Project status:** Research prototype in progress. Reliability-aware performance improvement remains an open experimental question.
