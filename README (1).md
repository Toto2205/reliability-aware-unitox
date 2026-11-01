# Reliability-Aware Machine Learning for Drug Toxicity Prediction

**Project group:** GOD'S MERCY  
**Authors:** Amankumar Sushil Dubey (M26DS001) and Deepajyoti Chakraborty (M26DS006)  
**Programme:** M.Tech. Data Science and Artificial Intelligence, Indian Institute of Technology Bhilai

This repository contains the completed project workflow extending **UniTox** with example-level reliability estimation and reliability-aware Random Forest training. The notebook includes data auditing, molecular identity curation, leakage-aware grouped evaluation, reliability estimation, model comparison, independent DILIrank validation, ablation experiments, and statistical analysis.

## 1. Research objective

LLM-curated toxicity labels can contain uncertainty and disagreement. This project investigates whether the reliability of each labelled compound can be estimated from complementary evidence and used to weight model training.

**Research question:** Does reliability-aware sample weighting improve liver-toxicity prediction on unseen molecular scaffolds compared with an otherwise matched Random Forest baseline?

The completed implementation is an experimental evaluation. It does **not** establish that reliability weighting improves predictive performance: internal changes were small and not statistically significant, while the external DILIrank prediction comparison numerically favoured the plain Random Forest.

## 2. Relationship to the UniTox base paper

The base paper is:

> Silberg et al. (2024). *UniTox: Leveraging LLMs to Curate a Unified Dataset of Drug-Induced Toxicity from FDA Labels*. NeurIPS 2024.

UniTox builds a curated drug-toxicity resource from FDA drug labels. Its workflow includes label collection and screening, GPT-4o-assisted summarisation and toxicity-specific evidence extraction, validation, and a molecular prediction benchmark. The resource covers eight toxicity categories and includes a 1,349-compound small-molecule benchmark.

**Our extension:** rather than proposing a new FDA-label curation pipeline, this project uses the UniTox benchmark and adds three reliability signals—external reference agreement, cross-fitted model support, and molecular-neighbour consistency—then evaluates reliability-based sample weighting.

## 3. Dataset and molecular representation

The final focused experiment uses the UniTox small-molecule benchmark for the binary liver-toxicity endpoint:

- **Compounds:** 1,349
- **Negative labels:** 613
- **Positive labels:** 736
- **Target:** `liver_toxicity_binary_rating_0_1`
- **Features:** official UniTox Morgan fingerprints, radius 2, 2,048 bits
- The notebook independently regenerates the Morgan fingerprints with RDKit and verifies agreement with the official feature array.

The notebook also retains the dataset audit and preliminary baseline workflow. The optional all-endpoint preliminary baseline is disabled by default.

## 4. Chemical identity audit and evaluation split

To reduce leakage between related compounds, the workflow audits original InChIKeys, standardized parent structures, fragment composition, and Bemis–Murcko scaffolds.

- 1,164 unique standardized parent InChIKeys were identified.
- 160 multi-record parent groups covered 345 rows.
- 36 parent groups contained conflicting liver labels; the labels were retained rather than averaged.
- The final curated grouping produced **774 components**.
- Ten outer folds were created using `StratifiedGroupKFold`, with frozen seed **397** selected using fold-size and class-balance criteria—not model performance.
- The adopted grouping produced zero crossings of curated components, Murcko scaffolds, and accepted parent-identity groups across outer train/test folds.

**Curation limitation:** 27 remaining parent-identity bridges were provisionally classified under the adopted policy and require further expert chemical review. Two false parent merges were excluded. See the notebook's identity-bridge policy and fragment audit sections for the implemented decisions.

## 5. Reliability-aware model

For compound \(i\), the project calculates three signals:

### Aᵢ — independent external-reference agreement

DILIrank SMILES are converted to full InChIKeys with RDKit and matched to UniTox by exact full InChIKey.

- DILIrank `No` agrees with UniTox label 0.
- DILIrank `Most` agrees with UniTox label 1.
- Opposite extreme combinations receive 0.
- `Less` and unmatched cases are treated as **missing**, not as disagreement.

There were 122 evaluable No/Most cases: 105 agreements and 17 disagreements (86.07% agreement). DILIrank is an imperfect external reference, not ground truth.

### Mᵢ — cross-fitted model support

\(M_i = \hat P(Y=y_i \mid X_i)\), the probability assigned to the observed UniTox label by a Random Forest that did not train on the query's inner validation component.

Within each outer training fold, five-fold `GroupKFold` cross-fitting is performed by curated component. The implementation generated 12,141 training-score rows; mean continuous M was 0.56029 (SD 0.16610), and hard agreement was 64.739%.

### Sᵢ — molecular-neighbour consistency

For each query, up to ten positive-similarity neighbours are retrieved using Tanimoto similarity over the official Morgan fingerprints. Neighbours from the same curated component and the exact original InChIKey are excluded. S is the similarity-weighted fraction of eligible neighbours sharing the query's observed label.

The implementation generated 12,141 training-score rows; mean S was 0.57167 (SD 0.22340).

### Combined reliability and sample weights

Available signals are averaged with equal component weights; missing A is omitted rather than treated as zero:

\[
R_i = \frac{m_{iA}A_i + M_i + S_i}{m_{iA}+2},
\]

where \(m_{iA}=1\) when A is available and 0 otherwise.

Raw sample weights are:

\[
u_i = 0.10 + 0.90R_i.
\]

Weights are normalized to mean 1 within each outer training fold. The weighted Random Forest receives these values through its `sample_weight` fit argument.

## 6. Experimental architecture and protocol

**Molecular input:** 2,048-bit Morgan radius-2 fingerprints.

**Outer evaluation:** 10-fold `StratifiedGroupKFold` on the 774 curated identity/scaffold components (seed 397).

**Inner reliability estimation:** five-fold `GroupKFold` within each outer training fold for M and S. External-reference agreement A is derived from exact DILIrank matching. Outer-test labels are not used to construct the corresponding outer-training weights.

**Matched models:**

- Plain Random Forest: 500 trees, `random_state=42`, `n_jobs=-1`, no sample weights.
- Reliability-weighted Random Forest: same features, folds, model settings, and test compounds, trained with fold-normalized reliability weights.

Each compound receives exactly one outer-held-out prediction. Metrics include ROC-AUC, PR-AUC, accuracy, balanced accuracy, precision, recall, specificity, F1, log loss, and Brier score.

## 7. Main results: grouped outer-held-out evaluation

| Metric | Plain RF | Reliability-weighted RF |
|---|---:|---:|
| ROC-AUC | 0.6937 | 0.6975 |
| PR-AUC | 0.6906 | 0.6962 |
| Accuracy | 0.6612 | 0.6642 |
| Balanced accuracy | 0.6469 | 0.6481 |
| Precision | 0.6541 | 0.6520 |
| Recall | 0.8043 | 0.8247 |
| Specificity | 0.4894 | 0.4715 |
| F1-score | 0.7215 | 0.7283 |
| Log loss | 0.6318 | 0.6329 |
| Brier score | 0.2200 | 0.2201 |

The weighted model's pooled ROC-AUC increased by 0.00381 and PR-AUC by 0.00556. Fold-mean ROC-AUC increased from 0.6851 to 0.6905 (7/10 folds improved); fold-mean PR-AUC increased from 0.7035 to 0.7117 (7/10 improved).

However, paired fold tests were not statistically significant:

- ROC-AUC: paired t-test p = 0.28990; Wilcoxon p = 0.37500.
- PR-AUC: paired t-test p = 0.11085; Wilcoxon p = 0.08398.

These ten-fold tests are exploratory because the training sets overlap.

## 8. Independent DILIrank validation

For the 122 held-out No/Most cases, reliability association with external agreement was:

| Signal | ROC-AUC | Average precision | Spearman ρ (p) |
|---|---:|---:|---:|
| M | 0.6202 | 0.9171 | 0.1442 (0.1131) |
| S | 0.6487 | 0.9203 | 0.1784 (0.0492) |
| M + S | 0.6459 | 0.9220 | 0.1751 (0.0538) |

The high average-precision values are influenced by the 86.1% agreement prevalence. Confidence intervals were wide; the small sample and multiple signals warrant cautious interpretation.

The separate task of predicting DILIrank No versus Most produced:

| Metric | Plain RF | Weighted RF |
|---|---:|---:|
| ROC-AUC | 0.7877 | 0.7776 |
| PR-AUC | 0.7873 | 0.7806 |
| Accuracy | 0.6475 | 0.6311 |
| Balanced accuracy | 0.6726 | 0.6577 |
| Recall | 0.9273 | 0.9273 |
| Specificity | 0.4179 | 0.3881 |
| Log loss | 0.6238 | 0.6424 |
| Brier score | 0.2182 | 0.2275 |

The plain RF performed numerically better. Paired bootstrap confidence intervals for weighted-minus-plain differences included zero for both ROC-AUC ([-0.0347, 0.0151]) and PR-AUC ([-0.0348, 0.0212]).

## 9. Ablation and statistical inference

| Variant | Pooled ROC-AUC | Pooled PR-AUC |
|---|---:|---:|
| Plain RF | 0.6937 | 0.6906 |
| A-only | 0.6970 | 0.6975 |
| M-only | 0.6919 | 0.6954 |
| S-only | 0.6976 | 0.6902 |
| MS-only | 0.6956 | 0.6968 |
| AMS-full | 0.6975 | 0.6962 |

S-only had the highest pooled ROC-AUC; MS-only had the highest fold-mean PR-AUC (0.7124). No configuration was consistently best. Five variants across two primary metrics were tested with paired Wilcoxon tests and Holm correction; no variant significantly improved over plain RF after correction.

## 10. Conclusion

The project successfully implemented and evaluated a reliability-aware training pipeline combining external reference agreement, cross-fitted model support, and molecular-neighbour consistency. The internal grouped results show small numerical gains on selected metrics, but statistical testing does not establish a reliable improvement. The independent DILIrank prediction comparison favours the plain Random Forest numerically.

**Conclusion:** the reliability-estimation framework is implemented and experimentally evaluated; its predictive benefit remains unproven.

## 11. Limitations and future work

- The final predictive experiment focuses on one endpoint (liver toxicity), not all eight UniTox toxicity categories.
- The external No/Most validation set is small (122 compounds) and DILIrank is not a gold-standard label source.
- Ten outer folds provide limited inferential power, and their training sets overlap.
- The 27 provisional identity bridges require expert chemical review.
- Future work should expand to all toxicity endpoints, use larger independent reference datasets, improve label adjudication, investigate calibrated uncertainty/adaptive weighting, and evaluate molecular graph or molecular-language models under the same leakage-aware protocol.

## 12. Repository structure

```text
reliability-aware-unitox/
├── Reliability_Aware_UniTox_Full_Project.ipynb
├── README.md
├── requirements.txt
├── data/       # place required input files here; do not commit restricted raw data
└── outputs/    # generated CSV results and audit artefacts
```

## 13. Input files required

Place the following files in `data/` (or in the supported extracted UniTox data directories searched by the notebook):

- `UniTox-Small-Molecule-Benchmark.csv`
- `UniTox-Small-Molecule-Benchmark-morgan.npz`
- `UniTox-Small-Molecule-Benchmark-rdkit.npz`
- `dilirank_raw.xlsx` (DILIrank workbook; expected sheet: `supporting table 1`)

The notebook does not download the datasets. Obtain them from their original sources and check applicable data-use terms before publishing or redistributing them.

## 14. Setup and execution

Recommended: Python 3.10 or 3.11 in a clean virtual environment.

```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter notebook
```

Open `Reliability_Aware_UniTox_Full_Project.ipynb` and run **Kernel → Restart & Run All** from top to bottom. The nested RF, external bootstrap, and ablation sections are computationally intensive. The optional preliminary baseline is disabled by default. The frozen outer-fold seed 397 is used by default.

## 15. References

1. Silberg et al. (2024). *UniTox: Leveraging LLMs to Curate a Unified Dataset of Drug-Induced Toxicity from FDA Labels*. NeurIPS 2024.
2. Breiman, L. (2001). Random Forests. *Machine Learning*, 45, 5–32.
3. Rogers, D. and Hahn, M. (2010). Extended-Connectivity Fingerprints. *Journal of Chemical Information and Modeling*, 50(5), 742–754.
4. RDKit documentation: https://www.rdkit.org/docs/
5. DILIrank: FDA drug-induced liver injury reference dataset.
