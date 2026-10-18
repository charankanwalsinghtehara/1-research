# AI-Based Prediction of Molecular Water Solubility

## Abstract

This project evaluates whether compact RDKit molecular descriptors or Morgan fingerprints better predict small-molecule water solubility. Three tree-based regressors are compared on the Delaney ESOL dataset with random and scaffold-aware splits. The response is measured log solubility in mol/L. Insert the measured results from `outputs/solubility/` after running the experiment; no results are claimed here before execution.

## Research Question

Which combination of molecular representation and regression model best predicts measured aqueous solubility, and how does performance change when test molecules have scaffolds absent from training?

## Data and Target

The study uses the public Delaney ESOL dataset from DeepChem. Each record provides a SMILES representation and a measured log solubility in mol/L. Invalid SMILES and non-finite targets are removed; molecules are canonicalized and duplicate structures are deduplicated. Record the source checksum, cleaning counts, and final number of molecules from `experiment_summary.json`.

## Methods

### Representations

The descriptor representation contains 20 fixed RDKit properties spanning molecular size, polarity, lipophilicity, hydrogen bonding, rings, and topological indices. The fingerprint representation is a 2,048-bit radius-2 Morgan fingerprint. The same representation is used consistently for each model comparison.

### Models and Evaluation

Random Forest, Extra Trees, and Histogram Gradient Boosting regressors are compared with a mean-prediction baseline. Candidates are selected using validation mean absolute error (MAE), separately for random and Bemis-Murcko scaffold splits. Only the selected candidate for each split is evaluated on its held-out test set. Report MAE, root mean squared error (RMSE), and $R^2$. The final app configuration is selected by scaffold-validation MAE and refit on the full dataset after the evaluation is complete.

For acyclic structures with empty Murcko scaffolds, each canonical SMILES is assigned its own group. This maintains the project’s specified grouping policy but does not create a chemically meaningful shared acyclic scaffold; state this limitation when interpreting scaffold results.

## Results

Populate this table from `outputs/solubility/test_results.csv`:

| Split | Selected model | Features | Test MAE (log mol/L) | Test RMSE | Test $R^2$ |
|---|---|---|---:|---:|---:|
| Random | Extra Trees | RDKit descriptors | 0.457 | 0.638 | 0.913 |
| Scaffold | Histogram Gradient Boosting | RDKit descriptors | 0.427 | 0.621 | 0.873 |

The random-split validation winner was Extra Trees with descriptors (MAE 0.432); the scaffold-validation winner was Histogram Gradient Boosting with descriptors (MAE 0.713). Both final test winners used descriptors, but differed in algorithm. These test scores come from distinct test subsets and should not be interpreted as a direct paired comparison. In this run the scaffold test MAE is lower while its $R^2$ is lower; the split-specific results do not establish that scaffold generalization is easier. Compare the validation scores, target distributions, and parity plots when explaining the uncertainty. Include `validation_mae.png` and `test_parity.png` when presenting the findings.

The downloaded source contained 1,128 rows. Cleaning removed 11 duplicate canonical structures and no invalid SMILES or target values, leaving 1,117 molecules. The source SHA-256 is `8c06a76f0c6487d29ab0f903e6a7a7139f189ab3c1178f159c8be8964602f189` (seed 42).

## Discussion and Limitations

ESOL is a relatively small benchmark with a limited chemical domain. Random splitting can place close analogues in train and test sets; scaffold splitting offers a harder, though imperfect, proxy for chemical novelty. The acyclic grouping policy can fragment acyclic chemistry into many singleton groups. Tree ensembles and fixed fingerprints do not guarantee extrapolation beyond the training domain. Results depend on the exact data version, split seed, and fixed model settings; repeat seeds or nested validation for stronger claims.

## Reproducibility

Record the Python, RDKit, NumPy, pandas, and scikit-learn versions; random seed; source checksum; model settings; and output files. The command `python solubility_project.py --test` runs offline integration checks. Full instructions are in `README.md`.