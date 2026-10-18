# AI-Based Molecular Water Solubility Prediction

An end-to-end research project comparing molecular descriptors and radius-2 Morgan fingerprints for prediction of Delaney ESOL water solubility. The target is the dataset's **measured log solubility in mol/L**; lower MAE is better. The existing `data_pipeline.py` and `model_comparison.py` remain a separate QM9 HOMO-LUMO gap project.

## Setup

Python 3.10 or newer is recommended. RDKit installation can vary by operating system; install it with the package manager appropriate to your environment if `pip install -r requirements.txt` cannot resolve it.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the research workflow

```bash
python solubility_project.py --test
python solubility_project.py
```

The default run downloads the public Delaney ESOL dataset and writes processed data, fixed random/scaffold split indices, a validation leaderboard, held-out test scores, predictions, plots, and the deployable model under `outputs/solubility/`. To use a local CSV, pass `--csv path/to/file.csv`; it must contain `smiles` and `measured log solubility in mols per litre` columns.

The comparison includes Random Forest, Extra Trees, and Histogram Gradient Boosting, each evaluated with the 20 RDKit descriptors or Morgan fingerprints. A mean baseline is included. Validation MAE selects the candidate separately for random and scaffold splits; the test score is calculated only for that selected candidate. The app model is selected using scaffold-validation MAE, then refit on all available ESOL rows after the comparison. Report both random and scaffold results because random splits can overstate performance on novel chemical scaffolds. Empty Bemis-Murcko scaffolds receive a unique group per canonical SMILES.

Key outputs:

- `validation_results.csv`: validation metrics for all candidates and baselines.
- `test_results.csv` and `test_predictions.csv`: selected held-out scores and row-level predictions.
- `validation_mae.png` and `test_parity.png`: generated visualizations.
- `experiment_summary.json`: settings, selection policy, and test results.
- `solubility_model.joblib`: fitted bundle consumed by Streamlit.

## Streamlit app

```bash
streamlit run app.py
```

The app expects `outputs/solubility/solubility_model.joblib`, created by the research workflow. Choose a familiar substance from the visible buttons; no chemical code is needed. The optional advanced section accepts a SMILES structure code. The interactive charts compare validation scores and held-out predictions, with controls for random versus unseen-scaffold evaluation and MAE, RMSE, or R-squared.

## Notebook and research report

`solubility_research.ipynb` is the interactive experiment entry point. `research_report.md` provides the report structure and records where actual outputs belong; no performance numbers are fabricated. Run the workflow before completing the Results and Discussion sections.

## Streamlit Community Cloud deployment

1. Push this repository to GitHub with `app.py`, `solubility_project.py`, `requirements.txt`, and the generated `outputs/solubility/solubility_model.joblib` artifact.
2. In Streamlit Community Cloud, create an app from the repository, select `app.py` as the entry point, and deploy.
3. Keep the artifact under the repository's GitHub file-size limits. If it exceeds those limits, store it in an artifact service and update `MODEL_PATH`/the load step to retrieve it securely.
4. Re-run the data and evaluation workflow when changing the model; redeploy the artifact together with the code and record its version/hash.

Do not commit private data or credentials. The public ESOL dataset and this lightweight model are intended for research/education and should not be treated as clinical, safety, or experimental advice.