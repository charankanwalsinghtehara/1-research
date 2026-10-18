"""Streamlit interface for the exported ESOL model."""

from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator

from solubility_project import DESCRIPTOR_FUNCTIONS, TARGET_COLUMN


MODEL_PATH = Path("outputs/solubility/solubility_model.joblib")
EXAMPLE_MOLECULES = {
    "Alcohol (ethanol)": "CCO",
    "Nail-polish remover (acetone)": "CC(=O)C",
    "Pain reliever (aspirin)": "CC(=O)Oc1ccccc1C(=O)O",
    "Pain reliever (acetaminophen)": "CC(=O)Nc1ccc(O)cc1",
    "Coffee stimulant (caffeine)": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
    "Sweetener molecule (glucose)": "C([C@@H]1[C@H]([C@@H]([C@H](C(O1)O)O)O)O)O",
    "Fuel component (benzene)": "c1ccccc1",
}


@st.cache_resource
def load_bundle(path: str):
    return joblib.load(path)


def make_input_features(smiles: str, bundle: dict) -> tuple[np.ndarray, str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("RDKit could not parse this SMILES string.")
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    if bundle["feature_set"] == "none":
        return np.zeros((1, 1), dtype=np.float32), canonical
    if bundle["feature_set"] == "descriptors":
        values = [function(mol) for function in DESCRIPTOR_FUNCTIONS.values()]
        return np.asarray(values, dtype=np.float32).reshape(1, -1), canonical
    if bundle["feature_set"] == "morgan":
        generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=bundle["fingerprint_radius"],
            fpSize=bundle["fingerprint_bits"],
        )
        fingerprint = generator.GetFingerprint(mol)
        array = np.zeros((bundle["fingerprint_bits"],), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fingerprint, array)
        return array.reshape(1, -1), canonical
    raise ValueError(f"Unsupported saved feature set: {bundle['feature_set']}")


@st.cache_data
def load_study_results(output_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(output_dir)
    validation = pd.read_csv(root / "validation_results.csv")
    test = pd.read_csv(root / "test_predictions.csv")
    return validation, test


def get_familiar_predictions(bundle: dict, selected_name: str) -> pd.DataFrame:
    rows = []
    for name, structure in EXAMPLE_MOLECULES.items():
        features, _ = make_input_features(structure, bundle)
        log_concentration = float(bundle["model"].predict(features)[0])
        mol = Chem.MolFromSmiles(structure)
        amount = (10 ** log_concentration) * Descriptors.MolWt(mol)
        rows.append(
            {
                "Substance": name,
                "Estimated grams per litre": amount,
                "Selected": "Chosen" if name == selected_name else "Other examples",
                "Model estimate (log mol/L)": log_concentration,
            }
        )
    return pd.DataFrame(rows)


def render_study_charts() -> None:
    output_dir = Path("outputs/solubility")
    required = (
        output_dir / "validation_results.csv",
        output_dir / "test_predictions.csv",
    )
    if not all(path.exists() for path in required):
        st.info("Run `python solubility_project.py` to generate charts for the research results.")
        return

    validation, test_predictions = load_study_results(str(output_dir))
    st.subheader("Model comparison")
    st.caption("Compare validation scores. Hover over bars for exact values; drag to zoom into model groups.")
    selected_split = st.radio(
        "Show results for",
        ["random", "scaffold"],
        format_func=lambda value: "Random molecule split" if value == "random" else "Unseen scaffold split",
        horizontal=True,
        key="chart_split",
    )
    selected_metric = st.radio(
        "Compare models by",
        ["mae", "rmse", "r2"],
        format_func=lambda value: {
            "mae": "Typical error (MAE, lower is better)",
            "rmse": "Large-error-sensitive score (RMSE, lower is better)",
            "r2": "Variation explained (R-squared, higher is better)",
        }[value],
        horizontal=True,
        key="chart_metric",
    )
    chart_data = validation.loc[validation["split"] == selected_split].copy()
    chart_data["model"] = chart_data["model"].str.replace("_", " ").str.title()
    chart_data["features"] = chart_data["features"].replace(
        {"descriptors": "Calculated properties", "morgan": "Atom-pattern fingerprints", "none": "Average baseline"}
    )
    chart_selection = alt.selection_interval(encodings=["x"])
    comparison_chart = (
        alt.Chart(chart_data)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("model:N", title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y(f"{selected_metric}:Q", title=selected_metric.upper(), scale=alt.Scale(zero=False)),
            color=alt.Color("features:N", title="Molecule information"),
            opacity=alt.condition(chart_selection, alt.value(1), alt.value(0.25)),
            xOffset=alt.XOffset("features:N"),
            tooltip=[
                alt.Tooltip("model:N", title="Model"),
                alt.Tooltip("features:N", title="Input information"),
                alt.Tooltip(f"{selected_metric}:Q", title=selected_metric.upper(), format=".3f"),
            ],
        )
        .add_params(chart_selection)
        .properties(height=330)
    )
    st.altair_chart(comparison_chart, use_container_width=True, theme="streamlit")

    st.subheader("Predictions on held-out molecules")
    st.caption("Each point is a test molecule. The diagonal indicates a perfect prediction; drag to zoom and hover for details.")
    plot_data = test_predictions.loc[test_predictions["split"] == selected_split].copy()
    parity_selection = alt.selection_interval(bind="scales")
    plot_min = float(min(plot_data["actual"].min(), plot_data["predicted"].min()))
    plot_max = float(max(plot_data["actual"].max(), plot_data["predicted"].max()))
    perfect_prediction = pd.DataFrame(
        {"actual": [plot_min, plot_max], "predicted": [plot_min, plot_max]}
    )
    parity_chart = (
        alt.layer(
            alt.Chart(perfect_prediction).mark_line(
                color="#c34e36", strokeDash=[5, 4], strokeWidth=2
            ),
            alt.Chart(plot_data)
            .mark_circle(size=58, opacity=0.65)
            .encode(
                color=alt.Color("actual:Q", title="Measured value", scale=alt.Scale(scheme="tealblues")),
                tooltip=[
                    alt.Tooltip("canonical_smiles:N", title="Molecule structure"),
                    alt.Tooltip("actual:Q", title="Measured", format=".2f"),
                    alt.Tooltip("predicted:Q", title="Estimated", format=".2f"),
                    alt.Tooltip("row_index:Q", title="Data row"),
                ],
            ),
        )
        .encode(
            x=alt.X("actual:Q", title="Measured solubility (log mol/L)"),
            y=alt.Y("predicted:Q", title="Estimated solubility (log mol/L)"),
        )
        .add_params(parity_selection)
        .properties(height=390)
    )
    st.altair_chart(parity_chart, use_container_width=True, theme="streamlit")


def main() -> None:
    st.set_page_config(page_title="Water Solubility Estimate", page_icon="⚗️", layout="centered")
    st.title("How much dissolves in water?")
    st.markdown(
        "Choose a familiar substance to get an estimate of how much could dissolve in water. "
        "This is a computer estimate, not a laboratory measurement."
    )
    if not MODEL_PATH.exists():
        st.error("Trained model not found. Run `python solubility_project.py` first.")
        st.stop()
    bundle = load_bundle(str(MODEL_PATH))

    st.subheader("Pick a familiar substance")
    chosen_molecule = st.session_state.setdefault(
        "chosen_molecule", next(iter(EXAMPLE_MOLECULES))
    )
    molecule_names = list(EXAMPLE_MOLECULES)
    for start in range(0, len(molecule_names), 2):
        columns = st.columns(2)
        for column, name in zip(columns, molecule_names[start : start + 2]):
            if column.button(
                name,
                key=f"molecule_choice_{start}_{name}",
                type="primary" if name == chosen_molecule else "secondary",
                use_container_width=True,
            ):
                st.session_state["chosen_molecule"] = name

    with st.expander("I have a molecule code (advanced)"):
        st.caption("Most people can use the substance buttons above. This is only for entering a chemical structure code.")
        custom_smiles = st.text_input(
            "Molecule code",
            placeholder="Example: CCO",
            help="Also called SMILES; paste it here if you already have one.",
        )

    chosen_molecule = st.session_state["chosen_molecule"]
    using_custom = bool(custom_smiles.strip())
    molecule_choice = "Advanced structure code" if using_custom else chosen_molecule
    smiles = custom_smiles.strip() if using_custom else EXAMPLE_MOLECULES[chosen_molecule]
    ready = bool(smiles)

    if st.button("Estimate amount that dissolves", type="primary", icon="🧪", disabled=not ready):
        try:
            features, canonical = make_input_features(smiles, bundle)
            prediction = float(bundle["model"].predict(features)[0])
            molecule = Chem.MolFromSmiles(canonical)
            grams_per_liter = (10 ** prediction) * Descriptors.MolWt(molecule)
            st.metric(
                "Estimated amount that may dissolve in 1 litre of water",
                f"{grams_per_liter:,.3g} grams",
            )
            st.caption(
                "A rough estimate from a computer model. Real results can vary with temperature and conditions."
            )
            if not using_custom:
                familiar_data = get_familiar_predictions(bundle, molecule_choice)
                st.subheader("Compare with familiar substances")
                comparison = (
                    alt.Chart(familiar_data)
                    .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
                    .encode(
                        x=alt.X("Estimated grams per litre:Q", title="Estimated grams in one litre of water"),
                        y=alt.Y("Substance:N", sort="-x", title=None),
                        color=alt.Color(
                            "Selected:N",
                            scale=alt.Scale(domain=["Chosen", "Other examples"], range=["#c34e36", "#247b78"]),
                            legend=None,
                        ),
                        tooltip=[
                            alt.Tooltip("Substance:N"),
                            alt.Tooltip("Estimated grams per litre:Q", format=".3g"),
                            alt.Tooltip("Model estimate (log mol/L):Q", format=".2f"),
                        ],
                    )
                    .properties(height=260)
                )
                st.altair_chart(comparison, use_container_width=True, theme="streamlit")
            with st.expander("Technical details (optional)"):
                st.write(f"Predicted value: {prediction:.2f} log(mol/L)")
                st.write(f"Model: {bundle['model_name'].replace('_', ' ').title()}")
                st.write(f"Molecule representation: {bundle['feature_set']}")
                st.write(f"Structure code (SMILES): `{canonical}`")
        except (ValueError, KeyError) as error:
            st.error(str(error))

    with st.expander("What does this estimate mean?"):
        st.write(
            "The number is an estimate of grams of the selected substance that may dissolve "
            "in one litre of water. It comes from patterns in previously measured molecules, "
            "so it should not replace a real measurement."
        )
        st.write(
            "The model was trained on the public ESOL collection of small molecules. "
            f"It learned from {bundle['training_rows']:,} examples."
        )
        st.warning(
            "Some substances are unlike anything in the training examples, so estimates can be inaccurate. "
            "Do not use this result for health, safety, or handling decisions."
        )

    with st.expander("What do the technical words mean?"):
        st.markdown(
            """
            - **SMILES / structure code:** A compact text spelling of a molecule's structure. The familiar-substance menu creates this automatically.
            - **Molecule representation:** The numbers the model uses to describe a molecule. Descriptors are calculated properties; fingerprints summarize nearby atoms and bonds.
            - **Model:** A computer program that learns patterns from known measurements to estimate a value for a new molecule.
            - **log(mol/L):** The model's internal concentration scale. The main result is converted to grams per litre to make it easier to read.
            - **ESOL:** The collection of measured small-molecule solubilities used to train this model.
            """
        )

    with st.expander("Explore the research charts", expanded=True):
        render_study_charts()


if __name__ == "__main__":
    main()