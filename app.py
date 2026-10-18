"""Streamlit interface for the exported ESOL model."""

from pathlib import Path

import joblib
import numpy as np
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

    molecule_choice = st.selectbox(
        "Choose a familiar substance",
        [*EXAMPLE_MOLECULES, "Enter a structure code (advanced)"],
    )
    if molecule_choice == "Enter a structure code (advanced)":
        st.caption("A structure code is a text format used to describe a molecule.")
        smiles = st.text_input(
            "Paste structure code",
            placeholder="Example: CCO",
            help="This format is called SMILES. You only need this option if you already have a code.",
        )
        ready = bool(smiles.strip())
    else:
        smiles = EXAMPLE_MOLECULES[molecule_choice]
        ready = True

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


if __name__ == "__main__":
    main()