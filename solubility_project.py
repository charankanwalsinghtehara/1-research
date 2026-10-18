"""End-to-end Delaney ESOL data, evaluation, and model-export workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Lipinski, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


DATA_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"
TARGET_COLUMN = "measured log solubility in mols per litre"
SMILES_COLUMN = "smiles"
DESCRIPTOR_FUNCTIONS = {
    "MolWt": Descriptors.MolWt,
    "HeavyAtomMolWt": Descriptors.HeavyAtomMolWt,
    "ExactMolWt": Descriptors.ExactMolWt,
    "NumValenceElectrons": Descriptors.NumValenceElectrons,
    "TPSA": rdMolDescriptors.CalcTPSA,
    "MolLogP": Descriptors.MolLogP,
    "NumHDonors": Lipinski.NumHDonors,
    "NumHAcceptors": Lipinski.NumHAcceptors,
    "NumRotatableBonds": Lipinski.NumRotatableBonds,
    "NumHeteroatoms": Lipinski.NumHeteroatoms,
    "NumRings": Lipinski.RingCount,
    "NumAromaticRings": Lipinski.NumAromaticRings,
    "NumAliphaticRings": Lipinski.NumAliphaticRings,
    "FractionCSP3": rdMolDescriptors.CalcFractionCSP3,
    "LabuteASA": rdMolDescriptors.CalcLabuteASA,
    "Chi0v": Descriptors.Chi0v,
    "Chi1v": Descriptors.Chi1v,
    "Kappa1": Descriptors.Kappa1,
    "Kappa2": Descriptors.Kappa2,
    "Kappa3": Descriptors.Kappa3,
}
MODEL_NAMES = ("random_forest", "extra_trees", "hist_gradient_boosting")
FEATURE_NAMES = ("descriptors", "morgan")
SPLIT_NAMES = ("random", "scaffold")
LOG = logging.getLogger("esol_project")


def download_dataset(data_dir: str | Path = "data") -> Path:
    """Download the Delaney ESOL CSV if it is not already present."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    destination = data_dir / "delaney-processed.csv"
    if destination.exists() and destination.stat().st_size > 1_000:
        return destination
    request = urllib.request.Request(
        DATA_URL, headers={"User-Agent": "Molecular-Solubility-Research/1.0"}
    )
    temporary = destination.with_suffix(".csv.part")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            temporary.write_bytes(response.read())
        if temporary.stat().st_size <= 1_000:
            raise ValueError("Downloaded ESOL file is unexpectedly small.")
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def load_and_clean_data(
    csv_path: str | Path, *, min_rows: int = 30
) -> Tuple[pd.DataFrame, dict]:
    """Read ESOL, remove invalid records, canonicalize, and deduplicate."""
    raw = pd.read_csv(csv_path)
    if SMILES_COLUMN not in raw or TARGET_COLUMN not in raw:
        raise ValueError(
            f"Expected source columns {SMILES_COLUMN!r} and {TARGET_COLUMN!r}; "
            f"found {list(raw.columns)}"
        )
    rows = []
    invalid_smiles = 0
    invalid_target = 0
    for source_row, (smiles, target) in enumerate(
        zip(raw[SMILES_COLUMN], raw[TARGET_COLUMN])
    ):
        mol = Chem.MolFromSmiles(str(smiles)) if pd.notna(smiles) else None
        numeric_target = pd.to_numeric(pd.Series([target]), errors="coerce").iloc[0]
        if mol is None:
            invalid_smiles += 1
            continue
        if not np.isfinite(numeric_target):
            invalid_target += 1
            continue
        rows.append(
            {
                "source_row": source_row,
                "smiles": str(smiles),
                "canonical_smiles": Chem.MolToSmiles(
                    mol, canonical=True, isomericSmiles=True
                ),
                TARGET_COLUMN: float(numeric_target),
            }
        )
    frame = pd.DataFrame(rows)
    before = len(frame)
    frame = frame.drop_duplicates("canonical_smiles", keep="first").reset_index(drop=True)
    report = {
        "input_rows": int(len(raw)),
        "invalid_smiles": invalid_smiles,
        "invalid_target_values": invalid_target,
        "duplicate_structures_removed": before - len(frame),
        "valid_rows": int(len(frame)),
        "source_sha256": _sha256(Path(csv_path)),
        "source_url": DATA_URL if Path(csv_path).name == "delaney-processed.csv" else str(csv_path),
        "target_unit": "log(mol/L)",
    }
    if len(frame) < min_rows:
        raise ValueError(
            f"Only {len(frame)} valid molecules; expected at least {min_rows}."
        )
    return frame, report


def calculate_features(
    frame: pd.DataFrame, fingerprint_bits: int = 2048
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Calculate fixed RDKit descriptors and radius-2 Morgan fingerprints."""
    descriptor_rows = []
    for smiles in frame["canonical_smiles"]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid canonical SMILES: {smiles}")
        descriptor_rows.append(
            {name: function(mol) for name, function in DESCRIPTOR_FUNCTIONS.items()}
        )
    descriptors = pd.DataFrame(descriptor_rows, columns=DESCRIPTOR_FUNCTIONS)
    descriptor_values = descriptors.to_numpy(dtype=np.float32)
    if not np.isfinite(descriptor_values).all():
        raise ValueError("Descriptor matrix contains non-finite values.")

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=fingerprint_bits)
    fingerprints = np.zeros((len(frame), fingerprint_bits), dtype=np.uint8)
    for index, smiles in enumerate(frame["canonical_smiles"]):
        mol = Chem.MolFromSmiles(smiles)
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(mol), fingerprints[index])
    return descriptors, fingerprints


def _scaffold_group(mol: Chem.Mol, canonical_smiles: str) -> str:
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    scaffold_smiles = Chem.MolToSmiles(scaffold, canonical=True)
    return f"SCAFFOLD:{scaffold_smiles}" if scaffold_smiles else f"ACYCLIC:{canonical_smiles}"


def create_splits(
    frame: pd.DataFrame, seed: int = 42
) -> Dict[str, Dict[str, np.ndarray]]:
    """Create deterministic random and scaffold-aware 80/10/10 partitions."""
    indices = np.arange(len(frame), dtype=np.int64)
    train_validation, test = train_test_split(
        indices, test_size=0.1, random_state=seed, shuffle=True
    )
    train, validation = train_test_split(
        train_validation, test_size=1 / 9, random_state=seed, shuffle=True
    )
    random_split = {
        "train": np.sort(train),
        "validation": np.sort(validation),
        "test": np.sort(test),
    }

    groups: Dict[str, List[int]] = {}
    for index, smiles in enumerate(frame["canonical_smiles"]):
        mol = Chem.MolFromSmiles(smiles)
        groups.setdefault(_scaffold_group(mol, smiles), []).append(index)
    rng = np.random.default_rng(seed)
    items = list(groups.items())
    tie_values = rng.random(len(items))
    items = [
        item
        for _, item in sorted(
            zip(tie_values, items),
            key=lambda pair: (len(pair[1][1]), pair[0]),
            reverse=True,
        )
    ]
    targets = np.array([0.8, 0.1, 0.1]) * len(frame)
    buckets: List[List[int]] = [[], [], []]
    for _, group_indices in items:
        deficits = [
            (targets[i] - len(buckets[i])) / max(targets[i], 1.0)
            for i in range(3)
        ]
        buckets[int(np.argmax(deficits))].extend(group_indices)
    scaffold_split = {
        name: np.asarray(sorted(bucket), dtype=np.int64)
        for name, bucket in zip(("train", "validation", "test"), buckets)
    }
    result = {"random": random_split, "scaffold": scaffold_split}
    for split_name, split in result.items():
        merged = np.concatenate(list(split.values()))
        if len(merged) != len(frame) or len(np.unique(merged)) != len(frame):
            raise AssertionError(f"{split_name} split does not partition rows exactly.")
        if any(len(split[part]) == 0 for part in split):
            raise ValueError(f"{split_name} split produced an empty subset.")
    return result


def _new_model(name: str, seed: int, n_jobs: int):
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, max_features=0.7,
            random_state=seed, n_jobs=n_jobs,
        )
    if name == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=300, min_samples_leaf=2, max_features=0.7,
            random_state=seed, n_jobs=n_jobs,
        )
    if name == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.08, max_leaf_nodes=15, random_state=seed
        )
    if name == "mean_baseline":
        return DummyRegressor(strategy="mean")
    raise ValueError(f"Unknown model: {name}")


def _metric_values(actual: np.ndarray, predicted: np.ndarray) -> dict:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_experiments(
    data: pd.DataFrame,
    descriptors: pd.DataFrame,
    fingerprints: np.ndarray,
    splits: Dict[str, Dict[str, np.ndarray]],
    output_dir: str | Path,
    *,
    seed: int = 42,
    n_jobs: int = -1,
) -> dict:
    """Select by validation MAE, report one test score per split, export winner."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = data[TARGET_COLUMN].to_numpy(dtype=np.float64)
    descriptor_matrix = descriptors.to_numpy(dtype=np.float32)
    feature_matrices = {
        "descriptors": descriptor_matrix,
        "morgan": fingerprints,
    }
    validation_rows = []
    test_rows = []
    test_predictions = []
    selections = {}

    for split_name in SPLIT_NAMES:
        split = splits[split_name]
        candidates = []
        for model_name in MODEL_NAMES:
            for feature_name in FEATURE_NAMES:
                matrix = feature_matrices[feature_name]
                model = _new_model(model_name, seed, n_jobs)
                model.fit(matrix[split["train"]], target[split["train"]])
                validation_prediction = model.predict(matrix[split["validation"]])
                row = {
                    "split": split_name,
                    "model": model_name,
                    "features": feature_name,
                    **_metric_values(target[split["validation"]], validation_prediction),
                    "n_validation": int(len(split["validation"])),
                }
                candidates.append(row)
                validation_rows.append(row)

        baseline = _new_model("mean_baseline", seed, n_jobs)
        baseline.fit(np.zeros((len(split["train"]), 1)), target[split["train"]])
        baseline_row = {
            "split": split_name,
            "model": "mean_baseline",
            "features": "none",
            **_metric_values(
                target[split["validation"]],
                baseline.predict(np.zeros((len(split["validation"]), 1))),
            ),
            "n_validation": int(len(split["validation"])),
        }
        validation_rows.append(baseline_row)
        candidates.append(baseline_row)
        winner = min(candidates, key=lambda item: (item["mae"], item["model"], item["features"]))
        model_name, feature_name = winner["model"], winner["features"]
        fit_indices = np.concatenate((split["train"], split["validation"]))

        if model_name == "mean_baseline":
            final_model = _new_model(model_name, seed, n_jobs)
            final_model.fit(np.zeros((len(fit_indices), 1)), target[fit_indices])
            test_prediction = final_model.predict(np.zeros((len(split["test"]), 1)))
        else:
            final_model = _new_model(model_name, seed, n_jobs)
            final_model.fit(feature_matrices[feature_name][fit_indices], target[fit_indices])
            test_prediction = final_model.predict(feature_matrices[feature_name][split["test"]])

        selections[split_name] = {
            "model": model_name,
            "features": feature_name,
            "validation_mae": winner["mae"],
        }
        test_rows.append(
            {
                "split": split_name,
                "model": model_name,
                "features": feature_name,
                **_metric_values(target[split["test"]], test_prediction),
                "n_test": int(len(split["test"])),
            }
        )
        for index, predicted in zip(split["test"], test_prediction):
            test_predictions.append(
                {
                    "split": split_name,
                    "row_index": int(index),
                    "canonical_smiles": data.iloc[index]["canonical_smiles"],
                    "actual": float(target[index]),
                    "predicted": float(predicted),
                }
            )

    validation_frame = pd.DataFrame(validation_rows).sort_values(["split", "mae"])
    test_frame = pd.DataFrame(test_rows)
    predictions_frame = pd.DataFrame(test_predictions)
    validation_frame.to_csv(output_dir / "validation_results.csv", index=False)
    test_frame.to_csv(output_dir / "test_results.csv", index=False)
    predictions_frame.to_csv(output_dir / "test_predictions.csv", index=False)

    _save_plots(validation_frame, test_frame, predictions_frame, output_dir)

    # Choose the deployable configuration using scaffold-validation MAE only.
    deployment_selection = selections["scaffold"]
    deployment_model = _new_model(deployment_selection["model"], seed, n_jobs)
    if deployment_selection["model"] == "mean_baseline":
        deployment_model.fit(np.zeros((len(target), 1)), target)
    else:
        deployment_model.fit(
            feature_matrices[deployment_selection["features"]], target
        )
    deployment_bundle = {
        "model": deployment_model,
        "model_name": deployment_selection["model"],
        "feature_set": deployment_selection["features"],
        "descriptor_names": list(descriptors.columns),
        "fingerprint_bits": int(fingerprints.shape[1]),
        "fingerprint_radius": 2,
        "target": TARGET_COLUMN,
        "target_unit": "log(mol/L)",
        "selection_split": "scaffold",
        "validation_mae": deployment_selection["validation_mae"],
        "training_rows": int(len(data)),
    }
    joblib.dump(deployment_bundle, output_dir / "solubility_model.joblib")
    summary = {
        "dataset": "Delaney ESOL",
        "target": TARGET_COLUMN,
        "target_unit": "log(mol/L)",
        "rows": int(len(data)),
        "seed": seed,
        "feature_sets": list(FEATURE_NAMES),
        "models": list(MODEL_NAMES),
        "selection_metric": "validation MAE",
        "test_policy": "Only the validation-selected candidate is evaluated per split.",
        "scaffold_policy": (
            "Bemis-Murcko scaffolds stay in one split; acyclic structures with empty "
            "scaffolds receive unique canonical-SMILES groups."
        ),
        "selections": selections,
        "deployment_selection": deployment_bundle | {"model": deployment_selection["model"]},
        "test_results": test_rows,
    }
    with (output_dir / "experiment_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return summary


def _save_plots(
    validation: pd.DataFrame,
    test: pd.DataFrame,
    predictions: pd.DataFrame,
    output_dir: Path,
) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for axis, split_name in zip(axes, SPLIT_NAMES):
        sns.barplot(
            data=validation.loc[validation["split"] == split_name],
            x="model",
            y="mae",
            hue="features",
            errorbar=None,
            ax=axis,
        )
        axis.set_title(f"{split_name.title()} validation")
        axis.set_ylabel("MAE (log mol/L)")
        axis.set_xlabel("")
        axis.tick_params(axis="x", rotation=15)
    figure.suptitle("Validation performance by model and representation")
    figure.tight_layout()
    figure.savefig(output_dir / "validation_mae.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    for axis, split_name in zip(axes, SPLIT_NAMES):
        subset = predictions.loc[predictions["split"] == split_name]
        sns.scatterplot(data=subset, x="actual", y="predicted", s=32, alpha=0.75, ax=axis)
        low = float(min(subset["actual"].min(), subset["predicted"].min()))
        high = float(max(subset["actual"].max(), subset["predicted"].max()))
        axis.plot([low, high], [low, high], linestyle="--", color="#c55a3d")
        metric = test.loc[test["split"] == split_name].iloc[0]
        axis.set_title(f"{split_name.title()} holdout | MAE {metric['mae']:.2f}")
        axis.set_xlabel("Measured log solubility (mol/L)")
        axis.set_ylabel("Predicted log solubility (mol/L)")
    figure.tight_layout()
    figure.savefig(output_dir / "test_parity.png", dpi=180)
    plt.close(figure)


def run_pipeline(
    *,
    csv_path: str | Path | None = None,
    data_dir: str | Path = "data",
    output_dir: str | Path = "outputs/solubility",
    seed: int = 42,
    fingerprint_bits: int = 2048,
    n_jobs: int = -1,
) -> dict:
    csv_path = Path(csv_path) if csv_path else download_dataset(data_dir)
    frame, cleaning_report = load_and_clean_data(csv_path)
    descriptors, fingerprints = calculate_features(frame, fingerprint_bits)
    splits = create_splits(frame, seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "esol_processed.csv", index=False)
    descriptors.to_csv(output_dir / "descriptors.csv", index=False)
    for split_name, split in splits.items():
        for subset, indices in split.items():
            pd.DataFrame({"row_index": indices}).to_csv(
                output_dir / f"{split_name}_{subset}_indices.csv", index=False
            )
    summary = run_experiments(
        frame, descriptors, fingerprints, splits, output_dir, seed=seed, n_jobs=n_jobs
    )
    summary["cleaning_report"] = cleaning_report
    summary["source_sha256"] = cleaning_report["source_sha256"]
    with (output_dir / "experiment_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    return summary


def run_self_tests() -> None:
    from tempfile import TemporaryDirectory

    smiles = [
        "CCO", "CCN", "CCCO", "CCCN", "c1ccccc1", "Cc1ccccc1", "CC(=O)O",
        "CC(=O)N", "CCS", "CCCl", "CCBr", "CCOC", "CC(C)O", "CC(C)N",
        "C1CCCCC1", "c1ccncc1", "CC(=O)C", "CC#N", "COC", "CCF",
    ] * 3
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        csv_path = root / "fixture.csv"
        pd.DataFrame(
            {SMILES_COLUMN: smiles, TARGET_COLUMN: np.linspace(-5.0, 0.0, len(smiles))}
        ).to_csv(csv_path, index=False)
        frame, _ = load_and_clean_data(csv_path, min_rows=15)
        descriptors, fingerprints = calculate_features(frame, fingerprint_bits=128)
        splits = create_splits(frame, seed=5)
        assert descriptors.shape == (len(frame), len(DESCRIPTOR_FUNCTIONS))
        assert fingerprints.shape == (len(frame), 128)
        summary = run_experiments(
            frame,
            descriptors,
            fingerprints,
            splits,
            root / "output",
            seed=5,
            n_jobs=1,
        )
        assert len(summary["test_results"]) == 2
        assert (root / "output" / "solubility_model.joblib").exists()
        assert (root / "output" / "validation_mae.png").exists()
        bundle = joblib.load(root / "output" / "solubility_model.joblib")
        assert bundle["target_unit"] == "log(mol/L)"
    LOG.info("All ESOL offline self-tests passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and evaluate the ESOL solubility project.")
    parser.add_argument("--csv", default=None, help="Use a local ESOL-compatible CSV.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs/solubility")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fingerprint-bits", type=int, default=2048)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--test", action="store_true", help="Run offline self-tests.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.test:
        run_self_tests()
        return
    run_pipeline(
        csv_path=args.csv,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        fingerprint_bits=args.fingerprint_bits,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()