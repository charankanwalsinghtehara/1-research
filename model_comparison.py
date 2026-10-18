

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data_pipeline import DESCRIPTOR_FUNCTIONS, TARGET_COLUMN, _write_json


LOG = logging.getLogger("qm9_model_comparison")
FEATURE_SETS = ("descriptors", "morgan", "combined")
SPLITS = ("random", "scaffold")


def build_feature_matrices(
    data: pd.DataFrame, fingerprint_bits: int = 2048
) -> Tuple[np.ndarray, np.ndarray]:
    """Return descriptor and Morgan fingerprint matrices in data row order."""
    missing = set(DESCRIPTOR_FUNCTIONS).difference(data.columns)
    if missing:
        raise ValueError(f"Processed data is missing descriptors: {sorted(missing)}")
    if "canonical_smiles" not in data.columns:
        raise ValueError("Processed data must contain canonical_smiles.")
    if fingerprint_bits < 128:
        raise ValueError("fingerprint_bits must be at least 128.")

    descriptors = data[list(DESCRIPTOR_FUNCTIONS)].to_numpy(dtype=np.float32)
    if not np.isfinite(descriptors).all():
        raise ValueError("Descriptor matrix contains non-finite values.")

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=fingerprint_bits
    )
    fingerprints = np.zeros((len(data), fingerprint_bits), dtype=np.uint8)
    for row_index, smiles in enumerate(data["canonical_smiles"]):
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            raise ValueError(f"Invalid canonical SMILES at row {row_index}: {smiles}")
        fingerprint = generator.GetFingerprint(mol)
        DataStructs.ConvertToNumpyArray(fingerprint, fingerprints[row_index])
    return descriptors, fingerprints


def _feature_matrix(
    feature_set: str, descriptors: np.ndarray, fingerprints: np.ndarray
) -> np.ndarray:
    if feature_set == "descriptors":
        return descriptors
    if feature_set == "morgan":
        return fingerprints
    if feature_set == "combined":
        return np.concatenate((descriptors, fingerprints), axis=1)
    raise ValueError(f"Unknown feature set: {feature_set}")


def _load_split(split_dir: Path, split_name: str, row_count: int) -> Dict[str, np.ndarray]:
    split = {}
    for subset in ("train", "validation", "test"):
        path = split_dir / f"{split_name}_{subset}_indices.csv"
        if not path.exists():
            raise FileNotFoundError(f"Split index file not found: {path}")
        indices = pd.read_csv(path)["row_index"].to_numpy(dtype=np.int64)
        if np.any(indices < 0) or np.any(indices >= row_count):
            raise ValueError(f"{path} contains out-of-range row indices.")
        split[subset] = indices

    joined = np.concatenate(list(split.values()))
    if len(joined) != row_count or len(np.unique(joined)) != row_count:
        raise ValueError(f"{split_name} indices must partition all rows exactly.")
    return split


def _new_model(
    model_name: str,
    seed: int,
    rf_trees: int,
    boosting_iterations: int,
    n_jobs: int,
):
    if model_name == "random_forest":
        return RandomForestRegressor(
            n_estimators=rf_trees,
            min_samples_leaf=2,
            max_features=0.5,
            random_state=seed,
            n_jobs=n_jobs,
        )
    if model_name == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            max_iter=boosting_iterations,
            learning_rate=0.1,
            max_leaf_nodes=31,
            random_state=seed,
        )
    if model_name == "mean_baseline":
        return DummyRegressor(strategy="mean")
    raise ValueError(f"Unknown model: {model_name}")


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    return {
        "mae_eV": float(mean_absolute_error(actual, predicted)),
        "rmse_eV": float(np.sqrt(mean_squared_error(actual, predicted))),
    }


def run_comparison(
    *,
    data_csv: str | Path = "outputs/module1/qm9_processed.csv",
    split_dir: str | Path = "outputs/module1/splits",
    output_dir: str | Path = "outputs/module2",
    seed: int = 42,
    fingerprint_bits: int = 2048,
    rf_trees: int = 200,
    boosting_iterations: int = 150,
    n_jobs: int = -1,
) -> Dict[str, object]:
    """Compare candidates, select on validation MAE, and test each winner."""
    data_csv = Path(data_csv)
    split_dir = Path(split_dir)
    output_dir = Path(output_dir)
    data = pd.read_csv(data_csv)
    if TARGET_COLUMN not in data:
        raise ValueError(f"Processed data must contain {TARGET_COLUMN!r}.")
    targets = data[TARGET_COLUMN].to_numpy(dtype=np.float64)
    if not np.isfinite(targets).all():
        raise ValueError("Target contains non-finite values.")
    if rf_trees < 1 or boosting_iterations < 1:
        raise ValueError("Model iteration counts must be positive.")

    descriptors, fingerprints = build_feature_matrices(data, fingerprint_bits)
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_rows: List[dict] = []
    test_rows: List[dict] = []
    selections = {}

    for split_name in SPLITS:
        split = _load_split(split_dir, split_name, len(data))
        candidates = []
        for model_name in ("random_forest", "hist_gradient_boosting"):
            for feature_set in FEATURE_SETS:
                matrix = _feature_matrix(feature_set, descriptors, fingerprints)
                model = _new_model(
                    model_name, seed, rf_trees, boosting_iterations, n_jobs
                )
                model.fit(matrix[split["train"]], targets[split["train"]])
                predicted = model.predict(matrix[split["validation"]])
                scores = _metrics(targets[split["validation"]], predicted)
                row = {
                    "split": split_name,
                    "model": model_name,
                    "feature_set": feature_set,
                    **scores,
                    "n_train": int(len(split["train"])),
                    "n_validation": int(len(split["validation"])),
                }
                validation_rows.append(row)
                candidates.append(row)

        baseline = _new_model("mean_baseline", seed, rf_trees, boosting_iterations, n_jobs)
        baseline.fit(targets[split["train"]].reshape(-1, 1), targets[split["train"]])
        baseline_prediction = baseline.predict(
            targets[split["validation"]].reshape(-1, 1)
        )
        baseline_row = {
            "split": split_name,
            "model": "mean_baseline",
            "feature_set": "none",
            **_metrics(targets[split["validation"]], baseline_prediction),
            "n_train": int(len(split["train"])),
            "n_validation": int(len(split["validation"])),
        }
        validation_rows.append(baseline_row)
        candidates.append(baseline_row)

        winner = min(
            candidates,
            key=lambda row: (row["mae_eV"], row["model"], row["feature_set"]),
        )
        model_name = winner["model"]
        feature_set = winner["feature_set"]
        fitting_indices = np.concatenate(
            (split["train"], split["validation"])
        )
        selections[split_name] = {
            "model": model_name,
            "feature_set": feature_set,
            "validation_mae_eV": winner["mae_eV"],
        }

        if model_name == "mean_baseline":
            test_model = _new_model(
                model_name, seed, rf_trees, boosting_iterations, n_jobs
            )
            test_model.fit(
                np.zeros((len(fitting_indices), 1)), targets[fitting_indices]
            )
            test_prediction = test_model.predict(
                np.zeros((len(split["test"]), 1))
            )
        else:
            matrix = _feature_matrix(feature_set, descriptors, fingerprints)
            test_model = _new_model(
                model_name, seed, rf_trees, boosting_iterations, n_jobs
            )
            test_model.fit(matrix[fitting_indices], targets[fitting_indices])
            test_prediction = test_model.predict(matrix[split["test"]])

        test_rows.append(
            {
                "split": split_name,
                "model": model_name,
                "feature_set": feature_set,
                **_metrics(targets[split["test"]], test_prediction),
                "n_train_validation": int(len(fitting_indices)),
                "n_test": int(len(split["test"])),
            }
        )

    validation_results = pd.DataFrame(validation_rows).sort_values(
        ["split", "mae_eV", "model", "feature_set"]
    )
    test_results = pd.DataFrame(test_rows)
    validation_results.to_csv(output_dir / "validation_results.csv", index=False)
    test_results.to_csv(output_dir / "selected_test_results.csv", index=False)
    report = {
        "target": TARGET_COLUMN,
        "selection_metric": "validation MAE (eV)",
        "test_policy": "Only the validation-selected configuration is evaluated per split.",
        "seed": seed,
        "fingerprint": {"type": "Morgan", "radius": 2, "bits": fingerprint_bits},
        "model_parameters": {
            "random_forest": {
                "n_estimators": rf_trees,
                "min_samples_leaf": 2,
                "max_features": 0.5,
            },
            "hist_gradient_boosting": {
                "max_iter": boosting_iterations,
                "learning_rate": 0.1,
                "max_leaf_nodes": 31,
            },
        },
        "selections": selections,
        "test_results": test_rows,
    }
    _write_json(output_dir / "experiment_summary.json", report)
    LOG.info("Wrote experiment results to %s", output_dir.resolve())
    return {
        "validation_results": validation_results,
        "test_results": test_results,
        "summary": report,
        "output_dir": output_dir,
    }


def run_self_tests() -> None:
    """Exercise feature construction and end-to-end evaluation offline."""
    from data_pipeline import DESCRIPTOR_FUNCTIONS

    smiles = [
        "C", "CC", "CCC", "CCO", "CCN", "CO", "CN", "CCCO",
        "CCCN", "CC(=O)O", "CC(=O)N", "c1ccccc1",
    ] * 2
    molecules = [Chem.MolFromSmiles(value) for value in smiles]
    data = pd.DataFrame(
        {
            "canonical_smiles": [Chem.MolToSmiles(mol) for mol in molecules],
            TARGET_COLUMN: np.linspace(2.0, 5.0, len(smiles)),
        }
    )
    for name, function in DESCRIPTOR_FUNCTIONS.items():
        data[name] = [function(mol) for mol in molecules]

    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        data_csv = root / "processed.csv"
        split_dir = root / "splits"
        split_dir.mkdir()
        data.to_csv(data_csv, index=False)
        for split_name in SPLITS:
            for subset, indices in {
                "train": np.arange(0, 16),
                "validation": np.arange(16, 20),
                "test": np.arange(20, 24),
            }.items():
                pd.DataFrame({"row_index": indices}).to_csv(
                    split_dir / f"{split_name}_{subset}_indices.csv", index=False
                )

        result = run_comparison(
            data_csv=data_csv,
            split_dir=split_dir,
            output_dir=root / "results",
            seed=3,
            fingerprint_bits=128,
            rf_trees=5,
            boosting_iterations=5,
            n_jobs=1,
        )
        assert len(result["validation_results"]) == 14
        assert len(result["test_results"]) == 2
        assert (root / "results" / "experiment_summary.json").exists()

    LOG.info("All model-comparison self-tests passed.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare QM9 feature sets and regressors on saved splits."
    )
    parser.add_argument("--data-csv", default="outputs/module1/qm9_processed.csv")
    parser.add_argument("--split-dir", default="outputs/module1/splits")
    parser.add_argument("--output-dir", default="outputs/module2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fingerprint-bits", type=int, default=2048)
    parser.add_argument("--rf-trees", type=int, default=200)
    parser.add_argument("--boosting-iterations", type=int, default=150)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--test", action="store_true", help="Run offline self-tests and exit.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s: %(message)s",
        force=True,
    )
    if args.test:
        run_self_tests()
        return
    run_comparison(
        data_csv=args.data_csv,
        split_dir=args.split_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        fingerprint_bits=args.fingerprint_bits,
        rf_trees=args.rf_trees,
        boosting_iterations=args.boosting_iterations,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()