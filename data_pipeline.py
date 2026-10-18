"""
data_pipeline.py
End-to-end QM9 molecular-property data pipeline for Google Colab.

Dataset source:
  https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv

Target:
  QM9 `gap` = LUMO - HOMO, reported by this source in Hartree.
  This module converts it to electronvolts (eV).

Important:
  - The pipeline does not train models.
  - Random and scaffold split indices are generated independently.
  - For acyclic molecules, the Bemis-Murcko scaffold is empty. To avoid
    putting every acyclic molecule into one artificial mega-group, each
    acyclic molecule is assigned a unique group using its canonical SMILES.
    Report this policy explicitly in any publication.
  - Test-set data must not be used to choose models or hyperparameters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split


DATA_URL = (
    "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv"
)
HARTREE_TO_EV = 27.211386245988
TARGET_SOURCE_COLUMN = "gap"
TARGET_COLUMN = "gap_eV"
SMILES_CANDIDATES = ("smiles", "SMILES", "canonical_smiles", "mol")
ID_CANDIDATES = ("mol_id", "id", "index")

# A compact, documented descriptor set that is fast enough for a first project.
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

LOG = logging.getLogger("qm9_pipeline")


def configure_logging(verbose: bool = True) -> None:
    """Configure readable console logging."""
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
        force=True,
    )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return SHA-256 checksum of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_dataset(
    data_dir: Path | str,
    url: str = DATA_URL,
    timeout_seconds: int = 60,
    force_download: bool = False,
) -> Path:
    """Download the QM9 CSV if absent; preserve an existing valid download."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    destination = data_dir / "qm9.csv"

    if destination.exists() and destination.stat().st_size > 1_000_000 and not force_download:
        LOG.info("Using existing dataset: %s", destination)
        return destination

    temporary = destination.with_suffix(".csv.part")
    LOG.info("Downloading QM9 CSV from %s", url)
    request = urllib.request.Request(
        url, headers={"User-Agent": "QM9-Research-Data-Pipeline/1.0"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            with temporary.open("wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
        if temporary.stat().st_size < 1_000_000:
            raise RuntimeError(
                "Downloaded file is unexpectedly small; the source may be unavailable."
            )
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    LOG.info(
        "Download complete: %.1f MB; SHA-256=%s",
        destination.stat().st_size / (1024 * 1024),
        sha256_file(destination),
    )
    return destination


def _find_column(columns: Iterable[str], candidates: Sequence[str]) -> Optional[str]:
    lookup = {str(column).strip().lower(): str(column) for column in columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def load_raw_dataset(csv_path: Path | str) -> pd.DataFrame:
    """Load CSV and validate the minimum required schema."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")

    frame = pd.read_csv(csv_path)
    if frame.empty:
        raise ValueError("Dataset CSV is empty.")

    smiles_col = _find_column(frame.columns, SMILES_CANDIDATES)
    target_col = _find_column(frame.columns, (TARGET_SOURCE_COLUMN,))
    if smiles_col is None:
        raise ValueError(
            f"No SMILES column found. Expected one of {SMILES_CANDIDATES}; "
            f"available columns: {list(frame.columns)}"
        )
    if target_col is None:
        raise ValueError(
            f"Required target column {TARGET_SOURCE_COLUMN!r} not found. "
            f"Available columns: {list(frame.columns)}"
        )

    id_col = _find_column(frame.columns, ID_CANDIDATES)
    result = pd.DataFrame()
    result["source_row"] = np.arange(len(frame), dtype=np.int64)
    result["molecule_id"] = (
        frame[id_col].astype(str)
        if id_col is not None
        else result["source_row"].astype(str)
    )
    result["smiles_raw"] = frame[smiles_col].astype("string")
    result["gap_hartree"] = pd.to_numeric(frame[target_col], errors="coerce")
    result["gap_eV"] = result["gap_hartree"] * HARTREE_TO_EV

    # Keep a copy of source-column names for provenance.
    result.attrs["source_columns"] = [str(c) for c in frame.columns]
    result.attrs["source_smiles_column"] = smiles_col
    result.attrs["source_target_column"] = target_col
    return result


def validate_and_canonicalize(
    raw: pd.DataFrame,
    *,
    drop_invalid: bool = True,
    drop_duplicate_molecules: bool = True,
    min_rows: int = 100,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Parse SMILES, canonicalize structures, and validate the target."""
    required = {"molecule_id", "smiles_raw", "gap_hartree", "gap_eV"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    frame = raw.copy()
    frame["mol"] = frame["smiles_raw"].map(
        lambda value: Chem.MolFromSmiles(str(value)) if pd.notna(value) else None
    )
    invalid_smiles = int(frame["mol"].isna().sum())
    invalid_target = int(
        (~np.isfinite(pd.to_numeric(frame["gap_eV"], errors="coerce"))).sum()
    )

    if not drop_invalid and (invalid_smiles or invalid_target):
        raise ValueError(
            f"Invalid records found: {invalid_smiles} invalid SMILES, "
            f"{invalid_target} invalid target values."
        )

    frame = frame.loc[frame["mol"].notna()].copy()
    frame = frame.loc[np.isfinite(frame["gap_eV"].astype(float))].copy()
    frame["canonical_smiles"] = frame["mol"].map(
        lambda mol: Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    )

    before_dedup = len(frame)
    if drop_duplicate_molecules:
        # QM9 is expected to be unique by molecule. If duplicated canonical
        # structures disagree in target, retain the first and report count;
        # later metadata includes duplicate count for transparency.
        frame = frame.drop_duplicates(subset="canonical_smiles", keep="first").copy()
    duplicate_molecules = before_dedup - len(frame)

    # Basic chemistry sanity checks: QM9 molecules contain C, N, O, F and H.
    allowed_atomic_numbers = {1, 6, 7, 8, 9}
    unsupported = []
    for idx, mol in frame["mol"].items():
        if any(atom.GetAtomicNum() not in allowed_atomic_numbers for atom in mol.GetAtoms()):
            unsupported.append(idx)
    if unsupported:
        frame = frame.drop(index=unsupported).copy()

    if len(frame) < min_rows:
        raise ValueError(
            f"Only {len(frame)} valid rows remain; minimum required is {min_rows}. "
            "Check the source file and schema."
        )
    if not np.isfinite(frame["gap_eV"].to_numpy(dtype=float)).all():
        raise ValueError("Target still contains non-finite values after cleaning.")
    if (frame["gap_eV"] <= 0).any():
        LOG.warning(
            "Some target values are <= 0 eV. Inspect the source and target mapping."
        )

    report = {
        "input_rows": int(len(raw)),
        "invalid_smiles": invalid_smiles,
        "invalid_target_values": invalid_target,
        "duplicate_canonical_molecules_removed": int(duplicate_molecules),
        "unsupported_element_rows_removed": int(len(unsupported)),
        "valid_rows": int(len(frame)),
    }
    frame = frame.reset_index(drop=True)
    return frame, report


def calculate_descriptors(frame: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Calculate fixed RDKit descriptors for each valid molecule."""
    if "mol" not in frame.columns:
        raise ValueError("Expected an RDKit 'mol' column; run validation first.")

    rows = []
    failed_indices = []
    for idx, mol in enumerate(frame["mol"]):
        try:
            values = {
                name: function(mol)
                for name, function in DESCRIPTOR_FUNCTIONS.items()
            }
            rows.append(values)
        except Exception as exc:
            failed_indices.append(idx)
            rows.append({name: np.nan for name in DESCRIPTOR_FUNCTIONS})

    descriptor_frame = pd.DataFrame(rows, index=frame.index)
    descriptor_frame = descriptor_frame.replace([np.inf, -np.inf], np.nan)

    # Do not silently impute failed descriptors. A row with any non-finite
    # descriptor is excluded and the caller is informed through the index list.
    bad_mask = ~np.isfinite(descriptor_frame.to_numpy(dtype=float)).all(axis=1)
    bad_rows = list(frame.index[bad_mask])
    if bad_rows:
        LOG.warning(
            "%d rows had missing/non-finite descriptors and should be excluded.",
            len(bad_rows),
        )
    return descriptor_frame, [str(i) for i in bad_rows]


def add_fingerprints(
    frame: pd.DataFrame,
    radius: int = 2,
    n_bits: int = 2048,
) -> np.ndarray:
    """Optional Morgan fingerprint matrix; included for later experiments."""
    if "mol" not in frame.columns:
        raise ValueError("Expected an RDKit 'mol' column; run validation first.")
    if radius < 1 or n_bits < 128:
        raise ValueError("Use radius >= 1 and n_bits >= 128.")
    fingerprints = []
    for mol in frame["mol"]:
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
            mol, radius, nBits=n_bits
        )
        array = np.zeros((n_bits,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, array)
        fingerprints.append(array)
    return np.vstack(fingerprints)


def scaffold_group(mol: Chem.Mol, canonical_smiles: str) -> str:
    """
    Return a Bemis-Murcko scaffold group.
    Acyclic molecules receive unique groups because their Murcko scaffold is empty.
    """
    scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
    scaffold_smiles = Chem.MolToSmiles(scaffold_mol, canonical=True)
    if scaffold_smiles:
        return "SCAFFOLD:" + scaffold_smiles
    return "ACYCLIC:" + canonical_smiles


def _allocate_scaffold_splits(
    frame: pd.DataFrame,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> Dict[str, np.ndarray]:
    """Greedy scaffold-group split with approximate requested proportions."""
    fractions = np.array(
        [train_fraction, validation_fraction, test_fraction], dtype=float
    )
    if not np.isclose(fractions.sum(), 1.0):
        raise ValueError("Train, validation, and test fractions must sum to 1.")

    groups: Dict[str, List[int]] = {}
    for idx, (mol, smiles) in enumerate(
        zip(frame["mol"], frame["canonical_smiles"])
    ):
        key = scaffold_group(mol, smiles)
        groups.setdefault(key, []).append(idx)

    rng = np.random.default_rng(seed)
    group_items = list(groups.items())
    # Random tie-breaks, then largest groups first.
    tie_breakers = rng.random(len(group_items))
    group_items = [
        item for _, item in sorted(
            zip(tie_breakers, group_items),
            key=lambda pair: (len(pair[1][1]), pair[0]),
            reverse=True,
        )
    ]

    total = len(frame)
    targets = fractions * total
    split_indices: List[List[int]] = [[], [], []]
    for _, indices in group_items:
        # Assign the entire group to the split with the largest relative deficit.
        deficits = [
            (targets[i] - len(split_indices[i])) / max(targets[i], 1.0)
            for i in range(3)
        ]
        chosen = int(np.argmax(deficits))
        split_indices[chosen].extend(indices)

    return {
        "train": np.array(sorted(split_indices[0]), dtype=np.int64),
        "validation": np.array(sorted(split_indices[1]), dtype=np.int64),
        "test": np.array(sorted(split_indices[2]), dtype=np.int64),
    }


def make_splits(
    frame: pd.DataFrame,
    *,
    seed: int = 42,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Create random and scaffold splits with deterministic seeds."""
    fractions = (train_fraction, validation_fraction, test_fraction)
    if any(f <= 0 for f in fractions) or not np.isclose(sum(fractions), 1.0):
        raise ValueError("Split fractions must be positive and sum to 1.")

    all_indices = np.arange(len(frame), dtype=np.int64)
    train_val, test_idx = train_test_split(
        all_indices,
        test_size=test_fraction,
        random_state=seed,
        shuffle=True,
    )
    adjusted_validation_fraction = validation_fraction / (
        train_fraction + validation_fraction
    )
    train_idx, val_idx = train_test_split(
        train_val,
        test_size=adjusted_validation_fraction,
        random_state=seed,
        shuffle=True,
    )
    random_split = {
        "train": np.sort(train_idx),
        "validation": np.sort(val_idx),
        "test": np.sort(test_idx),
    }
    scaffold_split = _allocate_scaffold_splits(
        frame,
        train_fraction,
        validation_fraction,
        test_fraction,
        seed,
    )

    for name, split in (("random", random_split), ("scaffold", scaffold_split)):
        merged = np.concatenate([split["train"], split["validation"], split["test"]])
        if len(merged) != len(frame) or len(np.unique(merged)) != len(frame):
            raise AssertionError(f"{name} split does not partition rows exactly.")
        if set(merged.tolist()) != set(all_indices.tolist()):
            raise AssertionError(f"{name} split omitted or added rows.")

    return {"random": random_split, "scaffold": scaffold_split}


def validate_splits(
    frame: pd.DataFrame,
    splits: Dict[str, Dict[str, np.ndarray]],
) -> Dict[str, dict]:
    """Validate partition coverage and report sizes and scaffold overlap."""
    report = {}
    scaffold_by_row = [
        scaffold_group(mol, smiles)
        for mol, smiles in zip(frame["mol"], frame["canonical_smiles"])
    ]
    for split_name, split in splits.items():
        sets = {name: set(indices.tolist()) for name, indices in split.items()}
        if sets["train"] & sets["validation"]:
            raise AssertionError(f"{split_name}: train/validation overlap.")
        if sets["train"] & sets["test"]:
            raise AssertionError(f"{split_name}: train/test overlap.")
        if sets["validation"] & sets["test"]:
            raise AssertionError(f"{split_name}: validation/test overlap.")
        if len(set.union(*sets.values())) != len(frame):
            raise AssertionError(f"{split_name}: split does not cover every row.")

        scaffold_sets = {
            name: {scaffold_by_row[i] for i in indices}
            for name, indices in split.items()
        }
        overlaps = {
            "train_validation": len(scaffold_sets["train"] & scaffold_sets["validation"]),
            "train_test": len(scaffold_sets["train"] & scaffold_sets["test"]),
            "validation_test": len(scaffold_sets["validation"] & scaffold_sets["test"]),
        }
        report[split_name] = {
            name: int(len(indices)) for name, indices in split.items()
        }
        report[split_name]["scaffold_overlap_counts"] = overlaps
    return report


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def build_pipeline(
    *,
    data_dir: str | Path = "data/raw",
    output_dir: str | Path = "outputs/module1",
    seed: int = 42,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    force_download: bool = False,
    source_csv: str | Path | None = None,
    min_rows: int = 100,
) -> Dict[str, object]:
    """Run the full data pipeline and save all artifacts."""
    start = time.time()
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = Path(source_csv) if source_csv else download_dataset(
        data_dir, force_download=force_download
    )
    raw = load_raw_dataset(csv_path)
    cleaned, cleaning_report = validate_and_canonicalize(raw, min_rows=min_rows)
    descriptors, bad_descriptor_rows = calculate_descriptors(cleaned)

    if bad_descriptor_rows:
        bad_indices = sorted({int(i) for i in bad_descriptor_rows})
        keep_mask = np.ones(len(cleaned), dtype=bool)
        keep_mask[bad_indices] = False
        cleaned = cleaned.loc[keep_mask].reset_index(drop=True)
        descriptors = descriptors.loc[keep_mask].reset_index(drop=True)

    if not np.isfinite(descriptors.to_numpy(dtype=float)).all():
        raise ValueError("Non-finite descriptors remain after filtering.")

    # Keep compact, portable records: do not serialize RDKit Mol objects.
    data = cleaned[
        ["source_row", "molecule_id", "smiles_raw", "canonical_smiles",
         "gap_hartree", "gap_eV"]
    ].copy()
    for column in descriptors.columns:
        data[column] = descriptors[column].to_numpy(dtype=float)

    splits = make_splits(
        cleaned,
        seed=seed,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )
    split_report = validate_splits(cleaned, splits)

    # Save combined processed table and split indices.
    processed_csv = output_dir / "qm9_processed.csv"
    data.to_csv(processed_csv, index=False)

    split_index_dir = output_dir / "splits"
    split_index_dir.mkdir(parents=True, exist_ok=True)
    for split_type, split in splits.items():
        for subset, indices in split.items():
            pd.DataFrame({"row_index": indices}).to_csv(
                split_index_dir / f"{split_type}_{subset}_indices.csv",
                index=False,
            )

    # Save separate ready-to-use split tables for convenience.
    for split_type, split in splits.items():
        for subset, indices in split.items():
            data.iloc[indices].to_csv(
                output_dir / f"{split_type}_{subset}.csv", index=False
            )

    metadata = {
        "dataset": "QM9",
        "source_url": DATA_URL if source_csv is None else str(csv_path),
        "source_file": str(csv_path),
        "source_sha256": sha256_file(csv_path),
        "source_columns": raw.attrs.get("source_columns", []),
        "source_smiles_column": raw.attrs.get("source_smiles_column"),
        "source_target_column": raw.attrs.get("source_target_column"),
        "target": {
            "source_column": TARGET_SOURCE_COLUMN,
            "output_column": TARGET_COLUMN,
            "source_unit": "Hartree",
            "output_unit": "eV",
            "conversion_factor": HARTREE_TO_EV,
            "definition": "HOMO-LUMO gap (LUMO - HOMO)",
        },
        "descriptor_names": list(DESCRIPTOR_FUNCTIONS.keys()),
        "n_descriptors": len(DESCRIPTOR_FUNCTIONS),
        "seed": seed,
        "fractions": {
            "train": train_fraction,
            "validation": validation_fraction,
            "test": test_fraction,
        },
        "split_report": split_report,
        "scaffold_policy": (
            "Bemis-Murcko scaffolds are kept in one split; acyclic molecules "
            "with empty Murcko scaffolds are assigned unique groups by canonical "
            "SMILES. Report this policy explicitly."
        ),
        "cleaning_report": cleaning_report,
        "descriptor_failure_rows_removed": len(bad_descriptor_rows),
        "n_final_rows": int(len(data)),
        "runtime_seconds": round(time.time() - start, 2),
        "python_version": sys.version,
        "platform": platform.platform(),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "rdkit_version": rdkit.__version__,
    }
    _write_json(output_dir / "metadata.json", metadata)

    LOG.info("Pipeline completed.")
    LOG.info("Processed rows: %d", len(data))
    LOG.info("Output directory: %s", output_dir.resolve())
    LOG.info("Split sizes: %s", json.dumps(split_report, indent=2))
    return {
        "data": data,
        "metadata": metadata,
        "splits": splits,
        "output_dir": output_dir,
        "processed_csv": processed_csv,
    }


def run_self_tests() -> None:
    """Fast offline tests for parsing, descriptors, splits, and validation."""
    from tempfile import TemporaryDirectory

    test_csv = pd.DataFrame(
        {
            "mol_id": ["m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "m10"],
            "smiles": [
                "C", "CC", "CCC", "CCO", "CCN",
                "c1ccccc1", "Cc1ccccc1", "O=C=O", "C#N", "CCF"
            ],
            "gap": [0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29],
        }
    )

    with TemporaryDirectory() as temporary:
        temporary = Path(temporary)
        csv_path = temporary / "tiny_qm9.csv"
        test_csv.to_csv(csv_path, index=False)

        raw = load_raw_dataset(csv_path)
        assert len(raw) == len(test_csv)
        assert np.isclose(
            raw.loc[0, "gap_eV"], test_csv.loc[0, "gap"] * HARTREE_TO_EV
        )

        clean, report = validate_and_canonicalize(raw, min_rows=5)
        assert len(clean) == len(test_csv)
        assert report["valid_rows"] == len(test_csv)
        assert clean["canonical_smiles"].notna().all()

        descriptors, bad_rows = calculate_descriptors(clean)
        assert not bad_rows
        assert descriptors.shape == (len(clean), len(DESCRIPTOR_FUNCTIONS))
        assert np.isfinite(descriptors.to_numpy(dtype=float)).all()

        splits = make_splits(clean, seed=7)
        split_report = validate_splits(clean, splits)
        assert sum(split_report["random"][k] for k in ("train", "validation", "test")) == len(clean)
        assert sum(split_report["scaffold"][k] for k in ("train", "validation", "test")) == len(clean)

        # Reproducibility check.
        splits_again = make_splits(clean, seed=7)
        for split_type in splits:
            for subset in splits[split_type]:
                assert np.array_equal(
                    splits[split_type][subset],
                    splits_again[split_type][subset],
                )

        # End-to-end run using local fixture; must not download real QM9.
        result = build_pipeline(
            output_dir=temporary / "outputs",
            source_csv=csv_path,
            seed=7,
            min_rows=5,
        )
        assert result["processed_csv"].exists()
        assert (temporary / "outputs" / "metadata.json").exists()
        assert (temporary / "outputs" / "random_train.csv").exists()
        assert (temporary / "outputs" / "scaffold_test.csv").exists()

    LOG.info("All offline self-tests passed.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the QM9 molecular-property data pipeline."
    )
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output-dir", default="outputs/module1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--source-csv", default=None)
    parser.add_argument("--test", action="store_true", help="Run offline self-tests and exit.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    configure_logging(not args.quiet)
    if args.test:
        run_self_tests()
        return

    build_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        force_download=args.force_download,
        source_csv=args.source_csv,
    )


if __name__ == "__main__":
    main()
