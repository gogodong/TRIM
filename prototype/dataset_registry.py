"""Registry for the datasets bundled with the TRIM experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .dataloader import (
    ACSIncomeDataLoader,
    ACSPublicCoverageDataLoader,
    BNGCreditGDataLoader,
    BankMarketingDataLoader,
    DiabetesReadmissionDataLoader,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
GENERALIZATION_TREE_DIR = PROJECT_ROOT / "configs" / "generalization_trees"


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    paper_name: str
    full_name: str
    loader_class: type
    tree_filename: str
    aliases: tuple[str, ...] = ()
    synthetic: bool = False

    @property
    def generalization_tree_path(self) -> Path:
        return GENERALIZATION_TREE_DIR / self.tree_filename


DATASETS = {
    "income": DatasetSpec(
        key="income",
        paper_name="Income",
        full_name="acs_income",
        loader_class=ACSIncomeDataLoader,
        tree_filename="acs_income_generalization_trees.yaml",
        aliases=("acs", "acs_income"),
    ),
    "pubcov": DatasetSpec(
        key="pubcov",
        paper_name="PubCov",
        full_name="acs_public_coverage",
        loader_class=ACSPublicCoverageDataLoader,
        tree_filename="acs_public_coverage_generalization_tree.yaml",
        aliases=("acs_pubcov", "public_coverage", "acs_public_coverage"),
    ),
    "diabetes": DatasetSpec(
        key="diabetes",
        paper_name="Diabetes",
        full_name="diabetes_130_us_hospitals",
        loader_class=DiabetesReadmissionDataLoader,
        tree_filename="diabetes_130_us_hospitals_generalization_tree.yaml",
        aliases=("diabetes_130_us_hospitals",),
    ),
    "bm": DatasetSpec(
        key="bm",
        paper_name="BM",
        full_name="bank_marketing",
        loader_class=BankMarketingDataLoader,
        tree_filename="bank_marketing_generalization_tree.yaml",
        aliases=("bank", "bank_marketing", "bankmarketing"),
    ),
    "bng": DatasetSpec(
        key="bng",
        paper_name="BNG",
        full_name="bng_credit_g",
        loader_class=BNGCreditGDataLoader,
        tree_filename="bng_credit_g_generalization_tree.yaml",
        aliases=("credit_g", "bng_credit_g", "bng-credit-g"),
        synthetic=True,
    ),
}

DATASET_ALIASES = {
    alias: key
    for key, spec in DATASETS.items()
    for alias in (key, *spec.aliases)
}


def resolve_dataset(dataset: str) -> DatasetSpec:
    normalized = str(dataset).strip().lower().replace("-", "_")
    key = DATASET_ALIASES.get(normalized)
    if key is None:
        supported = ", ".join(DATASETS)
        raise ValueError(
            f"Unsupported dataset {dataset!r}. Choose one of: {supported}."
        )
    return DATASETS[key]


def build_data_loader(dataset: str, data_path: str | Path | None = None):
    """Create the configured dataset loader."""
    spec = resolve_dataset(dataset)
    if data_path is None:
        return spec.loader_class()

    path = Path(data_path).expanduser()
    if spec.key in {"income", "pubcov"}:
        return spec.loader_class(data_dir=path)
    if path.is_dir():
        return spec.loader_class(data_dir=path)
    return spec.loader_class(csv_path=path)


def resolve_generalization_tree(
    dataset: str,
    tree_path: str | Path | None = None,
) -> Path:
    spec = resolve_dataset(dataset)
    path = (
        Path(tree_path).expanduser()
        if tree_path is not None
        else spec.generalization_tree_path
    )
    if not path.exists():
        raise FileNotFoundError(
            f"Generalization tree for {spec.paper_name} was not found: {path}"
        )
    return path
