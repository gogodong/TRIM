"""Dataset loaders and hierarchical feature encodings for TRIM."""
import re
from numbers import Real
from pathlib import Path

import yaml


ACS_INCOME_FEATURE_COLUMNS = (
    "AGEP",
    "COW",
    "SCHL",
    "MAR",
    "OCCP",
    "POBP",
    "RELP",
    "WKHP",
    "SEX",
    "RAC1P",
)
ACS_INCOME_NUMERIC_ATTRIBUTES = ("AGEP", "WKHP")
ACS_INCOME_CATEGORICAL_ATTRIBUTES = tuple(
    attribute
    for attribute in ACS_INCOME_FEATURE_COLUMNS
    if attribute not in ACS_INCOME_NUMERIC_ATTRIBUTES
)
BANK_MARKETING_TARGET_COLUMN = "y"
BANK_MARKETING_TARGET_LABELS = {
    "no": 0,
    "yes": 1,
}
BANK_MARKETING_FEATURE_COLUMNS = (
    "age",
    "job",
    "marital",
    "education",
    "default",
    "balance",
    "housing",
    "loan",
    "contact",
    "day",
    "month",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
)
BANK_MARKETING_QI_ATTRIBUTES = (
    "age",
    "job",
    "marital",
    "education",
    "housing",
    "loan",
    "month",
)
BANK_MARKETING_NUMERIC_ATTRIBUTES = (
    "age",
    "balance",
    "day",
    "duration",
    "campaign",
    "pdays",
    "previous",
)

BNG_CREDIT_G_TARGET_COLUMN = "class"
BNG_CREDIT_G_TARGET_LABELS = {
    "bad": 0,
    "good": 1,
}
BNG_CREDIT_G_FEATURE_COLUMNS = (
    "checking_status",
    "duration",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_status",
    "employment",
    "installment_commitment",
    "personal_status",
    "other_parties",
    "residence_since",
    "property_magnitude",
    "age",
    "other_payment_plans",
    "housing",
    "existing_credits",
    "job",
    "num_dependents",
    "own_telephone",
    "foreign_worker",
)
BNG_CREDIT_G_NUMERIC_ATTRIBUTES = (
    "duration",
    "credit_amount",
    "installment_commitment",
    "residence_since",
    "age",
    "existing_credits",
    "num_dependents",
)

DIABETES_READMISSION_FEATURE_COLUMNS = (
    "race",
    "gender",
    "age",
    "weight",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "time_in_hospital",
    "payer_code",
    "medical_specialty",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "diag_1",
    "diag_2",
    "diag_3",
    "number_diagnoses",
    "max_glu_serum",
    "A1Cresult",
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "examide",
    "citoglipton",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
    "change",
    "diabetesMed",
)
DIABETES_READMISSION_QI_ATTRIBUTES = (
    "age",
    "race",
    "gender",
    "admission_type_id",
    "admission_source_id",
    "discharge_disposition_id",
    "time_in_hospital",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "payer_code",
    "medical_specialty",
    "diag_1",
    "diag_2",
    "diag_3",
)
DIABETES_READMISSION_NUMERIC_ATTRIBUTES = (
    "time_in_hospital",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_diagnoses",
)
DIABETES_READMISSION_CODE_ATTRIBUTES = (
    "admission_type_id",
    "admission_source_id",
    "discharge_disposition_id",
)
DIABETES_READMISSION_DIAGNOSIS_ATTRIBUTES = ("diag_1", "diag_2", "diag_3")
DIABETES_READMISSION_TARGET_COLUMN = "readmitted"
DIABETES_READMISSION_TARGET_LABELS = {
    "NO": 0,
    ">30": 1,
    "<30": 1,
}
DIABETES_READMISSION_OMITTED_VALUES = {
    "race": frozenset({"?"}),
    "gender": frozenset({"Unknown/Invalid"}),
    "weight": frozenset({"?"}),
    "admission_type_id": frozenset({"5", "6", "8"}),
    "admission_source_id": frozenset({"9", "17", "20"}),
    "discharge_disposition_id": frozenset({"18", "25"}),
    "payer_code": frozenset({"?"}),
    "medical_specialty": frozenset({"?"}),
    "diag_1": frozenset({"?"}),
    "diag_2": frozenset({"?"}),
    "diag_3": frozenset({"?"}),
}


class ACSIncomeDataLoader:
    task_type = "classification"

    def __init__(
        self,
        data_dir=None,
        state="CA",
        year=2014,
        feature_columns=None,
        numeric_attributes=None,
        qi_attributes=None,
        dataset_stem="ACSIncome",
    ):
        # dataset_stem drives both the cached CSV file names and the folktables
        # problem used to regenerate features/labels; ACSIncomeDataLoader stays
        # the income default, subclasses override stem + columns.
        self.dataset_stem = dataset_stem
        self.state = state
        self.year = year
        self.data_dir = Path(data_dir) if data_dir is not None else self._default_data_dir()
        self.feature_columns = (
            tuple(feature_columns) if feature_columns is not None
            else ACS_INCOME_FEATURE_COLUMNS
        )
        self.qi_attributes = (
            tuple(qi_attributes) if qi_attributes is not None
            else tuple(self.feature_columns)
        )
        self.numeric_attributes = (
            tuple(numeric_attributes) if numeric_attributes is not None
            else ACS_INCOME_NUMERIC_ATTRIBUTES
        )
        self.categorical_attributes = tuple(
            attribute
            for attribute in self.feature_columns
            if attribute not in self.numeric_attributes
        )
        self.category_maps = {}
        self.X_raw = None
        self.y = None

    def load(self, nrows=None):
        import pandas as pd

        x_path, y_path = self._data_file_paths()
        self._ensure_data_files(x_path, y_path)

        # Load the original ACS Income feature table and labels.
        X = pd.read_csv(x_path, nrows=nrows)
        y = pd.read_csv(y_path, nrows=nrows).iloc[:, 0]

        # Keep only the ACS Income model feature columns in a stable order.
        self.X_raw = X.loc[:, self.feature_columns].copy()
        self.y = y

        # Build category maps from the full raw table if no tree has provided them.
        if not self.category_maps:
            self.category_maps = self._build_category_maps_from_full_rows(
                fallback_X=self.X_raw,
            )

        return self.X_raw, self.y

    def _data_file_paths(self):
        return (
            self.data_dir / f"{self.dataset_stem}_{self.state}_{self.year}_X.csv",
            self.data_dir / f"{self.dataset_stem}_{self.state}_{self.year}_y.csv",
        )

    def _ensure_data_files(self, x_path, y_path):
        if x_path.exists() and y_path.exists():
            return

        self._download_folktables_data(x_path, y_path)

    def _download_folktables_data(self, x_path, y_path):
        try:
            from folktables import ACSDataSource, ACSIncome
        except ImportError as exc:
            raise FileNotFoundError(
                f"Could not find {self.dataset_stem} CSV files under "
                f"{self.data_dir}. Install folktables with "
                "`python -m pip install folktables` to download them "
                "automatically."
            ) from exc

        # Resolve the folktables task lazily so cached CSV files do not require
        # folktables at runtime.
        if self.dataset_stem == "ACSIncome":
            problem_class = ACSIncome
        elif self.dataset_stem == "acs_public_coverage":
            try:
                from folktables import ACSPublicCoverage
            except ImportError as exc:
                raise FileNotFoundError(
                    "Could not find acs_public_coverage CSV files under "
                    f"{self.data_dir}. Install folktables with "
                    "`python -m pip install folktables` to download them "
                    "automatically."
                ) from exc
            problem_class = ACSPublicCoverage
        else:
            raise ValueError(
                f"No folktables problem mapping for dataset_stem={self.dataset_stem!r}"
            )

        self.data_dir.mkdir(parents=True, exist_ok=True)
        data_source = ACSDataSource(
            survey_year=str(self.year),
            horizon="1-Year",
            survey="person",
            root_dir=str(self.data_dir),
        )
        acs_data = data_source.get_data(states=[self.state], download=True)
        X, y, _ = problem_class.df_to_pandas(acs_data)

        X = X.loc[:, self.feature_columns].copy()
        X.to_csv(x_path, index=False)
        y.to_csv(y_path, index=False)

    def encode_original(self, X=None):
        import pandas as pd

        if X is None:
            if self.X_raw is None:
                self.load()
            X = self.X_raw

        if not self.category_maps:
            self.category_maps = self._build_category_maps_from_full_rows(
                fallback_X=self.X_raw if self.X_raw is not None else X,
            )

        encoded_parts = []

        # Numeric attributes stay as numeric model inputs.
        numeric_frame = X.loc[:, self.numeric_attributes].apply(pd.to_numeric)
        encoded_parts.append(numeric_frame.astype(float))

        # Categorical attributes become one-hot model inputs.
        for attribute in self.categorical_attributes:
            one_hot_columns = {}
            for category in self.category_maps[attribute]:
                column_name = f"{attribute}={category}"
                one_hot_columns[column_name] = (X[attribute] == category).astype(float)
            encoded_parts.append(pd.DataFrame(one_hot_columns, index=X.index))

        return pd.concat(encoded_parts, axis=1)

    def encode(self, X=None):
        return self.encode_original(X)

    def _build_category_maps_from_data(self, X):
        category_maps = {}
        for attribute in self.categorical_attributes:
            categories = sorted(
                {
                    self._normalize_category_value(category)
                    for category in X[attribute].dropna().unique().tolist()
                }
            )
            category_maps[attribute] = {
                category: index
                for index, category in enumerate(categories)
            }
        return category_maps

    def _normalize_category_value(self, value):
        if isinstance(value, Real) and not isinstance(value, bool):
            float_value = float(value)
            if float_value.is_integer():
                return int(value)
        return value

    def _build_category_maps_from_full_rows(self, fallback_X=None):
        import pandas as pd

        x_path, _ = self._data_file_paths()
        if x_path.exists():
            X_schema = pd.read_csv(x_path)
            X_schema = X_schema.loc[:, self.feature_columns]
            return self._build_category_maps_from_data(X_schema)

        if fallback_X is None:
            raise FileNotFoundError(f"Could not find feature data at {x_path}")
        return self._build_category_maps_from_data(fallback_X)

    def _default_data_dir(self):
        project_root = Path(__file__).resolve().parent.parent
        return project_root / "data" / "acs"


# ACS PUMS Public Coverage task used by the PubCov dataset loader.
ACS_PUBLIC_COVERAGE_FEATURE_COLUMNS = (
    "AGEP",
    "SCHL",
    "MAR",
    "SEX",
    "DIS",
    "ESP",
    "CIT",
    "MIG",
    "MIL",
    "ANC",
    "NATIVITY",
    "DEAR",
    "DEYE",
    "DREM",
    "PINCP",
    "ESR",
    "ST",
    "FER",
    "RAC1P",
)
ACS_PUBLIC_COVERAGE_NUMERIC_ATTRIBUTES = ("AGEP", "PINCP")


class ACSPublicCoverageDataLoader(ACSIncomeDataLoader):
    """Loader for the folktables ACS Public Coverage classification task.

    Reuses the ACSIncomeDataLoader machinery but swaps in the public coverage
    feature columns / numeric attributes and the lowercase
    `acs_public_coverage_<state>_<year>_*` CSV stem. Labels are regenerated from
    folktables on first load if the cached `_y.csv` is missing.
    """

    def __init__(self, data_dir=None, state="CA", year=2014):
        super().__init__(
            data_dir=data_dir,
            state=state,
            year=year,
            feature_columns=ACS_PUBLIC_COVERAGE_FEATURE_COLUMNS,
            numeric_attributes=ACS_PUBLIC_COVERAGE_NUMERIC_ATTRIBUTES,
            dataset_stem="acs_public_coverage",
        )


class BankMarketingDataLoader(ACSIncomeDataLoader):
    task_type = "classification"
    feature_columns = BANK_MARKETING_FEATURE_COLUMNS
    qi_attributes = BANK_MARKETING_QI_ATTRIBUTES
    numeric_attributes = BANK_MARKETING_NUMERIC_ATTRIBUTES

    def __init__(self, csv_path=None, data_dir=None, bank_csv=None):
        if csv_path is None:
            csv_path = bank_csv
        super().__init__(
            data_dir=data_dir,
            feature_columns=BANK_MARKETING_FEATURE_COLUMNS,
            numeric_attributes=BANK_MARKETING_NUMERIC_ATTRIBUTES,
            qi_attributes=BANK_MARKETING_QI_ATTRIBUTES,
            dataset_stem="bank_marketing",
        )
        self.csv_path = Path(csv_path).expanduser() if csv_path is not None else None

    def _default_data_dir(self):
        project_root = Path(__file__).resolve().parent.parent
        return project_root / "data" / "bank_marketing"

    def load(self, nrows=None):
        import pandas as pd

        csv_path = self._resolve_csv_path()
        frame = pd.read_csv(csv_path, sep=";", nrows=nrows, encoding="utf-8-sig")
        schema_frame = pd.read_csv(
            csv_path,
            sep=";",
            usecols=list(self.feature_columns),
            encoding="utf-8-sig",
        )
        missing = [
            column
            for column in (*self.feature_columns, BANK_MARKETING_TARGET_COLUMN)
            if column not in frame.columns
        ]
        if missing:
            raise KeyError(f"Bank Marketing CSV is missing columns: {missing}")

        self.X_raw = frame.loc[:, self.feature_columns].copy()
        target_text = (
            frame[BANK_MARKETING_TARGET_COLUMN].astype(str).str.strip().str.lower()
        )
        self.y = target_text.map(BANK_MARKETING_TARGET_LABELS)
        if self.y.isna().any():
            unknown = sorted(target_text[self.y.isna()].unique().tolist())
            raise ValueError(f"Unknown Bank Marketing labels: {unknown}")
        self.y = self.y.astype(int)
        self.classes_ = sorted(set(BANK_MARKETING_TARGET_LABELS.values()))
        self.n_classes = len(self.classes_)
        self.category_maps = self._build_category_maps_from_data(schema_frame)
        return self.X_raw, self.y

    def _resolve_csv_path(self):
        if self.csv_path is not None:
            if not self.csv_path.exists():
                raise FileNotFoundError(f"Bank Marketing CSV not found: {self.csv_path}")
            return self.csv_path

        candidates = []
        if self.data_dir is not None:
            data_path = Path(self.data_dir)
            if data_path.suffix.lower() == ".csv":
                candidates.append(data_path)
            else:
                candidates.append(data_path / "bank-full.csv")

        project_root = Path(__file__).resolve().parent.parent
        candidates.append(project_root / "data" / "bank_marketing" / "bank-full.csv")
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            "bank-full.csv not found. Pass its location via "
            "BankMarketingDataLoader(csv_path=...) or data_dir."
        )

    def _build_category_maps_from_full_rows(self, fallback_X=None):
        import pandas as pd

        try:
            schema_frame = pd.read_csv(
                self._resolve_csv_path(),
                sep=";",
                usecols=list(self.feature_columns),
                encoding="utf-8-sig",
            )
            return self._build_category_maps_from_data(schema_frame)
        except FileNotFoundError:
            if fallback_X is None:
                raise
            return self._build_category_maps_from_data(fallback_X)


class BNGCreditGDataLoader(ACSIncomeDataLoader):
    task_type = "classification"
    feature_columns = BNG_CREDIT_G_FEATURE_COLUMNS
    numeric_attributes = BNG_CREDIT_G_NUMERIC_ATTRIBUTES

    def __init__(self, csv_path=None, data_dir=None, bng_credit_g_file=None):
        if csv_path is None:
            csv_path = bng_credit_g_file
        super().__init__(
            data_dir=data_dir,
            feature_columns=BNG_CREDIT_G_FEATURE_COLUMNS,
            numeric_attributes=BNG_CREDIT_G_NUMERIC_ATTRIBUTES,
            qi_attributes=BNG_CREDIT_G_FEATURE_COLUMNS,
            dataset_stem="bng_credit_g",
        )
        self.csv_path = Path(csv_path).expanduser() if csv_path is not None else None

    def load(self, nrows=None):
        import pandas as pd

        data_path = self._resolve_data_path()
        frame = self._read_data_frame(data_path, nrows=nrows)
        missing = [
            column
            for column in (*self.feature_columns, BNG_CREDIT_G_TARGET_COLUMN)
            if column not in frame.columns
        ]
        if missing:
            raise KeyError(f"BNG credit-g file {data_path} is missing columns: {missing}")

        self.X_raw = frame.loc[:, self.feature_columns].copy()
        for attribute in self.numeric_attributes:
            self.X_raw[attribute] = pd.to_numeric(
                self.X_raw[attribute], errors="coerce"
            )
        for attribute in self.categorical_attributes:
            self.X_raw[attribute] = self.X_raw[attribute].astype("string").str.strip()

        target_text = (
            frame[BNG_CREDIT_G_TARGET_COLUMN].astype(str).str.strip().str.lower()
        )
        self.y = target_text.map(BNG_CREDIT_G_TARGET_LABELS)
        if self.y.isna().any():
            unknown = sorted(target_text[self.y.isna()].unique().tolist())
            raise ValueError(f"Unknown BNG credit-g labels: {unknown}")
        self.y = self.y.astype(int)
        self.classes_ = sorted(set(BNG_CREDIT_G_TARGET_LABELS.values()))
        self.n_classes = len(self.classes_)
        self.category_maps = self._build_category_maps_from_full_rows(
            fallback_X=self.X_raw,
        )
        return self.X_raw, self.y

    def _default_data_dir(self):
        project_root = Path(__file__).resolve().parent.parent
        return project_root / "data" / "bng_credit_g"

    def _resolve_data_path(self):
        if self.csv_path is not None:
            if not self.csv_path.exists():
                raise FileNotFoundError(f"BNG credit-g file not found: {self.csv_path}")
            return self.csv_path

        candidates = []
        if self.data_dir is not None:
            data_path = Path(self.data_dir)
            if data_path.suffix.lower() in {".arff", ".csv"}:
                candidates.append(data_path)
            else:
                candidates.append(data_path / "BNG_credit-g.arff")
                candidates.append(data_path / "BNG_credit-g.csv")

        project_root = Path(__file__).resolve().parent.parent
        candidates.extend(
            [
                project_root / "data" / "bng_credit_g" / "BNG_credit-g.arff",
                project_root / "data" / "bng_credit_g" / "BNG_credit-g.csv",
            ]
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            "BNG_credit-g ARFF/CSV not found. Pass its location via "
            "BNGCreditGDataLoader(csv_path=...) or data_dir."
        )

    def _read_data_frame(self, data_path, nrows=None, usecols=None):
        import pandas as pd

        data_path = Path(data_path)
        if data_path.suffix.lower() == ".arff":
            attribute_names, data_start, _domains = self._read_arff_header(data_path)
            frame = pd.read_csv(
                data_path,
                header=None,
                names=attribute_names,
                skiprows=data_start,
                quotechar="'",
                comment="%",
                skip_blank_lines=True,
                nrows=nrows,
            )
        else:
            frame = pd.read_csv(data_path, nrows=nrows, encoding="utf-8-sig")

        if usecols is not None:
            return frame.loc[:, list(usecols)]
        return frame

    def _read_arff_header(self, data_path):
        import csv

        attribute_names = []
        domains = {}
        with Path(data_path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle):
                stripped = line.strip()
                lowered = stripped.lower()
                if not stripped or stripped.startswith("%"):
                    continue
                if lowered.startswith("@data"):
                    return attribute_names, line_number + 1, domains
                if not lowered.startswith("@attribute"):
                    continue

                parts = stripped.split(None, 2)
                if len(parts) < 3:
                    raise ValueError(
                        f"Invalid ARFF attribute declaration in {data_path}: {stripped}"
                    )
                attribute_name = parts[1].strip("'\"")
                attribute_names.append(attribute_name)
                declaration = parts[2].strip()
                if declaration.startswith("{") and declaration.endswith("}"):
                    values = next(
                        csv.reader(
                            [declaration[1:-1]],
                            quotechar="'",
                            skipinitialspace=True,
                        )
                    )
                    domains[attribute_name] = tuple(value.strip() for value in values)
        raise ValueError(f"ARFF file {data_path} does not contain an @data section")

    def _build_category_maps_from_full_rows(self, fallback_X=None):
        import pandas as pd

        data_path = self._resolve_data_path()
        if data_path.suffix.lower() == ".arff":
            _attribute_names, _data_start, domains = self._read_arff_header(data_path)
            category_maps = {}
            for attribute in self.categorical_attributes:
                if attribute not in domains:
                    continue
                categories = sorted(
                    self._normalize_category_value(category)
                    for category in domains[attribute]
                )
                category_maps[attribute] = {
                    category: index
                    for index, category in enumerate(categories)
                }
            if len(category_maps) == len(self.categorical_attributes):
                return category_maps

        try:
            schema_frame = self._read_data_frame(
                data_path,
                usecols=list(self.feature_columns),
            )
            for attribute in self.numeric_attributes:
                schema_frame[attribute] = pd.to_numeric(
                    schema_frame[attribute], errors="coerce"
                )
            return self._build_category_maps_from_data(schema_frame)
        except FileNotFoundError:
            if fallback_X is None:
                raise
            return self._build_category_maps_from_data(fallback_X)


class DiabetesReadmissionDataLoader(ACSIncomeDataLoader):
    task_type = "classification"
    feature_columns = DIABETES_READMISSION_FEATURE_COLUMNS
    qi_attributes = DIABETES_READMISSION_QI_ATTRIBUTES
    numeric_attributes = DIABETES_READMISSION_NUMERIC_ATTRIBUTES
    string_categorical_attributes = DIABETES_READMISSION_DIAGNOSIS_ATTRIBUTES

    def __init__(self, csv_path=None, data_dir=None):
        super().__init__(
            data_dir=data_dir,
            feature_columns=DIABETES_READMISSION_FEATURE_COLUMNS,
            numeric_attributes=DIABETES_READMISSION_NUMERIC_ATTRIBUTES,
            qi_attributes=DIABETES_READMISSION_QI_ATTRIBUTES,
            dataset_stem="diabetes_130_us_hospitals",
        )
        self.csv_path = Path(csv_path).expanduser() if csv_path is not None else None

    def _default_data_dir(self):
        project_root = Path(__file__).resolve().parent.parent
        return project_root / "data" / "diabetes_130_us_hospitals"

    def load(self, nrows=None):
        import pandas as pd

        csv_path = self._resolve_csv_path()
        frame = pd.read_csv(csv_path, nrows=nrows, encoding="utf-8-sig")
        schema_frame = pd.read_csv(
            csv_path,
            usecols=list(self.feature_columns),
            encoding="utf-8-sig",
        )
        self.X_raw = self._prepare_features(frame)
        if DIABETES_READMISSION_TARGET_COLUMN not in frame.columns:
            raise KeyError(
                f"Diabetes CSV {csv_path} is missing target column "
                f"{DIABETES_READMISSION_TARGET_COLUMN!r}."
            )
        target_text = frame[DIABETES_READMISSION_TARGET_COLUMN].astype(str).str.strip()
        self.y = target_text.map(DIABETES_READMISSION_TARGET_LABELS)
        if self.y.isna().any():
            unknown = sorted(target_text[self.y.isna()].unique().tolist())
            raise ValueError(f"Unknown diabetes readmission labels: {unknown}")
        self.y = self.y.astype(int)
        self.classes_ = sorted(set(DIABETES_READMISSION_TARGET_LABELS.values()))
        self.n_classes = len(self.classes_)
        self.category_maps = self._build_category_maps_from_data(
            self._prepare_features(schema_frame)
        )
        return self.X_raw, self.y

    def _prepare_features(self, frame):
        import pandas as pd

        missing = [column for column in self.feature_columns if column not in frame.columns]
        if missing:
            raise KeyError(f"Diabetes CSV is missing feature columns: {missing}")

        features = frame.loc[:, self.feature_columns].copy()
        for attribute, omitted_values in DIABETES_READMISSION_OMITTED_VALUES.items():
            omitted_text = {str(value).strip() for value in omitted_values}
            values = features[attribute].astype("string").str.strip()
            features.loc[values.isin(omitted_text), attribute] = pd.NA

        for attribute in self.numeric_attributes:
            features[attribute] = pd.to_numeric(features[attribute], errors="coerce")
        for attribute in DIABETES_READMISSION_CODE_ATTRIBUTES:
            features[attribute] = pd.to_numeric(features[attribute], errors="coerce")
        for attribute in DIABETES_READMISSION_DIAGNOSIS_ATTRIBUTES:
            values = features[attribute].astype("string").str.strip()
            features[attribute] = values.where(values.notna(), pd.NA)
        return features

    def _resolve_csv_path(self):
        if self.csv_path is not None:
            if not self.csv_path.exists():
                raise FileNotFoundError(f"Diabetes readmission CSV not found: {self.csv_path}")
            return self.csv_path

        candidates = []
        if self.data_dir is not None:
            candidates.append(self.data_dir / "diabetic_data.csv")
        project_root = Path(__file__).resolve().parent.parent
        candidates.append(
            project_root
            / "data"
            / "diabetes_130_us_hospitals"
            / "diabetic_data.csv"
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            "diabetic_data.csv not found. Pass its location via "
            "DiabetesReadmissionDataLoader(csv_path=...)."
        )

    def _build_category_maps_from_full_rows(self, fallback_X=None):
        import pandas as pd

        try:
            schema_frame = pd.read_csv(
                self._resolve_csv_path(),
                usecols=list(self.feature_columns),
                encoding="utf-8-sig",
            )
            return self._build_category_maps_from_data(
                self._prepare_features(schema_frame)
            )
        except FileNotFoundError:
            if fallback_X is None:
                raise
            return self._build_category_maps_from_data(
                self._prepare_features(fallback_X)
            )


class DatasetGeneralization:
    def __init__(
        self,
        generalization_level,
        generalization_rules,
        numeric_attributes,
        categorical_attributes,
        category_maps,
        trees=None,
        data_loader=None,
    ):
        self.generalization_level = generalization_level
        self.generalization_rules = generalization_rules
        self.numeric_attributes = tuple(numeric_attributes)
        self.categorical_attributes = tuple(categorical_attributes)
        self.category_maps = category_maps
        self.trees = trees
        self.data_loader = data_loader

    def change_level(self, generalization_level):
        if self.trees is None or self.data_loader is None:
            raise ValueError("Cannot change level without loaded generalization trees.")

        return load_generalization_rules(
            trees=self.trees,
            data_loader=self.data_loader,
            generalization_level=generalization_level,
        )

    def generalize_value(self, attribute, value):
        attribute_rules = self.generalization_rules.get(attribute, {})
        if isinstance(attribute_rules, list):
            # Continuous attribute whose leaves are value intervals.
            return self._interval_generalized(attribute_rules, value)
        return attribute_rules.get(value, value)

    def _interval_generalized(self, interval_rules, value):
        # interval_rules is a list of (low, high, generalized_value) tuples;
        # pick the generalized value of the leaf interval that contains value.
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return value
        for low, high, generalized in interval_rules:
            if low <= numeric_value <= high:
                return generalized
        return value

    def encode(self, X):
        import pandas as pd

        encoded_parts = []

        # Numeric generalized values are scalar means of the target leaf group.
        for attribute in self.numeric_attributes:
            rules = self.generalization_rules.get(attribute, {})
            column = self._generalized_numeric_series(X, attribute, rules)
            encoded_parts.append(
                pd.DataFrame({attribute: column}, index=X.index)
            )

        # Categorical generalized values are mean one-hot vectors of target leaves.
        for attribute in self.categorical_attributes:
            encoded_parts.append(self._generalized_categorical_frame(X, attribute))

        return pd.concat(encoded_parts, axis=1)

    def encode_xgboost_model_input(self, X):
        import pandas as pd

        encoded_parts = []

        for attribute in self.numeric_attributes:
            level = int(self.generalization_level.get(attribute, 0))
            rules = self.generalization_rules.get(attribute, {})
            if level <= 0:
                encoded_parts.append(
                    pd.DataFrame(
                        {attribute: pd.to_numeric(X[attribute]).astype(float)},
                        index=X.index,
                    )
                )
                continue

            if isinstance(rules, list):
                generalized = self._generalized_numeric_series(X, attribute, rules)
                category_values = sorted({rule[2] for rule in rules})
            else:
                generalized = self._generalized_numeric_series(X, attribute, rules)
                category_values = sorted({float(value) for value in rules.values()})

            if not category_values:
                encoded_parts.append(
                    pd.DataFrame(
                        {attribute: pd.to_numeric(X[attribute]).astype(float)},
                        index=X.index,
                    )
                )
                continue

            category_columns = {}
            for category_value in category_values:
                category_name = self._format_numeric_category_value(category_value)
                column_name = f"{attribute}_cat={category_name}"
                category_columns[column_name] = (
                    generalized.astype(float) == float(category_value)
                ).astype(float)
            encoded_parts.append(pd.DataFrame(category_columns, index=X.index))

        for attribute in self.categorical_attributes:
            encoded_parts.append(self._generalized_categorical_frame(X, attribute))

        return pd.concat(encoded_parts, axis=1)

    def encode_xgboost_leaf_space(self, X, *, include_generalization_level=True):
        """Encode QI hierarchies and raw non-QIs in one fixed sparse space.

        A raw QI leaf is one-hot and a generalized QI node is uniform over its
        descendant leaves. Non-QI model features pass through unchanged. The
        optional QI level columns make released resolution explicit while the
        schema remains identical for every snapshot.
        """
        import numpy as np
        import pandas as pd
        from scipy import sparse

        if self.trees is None or self.data_loader is None:
            raise ValueError(
                "Leaf-space encoding requires generalization trees and a data loader."
            )

        row_count = len(X)
        attribute_blocks = []
        for attribute in self.data_loader.feature_columns:
            tree = self.trees.get(attribute)
            if tree is None:
                if attribute in self.numeric_attributes:
                    values = pd.to_numeric(
                        X[attribute], errors="coerce"
                    ).to_numpy(dtype=np.float32).reshape(-1, 1)
                    attribute_blocks.append(sparse.csr_matrix(values))
                else:
                    raw_frame = self._generalized_categorical_frame(
                        X, attribute
                    )
                    attribute_blocks.append(
                        sparse.csr_matrix(
                            raw_frame.to_numpy(dtype=np.float32, copy=False)
                        )
                    )
                continue

            nodes_by_id = {
                node["id"]: node
                for node in tree.get("nodes", [])
            }
            leaf_nodes = [
                node
                for node in tree.get("nodes", [])
                if node.get("kind") == "leaf"
            ]
            if not leaf_nodes:
                raise ValueError(
                    f"Generalization tree for {attribute!r} has no leaves."
                )

            level = int(self.generalization_level.get(attribute, 0))
            target_ids = []
            for leaf in leaf_nodes:
                target = leaf
                while (
                    int(target.get("height_from_leaf", 0)) < level
                    and target.get("parent") is not None
                ):
                    target = nodes_by_id[target["parent"]]
                target_ids.append(target["id"])

            ordered_target_ids = list(dict.fromkeys(target_ids))
            target_position = {
                target_id: position
                for position, target_id in enumerate(ordered_target_ids)
            }
            leaf_group_ids = np.asarray(
                [target_position[target_id] for target_id in target_ids],
                dtype=np.int32,
            )

            leaf_positions = np.full(row_count, -1, dtype=np.int32)
            if attribute in self.numeric_attributes:
                numeric_values = pd.to_numeric(X[attribute], errors="coerce")
                for leaf_position, leaf in enumerate(leaf_nodes):
                    parsed = _parse_leaf_value(leaf["value"])
                    if parsed[0] == "interval":
                        _, low, high = parsed
                        mask = numeric_values.between(
                            float(low), float(high), inclusive="both"
                        )
                    else:
                        _, value = parsed
                        mask = numeric_values == float(value)
                    assignable = mask.to_numpy() & (leaf_positions < 0)
                    leaf_positions[assignable] = leaf_position
            else:
                leaf_lookup = {
                    _coerce_categorical_leaf_value(
                        self.data_loader, attribute, leaf["value"]
                    ): leaf_position
                    for leaf_position, leaf in enumerate(leaf_nodes)
                }
                values = X[attribute]
                if attribute in getattr(
                    self.data_loader, "string_categorical_attributes", ()
                ):
                    values = values.astype(str)
                mapped = values.map(leaf_lookup)
                present = mapped.notna().to_numpy()
                leaf_positions[present] = mapped.loc[present].astype(int).to_numpy()

            missing_value_mask = X[attribute].isna().to_numpy()
            invalid_value_mask = (leaf_positions < 0) & ~missing_value_mask
            if np.any(invalid_value_mask):
                missing_values = X.iloc[
                    np.flatnonzero(invalid_value_mask)
                ][attribute].drop_duplicates().head(5).tolist()
                raise ValueError(
                    f"Values for {attribute!r} are outside the leaf-space schema: "
                    f"{missing_values!r}"
                )

            present_positions = np.flatnonzero(leaf_positions >= 0)
            row_group_ids = leaf_group_ids[leaf_positions[present_positions]]
            assignment = sparse.csr_matrix(
                (
                    np.ones(len(present_positions), dtype=np.float32),
                    (
                        present_positions.astype(np.int32, copy=False),
                        row_group_ids,
                    ),
                ),
                shape=(row_count, len(ordered_target_ids)),
                dtype=np.float32,
            )
            group_sizes = np.bincount(
                leaf_group_ids, minlength=len(ordered_target_ids)
            ).astype(np.float32)
            group_to_leaf = sparse.csr_matrix(
                (
                    1.0 / group_sizes[leaf_group_ids],
                    (
                        leaf_group_ids,
                        np.arange(len(leaf_nodes), dtype=np.int32),
                    ),
                ),
                shape=(len(ordered_target_ids), len(leaf_nodes)),
                dtype=np.float32,
            )
            attribute_blocks.append(
                assignment.dot(group_to_leaf).tocsr()
            )
            if include_generalization_level:
                if level == 0:
                    level_block = sparse.csr_matrix(
                        (row_count, 1), dtype=np.float32
                    )
                else:
                    level_block = sparse.csr_matrix(
                        np.full((row_count, 1), level, dtype=np.float32)
                    )
                attribute_blocks.append(level_block)

        if not attribute_blocks:
            return sparse.csr_matrix((row_count, 0), dtype=np.float32)
        return sparse.hstack(attribute_blocks, format="csr", dtype=np.float32)

    def _generalized_numeric_series(self, X, attribute, rules):
        import pandas as pd

        values = X[attribute]
        if isinstance(rules, list):
            return self._interval_generalized_series(values, rules)

        if not rules:
            return pd.to_numeric(values).astype(float)

        mapped = values.map(rules)
        generalized = mapped.where(mapped.notna(), values)
        return pd.to_numeric(generalized).astype(float)

    def _interval_generalized_series(self, values, interval_rules):
        import pandas as pd

        numeric_values = pd.to_numeric(values, errors="coerce")
        generalized = pd.Series(values.to_numpy(dtype=object), index=values.index)
        for low, high, replacement in interval_rules:
            mask = numeric_values.between(float(low), float(high), inclusive="both")
            if mask.any():
                generalized.loc[mask] = replacement

        try:
            return generalized.astype(float)
        except (TypeError, ValueError):
            return generalized

    def _generalized_categorical_frame(self, X, attribute):
        import pandas as pd

        rules = self.generalization_rules.get(attribute, {})
        columns = [
            f"{attribute}={category}"
            for category in self.category_maps[attribute]
        ]
        rows = self._generalized_categorical_rows(X[attribute], attribute, rules)
        return pd.DataFrame(rows, columns=columns, index=X.index)

    def _generalized_categorical_rows(self, values, attribute, rules):
        sentinel = object()
        vector_cache = {}
        rows = []

        for value in values:
            cache_key = self._categorical_cache_key(value)
            if cache_key not in vector_cache:
                try:
                    vector = rules.get(value, sentinel)
                except TypeError:
                    vector = sentinel
                if vector is sentinel:
                    vector = self._one_hot_vector(attribute, value)
                vector_cache[cache_key] = tuple(vector)
            rows.append(vector_cache[cache_key])

        return rows

    def _categorical_cache_key(self, value):
        try:
            if value != value:
                return ("__missing__", type(value).__name__)
        except (TypeError, ValueError):
            pass
        return value

    def generalize_record(self, record):
        return {
            attribute: self.generalize_value(attribute, value)
            for attribute, value in record.items()
        }

    def _one_hot_vector(self, attribute, value):
        categories = self.category_maps[attribute]
        vector = [0.0] * len(categories)

        if value in categories:
            vector[categories[value]] = 1.0

        return tuple(vector)

    def _format_numeric_category_value(self, value):
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return str(value)
        if numeric_value.is_integer():
            return str(int(numeric_value))
        return f"{numeric_value:.12g}"


# A continuous leaf may be expressed as an inclusive value range (e.g. AGEP
# "0 ~ 4"). The separator alternation covers the styles the tree-generation
# prompt emits (~, "to", en/em dashes, hyphen); ACS codes are non-negative so a
# hyphen only acts as a range separator here.
_INTERVAL_PATTERN = re.compile(
    r"^\s*([+-]?\d+(?:\.\d+)?)\s*(?:~|to|\u2013|\u2014|-)\s*([+-]?\d+(?:\.\d+)?)\s*$"
)


def _coerce_scalar(raw):
    # Normalise a scalar leaf value to match the codes read from the data CSV.
    # Some coded categorical leaves are quoted strings such as '1' while the
    # corresponding data column is float64, so '1' must become int 1 for the
    # dict/one-hot lookups to hit (int 1 == float 1.0).
    if isinstance(raw, str):
        stripped = raw.strip()
        try:
            number = float(stripped)
        except ValueError:
            return raw
        return int(number) if number.is_integer() else number
    if isinstance(raw, float) and raw.is_integer():
        return int(raw)
    return raw


def _parse_leaf_value(raw):
    """Return ('scalar', value) or ('interval', low, high) for a tree leaf.

    Scalar values are coerced so numeric string codes align with the float CSV
    codes; continuous range leaves ('0 ~ 4') become an inclusive integer tuple.
    """
    if isinstance(raw, str):
        match = _INTERVAL_PATTERN.match(raw)
        if match:
            low = float(match.group(1))
            high = float(match.group(2))
            if low.is_integer() and high.is_integer():
                low, high = int(low), int(high)
            return ("interval", low, high)
    return ("scalar", _coerce_scalar(raw))


def _coerce_categorical_leaf_value(data_loader, attribute, raw):
    if attribute in getattr(data_loader, "string_categorical_attributes", ()):
        return str(raw)
    return _coerce_scalar(raw)


def _interval_union_mean(intervals):
    """Mean of every integer covered by the union of [low, high] intervals.

    Mirrors the income convention where a numeric generalization group is the
    unweighted mean of the distinct observed values it contains.
    """
    covered = set()
    for low, high in intervals:
        covered.update(range(int(low), int(high) + 1))
    return sum(covered) / len(covered)


def load_generalization_rules_from_file(
    file_path,
    data_loader,
    generalization_level=1,
):
    """Load a generalization tree file and build its encoding rules."""
    # Read the YAML generalization tree file.
    data = yaml.safe_load(Path(file_path).read_text(encoding="utf-8-sig"))
    trees = data.get("trees", {})

    return load_generalization_rules(
        trees=trees,
        data_loader=data_loader,
        generalization_level=generalization_level,
    )


def load_generalization_rules(
    trees,
    data_loader,
    generalization_level=1,
):

    qi_attributes = tuple(
        getattr(data_loader, "qi_attributes", data_loader.feature_columns)
    )
    unknown_qi_attributes = sorted(
        set(qi_attributes) - set(data_loader.feature_columns)
    )
    if unknown_qi_attributes:
        raise ValueError(
            "QI attributes must be model feature columns; unknown attributes: "
            f"{unknown_qi_attributes}."
        )
    missing_trees = [
        attribute for attribute in qi_attributes if attribute not in trees
    ]
    if missing_trees:
        raise KeyError(
            "Generalization trees are missing QI attributes: "
            f"{missing_trees}."
        )
    active_trees = {
        attribute: trees[attribute]
        for attribute in qi_attributes
    }

    # Normalize the requested generalization level for each attribute.
    if isinstance(generalization_level, dict):
        levels = {
            attribute: int(generalization_level.get(attribute, 0))
            for attribute in qi_attributes
        }
    else:
        levels = {
            attribute: int(generalization_level)
            for attribute in qi_attributes
        }

    # Tree-backed QIs use their audited leaf domains. Non-QI categoricals keep
    # the full-data maps built by the loader and remain unchanged at every
    # generalization level.
    category_maps = dict(data_loader.category_maps)
    for attribute in data_loader.categorical_attributes:
        if attribute not in active_trees:
            if attribute not in category_maps:
                raise KeyError(
                    "No category map is available for non-QI model feature "
                    f"{attribute!r}. Load the dataset before its tree."
                )
            continue
        tree = active_trees[attribute]
        categories = sorted(
            (
                _coerce_categorical_leaf_value(data_loader, attribute, node["value"])
                for node in tree.get("nodes", [])
                if node.get("kind") == "leaf"
            ),
            key=lambda value: (
                0,
                float(value),
            )
            if isinstance(value, Real) and not isinstance(value, bool)
            else (1, str(value)),
        )
        category_maps[attribute] = {
            category: index
            for index, category in enumerate(categories)
        }
    data_loader.category_maps = category_maps

    # Build model-input generalization rules for each attribute.
    generalization_rules = {}
    for attribute, tree in active_trees.items():

        level = levels.get(attribute, 0)
        nodes_by_id = {
            node["id"]: node
            for node in tree.get("nodes", [])
        }
        leaf_nodes = [
            node
            for node in nodes_by_id.values()
            if node.get("kind") == "leaf"
        ]

        # Parse every leaf once. Numeric attributes may use continuous range
        # leaves like "0 ~ 4"; categorical attributes may legitimately contain
        # hyphenated string values such as "0-500", so they stay scalar.
        if attribute in data_loader.numeric_attributes:
            parsed_leaves = {
                leaf["id"]: _parse_leaf_value(leaf["value"])
                for leaf in leaf_nodes
            }
        else:
            parsed_leaves = {
                leaf["id"]: (
                    "scalar",
                    _coerce_categorical_leaf_value(
                        data_loader, attribute, leaf["value"]
                    ),
                )
                for leaf in leaf_nodes
            }
        is_interval_numeric = (
            attribute in data_loader.numeric_attributes
            and any(parsed[0] == "interval" for parsed in parsed_leaves.values())
        )
        descendant_leaf_values = {}

        # Collect the parsed leaf values under every node.
        for node_id, node in nodes_by_id.items():
            values = []
            for leaf in leaf_nodes:
                current = leaf
                while current is not None:
                    if current["id"] == node_id:
                        values.append(parsed_leaves[leaf["id"]])
                        break
                    parent_id = current.get("parent")
                    current = nodes_by_id.get(parent_id) if parent_id is not None else None
            descendant_leaf_values[node_id] = values

        # Interval-numeric attributes keep a list of (low, high, generalized);
        # everything else stays a dict keyed by the (coerced) leaf value.
        rules = [] if is_interval_numeric else {}
        for leaf in leaf_nodes:
            target_node = leaf

            # Move from the original leaf upward to the requested level.
            while (
                target_node.get("height_from_leaf", 0) < level
                and target_node.get("parent") is not None
            ):
                target_node = nodes_by_id[target_node["parent"]]

            target_leaf_values = descendant_leaf_values[target_node["id"]]

            # Numeric groups are represented by the mean of their leaf values.
            if attribute in data_loader.numeric_attributes:
                if is_interval_numeric:
                    intervals = [
                        (low, high)
                        for kind, low, high in target_leaf_values
                        if kind == "interval"
                    ]
                    generalized = _interval_union_mean(intervals)
                    _, low, high = parsed_leaves[leaf["id"]]
                    rules.append((low, high, generalized))
                else:
                    scalars = [
                        value
                        for kind, value in target_leaf_values
                        if kind == "scalar"
                    ]
                    generalized = sum(scalars) / len(scalars)
                    rules[parsed_leaves[leaf["id"]][1]] = generalized
                continue

            # Categorical groups are represented by the mean of their one-hot leaves.
            vector = [0.0] * len(data_loader.category_maps[attribute])
            for parsed in target_leaf_values:
                index = data_loader.category_maps[attribute][parsed[1]]
                vector[index] += 1.0 / len(target_leaf_values)
            rules[parsed_leaves[leaf["id"]][1]] = tuple(vector)

        generalization_rules[attribute] = rules

    return DatasetGeneralization(
        generalization_level=levels,
        generalization_rules=generalization_rules,
        numeric_attributes=data_loader.numeric_attributes,
        categorical_attributes=data_loader.categorical_attributes,
        category_maps=data_loader.category_maps,
        trees=active_trees,
        data_loader=data_loader,
    )
