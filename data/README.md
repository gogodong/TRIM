# Data preparation

Raw datasets are not distributed with TRIM. Follow the license and terms of
use of each data source. Generalization trees are provided under
`../configs/generalization_trees/`.

Set `data_path` in a configuration file to use a custom location, or use the
default layout below:

```text
data/
├── acs/
│   ├── ACSIncome_CA_2014_X.csv
│   ├── ACSIncome_CA_2014_y.csv
│   ├── acs_public_coverage_CA_2014_X.csv
│   └── acs_public_coverage_CA_2014_y.csv
├── bank_marketing/
│   └── bank-full.csv
├── diabetes_130_us_hospitals/
│   └── diabetic_data.csv
└── bng_credit_g/
    └── BNG_credit-g.arff
```

## Income and PubCov

Income and PubCov use the Folktables `ACSIncome` and `ACSPublicCoverage`
classification tasks. Both use the 2014 California ACS 1-Year person PUMS
data with `state=CA`, `year=2014`, `horizon=1-Year`, and `survey=person`.

When the cached files are missing, the loader can use Folktables to download
the PUMS data and create the four CSV files listed above.

- [Folktables task definitions](https://github.com/socialfoundations/folktables)
- [Census ACS PUMS data](https://www.census.gov/programs-surveys/acs/microdata/access.html)
- [2014 ACS 1-Year PUMS documentation](https://www.census.gov/programs-surveys/acs/microdata/documentation/2014.html)
- [2014 California person PUMS archive](https://www2.census.gov/programs-surveys/acs/data/pums/2014/1-Year/csv_pca.zip)
- [Census terms of service](https://www.census.gov/data/developers/about/terms-of-service.html)

Folktables is distributed under the MIT License. Use ACS data according to the
Census terms.

## Diabetes

Download [Diabetes 130-US Hospitals for Years 1999-2008 (UCI dataset
296)](https://archive.ics.uci.edu/dataset/296/diabetes-130-us-hospitals-for-years-1999-2008)
from the [official ZIP
archive](https://archive.ics.uci.edu/static/public/296/diabetes%2B130-us%2Bhospitals%2Bfor%2Byears%2B1999-2008.zip).
Place `diabetic_data.csv` at
`data/diabetes_130_us_hospitals/diabetic_data.csv`. The UCI page lists the
dataset under CC BY 4.0.

## BM

Download [Bank Marketing (UCI dataset
222)](https://archive.ics.uci.edu/dataset/222/bank%2Bmarketing) from the
[official ZIP
archive](https://archive.ics.uci.edu/static/public/222/bank%2Bmarketing.zip).
Use the full original `bank-full.csv`, not `bank-additional-full.csv`, and
place it at `data/bank_marketing/bank-full.csv`. The UCI page lists the
dataset under CC BY 4.0.

## BNG

Use the synthetic [BNG(credit-g) dataset (OpenML dataset 260, version
1)](https://www.openml.org/d/260), not the smaller `credit-g` dataset 31.
Download the [official ARFF
file](https://www.openml.org/data/v1/download/6349/BNG%28credit-g%29.arff)
and save it as `data/bng_credit_g/BNG_credit-g.arff`. The target column is
`class`.

Follow the [OpenML terms](https://www.openml.org/terms).
