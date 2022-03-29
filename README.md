[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
# WOE-Scoring
Monotone Weight Of Evidence Transformer and LogisticRegression model with scikit-learn API

# Quickstart

1. Install the package:
```bash
pip install woe-scoring
```

2. Use WOETransformer:
```python
import pandas as pd
from woe_scoring import WOETransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('titanic_data.csv')
train, test = train_test_split(
    df, test_size=0.3, random_state=42, stratify=df["Survived"]
)

special_cols = [
    "PassengerId",
    "Survived",
    "Name",
    "Ticket",
    "Cabin",
]

cat_cols = [
    "Pclass",
    "Sex",
    "SibSp",
    "Parch",
    "Embarked",
]

encoder = WOETransformer(
    max_bins=8,
    min_pct_group=0.1,
    diff_woe_threshold=0.1,
    cat_features=cat_cols,
    special_cols=special_cols,
    n_jobs=-1,
    merge_type='chi2',
)

encoder.fit(train, train["Survived"])
encoder.save("train_dict.json")

enc_train = encoder.transform(train)
enc_test = encoder.transform(test)

model = LogisticRegression()
model.fit(enc_train, train["Survived"])
test_proba = model.predict_proba(enc_test)[:, 1]
```
3. Use CreateModel:

```python
import pandas as pd
from woe_scoring import CreateModel
from sklearn.model_selection import train_test_split

df = pd.read_csv('titanic_data.csv')
train, test = train_test_split(
    df, test_size=0.3, random_state=42, stratify=df["Survived"]
)

special_cols = [
    "PassengerId",
    "Survived",
    "Name",
    "Ticket",
    "Cabin",
]

model = CreateModel(
    max_vars=0.8,
    special_cols=special_cols,
    n_jobs=-1,
    random_state=42,
    class_weight='balanced',
    cv=3,
)
model.fit(train, train["Survived"])
model.save_reports("/")
test_proba = model.predict_proba(test[model.feature_names_])
```
