from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

print("scikit-learn version:", sklearn.__version__)

data_path = Path(__file__).resolve().parent / "modified_data.csv"

if not data_path.exists():
    raise FileNotFoundError(
        f"Place modified_data.csv in this folder: {data_path.parent}"
    )

data = pd.read_csv(data_path)

print("\nDATASET OVERVIEW")
print("Rows and columns:", data.shape)
print(data.head().to_string(index=False))

print("\nDATA QUALITY CHECKS")
print("Missing values:", int(data.isna().sum().sum()))
print("Exact duplicate rows:", int(data.duplicated().sum()))
print("Nonpositive prices:", int((data["price"] <= 0).sum()))
print("Price range:", data["price"].min(), "to", data["price"].max())

clean = data.loc[
    data["price"].notna() & (data["price"] > 0)
].drop_duplicates().copy()

groups = clean[["street", "city", "statezip"]].astype(str).agg(
    "|".join, axis=1
)

X = clean.drop(
    columns=["price", "price_per_sqft", "street", "date"]
)
y = clean["price"]

print("\nMODEL INPUTS")
print("Usable records:", len(clean))
print("Number of predictors:", X.shape[1])
print("Predictors:", X.columns.tolist())

assert "price_per_sqft" not in X.columns
assert "price" not in X.columns

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(splitter.split(X, y, groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

assert set(groups.iloc[train_idx]).isdisjoint(
    set(groups.iloc[test_idx])
)

print("\nTRAINING AND TEST SETS")
print("Training rows:", len(X_train))
print("Test rows:", len(X_test))

numeric = X_train.select_dtypes(include="number").columns.tolist()
categorical = X_train.select_dtypes(exclude="number").columns.tolist()

preprocess = ColumnTransformer([
    (
        "numeric",
        SimpleImputer(strategy="median"),
        numeric
    ),
    (
        "categorical",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]),
        categorical
    )
])

model = Pipeline([
    ("preprocess", preprocess),
    (
        "forest",
        RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    )
])

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("\nMODEL TRAINED")
print("Random Forest: 200 trees, minimum 2 samples per leaf.")

baseline = DummyRegressor(strategy="mean").fit(X_train, y_train)

def scores(name, actual, predicted):
    return {
        "Model": name,
        "MAE": mean_absolute_error(actual, predicted),
        "MSE": mean_squared_error(actual, predicted),
        "R^2": r2_score(actual, predicted)
    }

results = pd.DataFrame([
    scores(
        "Training mean baseline",
        y_test,
        baseline.predict(X_test)
    ),
    scores("Random Forest", y_test, predictions)
])

print("\nTEST RESULTS")
print(
    results.to_string(
        index=False,
        float_format=lambda value: f"{value:,.4f}"
    )
)


output = Path(__file__).resolve().parent / "results"
output.mkdir(exist_ok=True)
results.to_csv(output / "metrics.csv", index=False)
fig, ax = plt.subplots(figsize=(8, 3.1), layout="constrained")
bars = ax.barh(results["Model"], results["MAE"], color=["#89949e", "#14665d"])
ax.bar_label(bars, labels=[f"${value:,.0f}" for value in results["MAE"]], padding=5)
ax.set_xlim(0, results["MAE"].max() * 1.23)
ax.set_title("How far off were the sale-price estimates?", loc="left")
ax.set_xlabel("Mean absolute error on held-out sales (USD)")
ax.xaxis.set_major_formatter(lambda x, pos: f"${x/1000:.0f}k")
ax.spines[["top", "right", "left"]].set_visible(False)
fig.savefig(output / "prediction-error.png", dpi=160)

