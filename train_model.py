"""
STEP 1 - Train the Random Forest and save everything the dashboard needs.
Run once:   python train_model.py
"""
import json, joblib, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold, cross_validate

FEATURES = ["c160", "c161", "c180", "c181", "c182", "c183",
            "acidite", "peroxyde", "k232", "k270"]
KEEP = ["Chetoui", "Chemlali", "Zarrazi", "Fakhari"]      # varieties with n >= 4

df = pd.read_csv("data_clean.csv")
df["target"] = np.where(df["variete"].isin(KEEP), df["variete"], "Autres")
X, y = df[FEATURES], df["target"]

def make_model():
    # imputer fills missing values with the median -> optional inputs can be left empty
    return make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=0),
    )

# honest performance estimate: samples of one publication never in train and test together
scores = []
for seed in range(10):
    cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
    r = cross_validate(make_model(), X, y, cv=cv, groups=df["lienPublication"],
                       scoring=["accuracy", "balanced_accuracy", "f1_macro"])
    scores.append([r["test_accuracy"].mean(), r["test_balanced_accuracy"].mean(), r["test_f1_macro"].mean()])
acc, bacc, f1 = np.mean(scores, axis=0)

# final model = trained on ALL the data
model = make_model().fit(X, y)

meta = {
    "features": FEATURES,
    "classes": list(model.classes_),
    "medians": X.median().round(3).to_dict(),
    "min": X.min().to_dict(),
    "max": X.max().to_dict(),
    "profiles": df.groupby("target")[FEATURES].mean().round(2).to_dict(orient="index"),
    "counts": y.value_counts().to_dict(),
    "n_samples": int(len(df)),
    "cv": {"accuracy": round(acc, 3), "balanced_accuracy": round(bacc, 3), "f1_macro": round(f1, 3)},
    "importance": dict(zip(FEATURES, model[-1].feature_importances_.round(3).tolist())),
}
joblib.dump(model, "model.joblib")
json.dump(meta, open("model_meta.json", "w"), indent=2)
print("Saved model.joblib and model_meta.json")
print("Grouped CV  accuracy=%.3f  balanced_acc=%.3f  macroF1=%.3f" % (acc, bacc, f1))
