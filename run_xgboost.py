"""Runnable script version of xgboost_model.ipynb — outputs all results to console and saves figures."""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
    f1_score,
    accuracy_score,
)
from xgboost import XGBClassifier

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("=" * 80)
print("SECTION 1: Libraries imported successfully")
print("=" * 80)

# ── 2. Load Data ──────────────────────────────────────────────────────────────
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").values.ravel()
y_test = pd.read_csv("y_test.csv").values.ravel()

print(f"\nSECTION 2: Data loaded")
print(f"  X_train shape: {X_train.shape}")
print(f"  X_test  shape: {X_test.shape}")
print(f"  y_train shape: {y_train.shape}")
print(f"  y_test  shape: {y_test.shape}")

# ── 3. Encode Target & Check Imbalance ────────────────────────────────────────
le = LabelEncoder()
le.fit(["No", "Yes"])
y_train_enc = le.transform(y_train)
y_test_enc = le.transform(y_test)

print(f"\nSECTION 3: Target encoding")
print(f"  Label mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")
print(f"  Training set target distribution:")
unique, counts = np.unique(y_train_enc, return_counts=True)
for label, count in zip(unique, counts):
    print(f"    {le.inverse_transform([label])[0]} ({label}): {count} ({count/len(y_train_enc)*100:.2f}%)")

neg_count = np.sum(y_train_enc == 0)
pos_count = np.sum(y_train_enc == 1)
scale_pos_weight = neg_count / pos_count
print(f"  scale_pos_weight = {scale_pos_weight:.2f}")

# ── 4. Build Pipeline ────────────────────────────────────────────────────────
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("xgb", XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=2,
    )),
])
print(f"\nSECTION 4: Pipeline created")
print(pipeline)

# ── 5. Hyperparameter Tuning ─────────────────────────────────────────────────
param_distributions = {
    "xgb__n_estimators": [100, 200, 300, 400, 500],
    "xgb__max_depth": [3, 4, 5, 6, 7, 8, 10],
    "xgb__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2, 0.3],
}

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_distributions,
    n_iter=50,
    scoring="f1",
    cv=cv_strategy,
    random_state=RANDOM_STATE,
    n_jobs=2,
    verbose=1,
    return_train_score=True,
)

print(f"\nSECTION 5: Starting RandomizedSearchCV (50 iterations, 5-fold CV)...")
t0 = time.time()
random_search.fit(X_train, y_train_enc)
elapsed = time.time() - t0
print(f"Search complete in {elapsed/60:.1f} minutes")

print(f"\n  Best hyperparameters:")
for param, value in random_search.best_params_.items():
    print(f"    {param}: {value}")
print(f"  Best CV F1-score: {random_search.best_score_:.4f}")

best_pipeline = random_search.best_estimator_

# ── 6. Probability Output & Custom Threshold ─────────────────────────────────
y_proba = best_pipeline.predict_proba(X_test)[:, 1]

print(f"\nSECTION 6: Probability output")
print(f"  Min:    {y_proba.min():.4f}")
print(f"  Max:    {y_proba.max():.4f}")
print(f"  Mean:   {y_proba.mean():.4f}")
print(f"  Median: {np.median(y_proba):.4f}")

THRESHOLD = 0.65
y_pred_custom = (y_proba >= THRESHOLD).astype(int)
y_pred_default = (y_proba >= 0.5).astype(int)

print(f"\n  Default threshold (0.50): Predicted Yes={np.sum(y_pred_default==1)}, No={np.sum(y_pred_default==0)}")
print(f"  Custom threshold ({THRESHOLD}): Predicted Yes={np.sum(y_pred_custom==1)}, No={np.sum(y_pred_custom==0)}")

# Probability distribution plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(y_proba[y_test_enc == 0], bins=50, alpha=0.6, label="Actual: No", color="steelblue")
ax.hist(y_proba[y_test_enc == 1], bins=50, alpha=0.6, label="Actual: Yes", color="salmon")
ax.axvline(THRESHOLD, color="red", linestyle="--", linewidth=2, label=f"Threshold = {THRESHOLD}")
ax.set_xlabel("Predicted Probability of Heart Disease")
ax.set_ylabel("Count")
ax.set_title("Distribution of Predicted Probabilities")
ax.legend()
plt.tight_layout()
plt.savefig("fig_probability_distribution.png", dpi=150)
plt.close()
print("  Saved: fig_probability_distribution.png")

# ── 7. Evaluation & Failure Analysis ─────────────────────────────────────────
print(f"\nSECTION 7: Evaluation")
print(f"\n7.1 Classification Report (Threshold = {THRESHOLD})")
print("=" * 60)
target_names = ["No Heart Disease (0)", "Heart Disease (1)"]
print(classification_report(y_test_enc, y_pred_custom, target_names=target_names))

# Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test_enc, y_pred_default, display_labels=target_names, cmap="Blues", ax=axes[0],
)
axes[0].set_title("Confusion Matrix (Default Threshold = 0.50)")
ConfusionMatrixDisplay.from_predictions(
    y_test_enc, y_pred_custom, display_labels=target_names, cmap="Blues", ax=axes[1],
)
axes[1].set_title(f"Confusion Matrix (Custom Threshold = {THRESHOLD})")
plt.tight_layout()
plt.savefig("fig_confusion_matrices.png", dpi=150)
plt.close()
print("  Saved: fig_confusion_matrices.png")

# ROC-AUC
roc_auc = roc_auc_score(y_test_enc, y_proba)
print(f"\n7.3 ROC-AUC Score: {roc_auc:.4f}")

fpr, tpr, thresholds = roc_curve(y_test_enc, y_proba)
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, color="steelblue", linewidth=2, label=f"XGBoost (AUC = {roc_auc:.4f})")
ax.plot([0, 1], [0, 1], color="grey", linestyle="--", label="Random Classifier")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve — XGBoost Classifier")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("fig_roc_curve.png", dpi=150)
plt.close()
print("  Saved: fig_roc_curve.png")

# Failure analysis
print(f"\n7.4 Failure Analysis — Top 10 Most Confident Failures")
print("=" * 80)

failure_df = pd.DataFrame({
    "actual": y_test_enc,
    "predicted_prob": y_proba,
    "predicted_label": y_pred_custom,
})
failure_df["error"] = failure_df["actual"] != failure_df["predicted_label"]
failure_df["confidence"] = np.abs(failure_df["predicted_prob"] - 0.5)

errors = failure_df[failure_df["error"]].sort_values("confidence", ascending=False)
print(f"Total misclassifications (threshold={THRESHOLD}): {len(errors)} / {len(failure_df)} ({len(errors)/len(failure_df)*100:.2f}%)")

top_10 = errors.head(10).copy()
top_10["actual_label"] = le.inverse_transform(top_10["actual"])
top_10["failure_type"] = top_10.apply(
    lambda r: "FALSE POSITIVE (predicted Yes, actual No)" if r["actual"] == 0
    else "FALSE NEGATIVE (predicted No, actual Yes)",
    axis=1
)

for i, (idx, row) in enumerate(top_10.iterrows(), 1):
    print(f"\n  #{i} | Test index: {idx}")
    print(f"     Predicted probability: {row['predicted_prob']:.4f}")
    print(f"     Actual label:          {row['actual_label']}")
    print(f"     Type:                  {row['failure_type']}")

# Feature values of top 10 failures
top_10_indices = top_10.index.tolist()
top_10_features = X_test.iloc[top_10_indices].copy()
top_10_features.insert(0, "actual_label", top_10["actual_label"].values)
top_10_features.insert(1, "predicted_prob", top_10["predicted_prob"].values)
top_10_features.insert(2, "failure_type", top_10["failure_type"].values)
print("\n\nFeature values for top 10 failures (transposed):")
print(top_10_features.T.to_string())

# ── 8. Feature Importance ────────────────────────────────────────────────────
xgb_model = best_pipeline.named_steps["xgb"]
importances = xgb_model.feature_importances_
feature_names = X_train.columns
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 8))
feat_imp.head(20).plot(kind="barh", ax=ax, color="steelblue")
ax.set_xlabel("Feature Importance (Gain)")
ax.set_title("Top 20 Feature Importances — XGBoost")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig("fig_feature_importance.png", dpi=150)
plt.close()
print(f"\nSECTION 8: Feature Importance")
print("  Saved: fig_feature_importance.png")
print("  Top 20 features:")
for fname, fimp in feat_imp.head(20).items():
    print(f"    {fname}: {fimp:.4f}")

# ── 9. Comparison Summary ───────────────────────────────────────────────────
cv_results = pd.DataFrame(random_search.cv_results_)
best_idx = random_search.best_index_

mean_cv_f1 = cv_results.loc[best_idx, "mean_test_score"]
std_cv_f1 = cv_results.loc[best_idx, "std_test_score"]

test_accuracy = accuracy_score(y_test_enc, y_pred_custom)
test_f1_macro = f1_score(y_test_enc, y_pred_custom, average="macro")
test_f1_yes = f1_score(y_test_enc, y_pred_custom, average="binary")

print(f"\n{'='*80}")
print("SECTION 9: MODEL COMPARISON SUMMARY")
print(f"{'='*80}")
print(f"  Model:            XGBoost (tuned)")
print(f"  Best CV F1:       {mean_cv_f1:.4f} +/- {std_cv_f1:.4f}")
print(f"  Test Accuracy:    {test_accuracy:.4f}")
print(f"  Test F1 (Macro):  {test_f1_macro:.4f}")
print(f"  Test F1 (Yes):    {test_f1_yes:.4f}")
print(f"  ROC-AUC:          {roc_auc:.4f}")
print(f"  Threshold:        {THRESHOLD}")
print(f"  Best Params:      {random_search.best_params_}")
print(f"\n{'='*80}")
print("DONE — All results printed and figures saved.")
print(f"{'='*80}")
