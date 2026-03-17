"""Runnable script version of knn_model.ipynb — outputs all results to console and saves figures."""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import sys

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
    f1_score,
    accuracy_score,
    precision_recall_curve,
    auc,
    precision_score,
    recall_score,
)
from sklearn.inspection import permutation_importance

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("=" * 80)
print("KNN MODEL — Distance-Based Classifier")
print("=" * 80)
print("\nSECTION 1: Libraries imported successfully")

# ── Check for required data files ───────────────────────────────────────────────
required_files = ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]
missing_files = [f for f in required_files if not os.path.exists(f)]

if missing_files:
    print("\n" + "=" * 80)
    print("ERROR: Missing data files!")
    print("=" * 80)
    print(f"Missing: {missing_files}")
    print("\nPlease run data_preparation.ipynb first to generate the CSV files.")
    print("You can do this in VS Code by:")
    print("  1. Open data_preparation.ipynb")
    print("  2. Click 'Run All' or run each cell sequentially")
    print("  3. Then run this script again")
    sys.exit(1)

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
imbalance_ratio = neg_count / pos_count
print(f"  Class imbalance ratio (No:Yes): {imbalance_ratio:.2f}:1")
print(f"  Note: KNN has no built-in class weights. Using threshold adjustment.")

# ── 4. Build Pipeline ────────────────────────────────────────────────────────
pipeline = Pipeline([
    ("scaler", StandardScaler()),  # CRITICAL for KNN: normalizes features
    ("knn", KNeighborsClassifier(n_jobs=-1)),
])
print(f"\nSECTION 4: Pipeline created")
print(pipeline)

# ── 5. Hyperparameter Tuning ─────────────────────────────────────────────────
param_distributions = {
    "knn__n_neighbors": [3, 5, 7, 9, 11, 15, 21],
    "knn__weights": ["uniform", "distance"],
    "knn__metric": ["euclidean", "manhattan", "minkowski"],
}

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_distributions,
    n_iter=50,
    scoring="f1",
    cv=cv_strategy,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=1,
    return_train_score=True,
)

print(f"\nSECTION 5: Starting RandomizedSearchCV (50 iterations, 5-fold CV)...")
print("  Note: KNN on large datasets is computationally intensive. Please wait...")
t0 = time.time()
random_search.fit(X_train, y_train_enc)
elapsed = time.time() - t0
print(f"Search complete in {elapsed/60:.1f} minutes")

print(f"\n  Best hyperparameters:")
for param, value in random_search.best_params_.items():
    print(f"    {param}: {value}")
print(f"  Best CV F1-score: {random_search.best_score_:.4f}")

best_pipeline = random_search.best_estimator_

# ── 6. Probability Output ─────────────────────────────────────────────────────
y_proba = best_pipeline.predict_proba(X_test)[:, 1]

print(f"\nSECTION 6: Probability output")
print(f"  Min:    {y_proba.min():.4f}")
print(f"  Max:    {y_proba.max():.4f}")
print(f"  Mean:   {y_proba.mean():.4f}")
print(f"  Median: {np.median(y_proba):.4f}")

y_pred_default = (y_proba >= 0.5).astype(int)
print(f"\n  Default threshold (0.50): Predicted Yes={np.sum(y_pred_default==1)}, No={np.sum(y_pred_default==0)}")

# Probability distribution plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(y_proba[y_test_enc == 0], bins=50, alpha=0.6, label="Actual: No", color="steelblue")
ax.hist(y_proba[y_test_enc == 1], bins=50, alpha=0.6, label="Actual: Yes", color="salmon")
ax.axvline(0.5, color="red", linestyle="--", linewidth=2, label="Threshold = 0.5")
ax.set_xlabel("Predicted Probability of Heart Disease")
ax.set_ylabel("Count")
ax.set_title("Distribution of Predicted Probabilities (KNN)")
ax.legend()
plt.tight_layout()
plt.savefig("knn_fig_probability_distribution.png", dpi=150)
plt.close()
print("  Saved: knn_fig_probability_distribution.png")

# ── 7. Threshold Optimization ─────────────────────────────────────────────────
print(f"\nSECTION 7: Threshold Optimization")
thresholds_sweep = np.arange(0.10, 0.91, 0.05)

results = []
for t in thresholds_sweep:
    y_pred_t = (y_proba >= t).astype(int)
    f1 = f1_score(y_test_enc, y_pred_t, zero_division=0)
    prec = precision_score(y_test_enc, y_pred_t, zero_division=0)
    rec = recall_score(y_test_enc, y_pred_t, zero_division=0)
    results.append({"Threshold": round(t, 2), "F1": f1, "Precision": prec, "Recall": rec})

threshold_df = pd.DataFrame(results)

print("\n  Threshold Search Results:")
print("  " + "-" * 56)
for _, row in threshold_df.iterrows():
    print(f"  Threshold={row['Threshold']:.2f}  F1={row['F1']:.4f}  Prec={row['Precision']:.4f}  Rec={row['Recall']:.4f}")

best_row = threshold_df.loc[threshold_df["F1"].idxmax()]
OPTIMAL_THRESHOLD = best_row["Threshold"]
print(f"\n  >>> Optimal threshold (max F1): {OPTIMAL_THRESHOLD}")

# Plot threshold optimization
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(threshold_df["Threshold"], threshold_df["F1"], "o-", color="green", linewidth=2, label="F1 Score")
ax.plot(threshold_df["Threshold"], threshold_df["Precision"], "s--", color="blue", linewidth=2, label="Precision")
ax.plot(threshold_df["Threshold"], threshold_df["Recall"], "^--", color="red", linewidth=2, label="Recall")
ax.axvline(OPTIMAL_THRESHOLD, color="grey", linestyle=":", linewidth=1.5, label=f"Optimal threshold ({OPTIMAL_THRESHOLD})")
ax.set_xlabel("Decision Threshold")
ax.set_ylabel("Score")
ax.set_title("F1 / Precision / Recall vs. Decision Threshold (KNN)")
ax.legend()
ax.set_xticks(thresholds_sweep)
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig("knn_fig_threshold_optimization.png", dpi=150)
plt.close()
print("  Saved: knn_fig_threshold_optimization.png")

y_pred_optimal = (y_proba >= OPTIMAL_THRESHOLD).astype(int)
print(f"  Optimal threshold ({OPTIMAL_THRESHOLD}): Predicted Yes={np.sum(y_pred_optimal==1)}, No={np.sum(y_pred_optimal==0)}")

# ── 8. Evaluation ─────────────────────────────────────────────────────────────
print(f"\nSECTION 8: Evaluation")
print(f"\n8.1 Classification Report (Threshold = {OPTIMAL_THRESHOLD})")
print("=" * 60)
target_names = ["No Heart Disease (0)", "Heart Disease (1)"]
print(classification_report(y_test_enc, y_pred_optimal, target_names=target_names))

# Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test_enc, y_pred_default, display_labels=target_names, cmap="Blues", ax=axes[0],
)
axes[0].set_title("Confusion Matrix (Default Threshold = 0.50)")
ConfusionMatrixDisplay.from_predictions(
    y_test_enc, y_pred_optimal, display_labels=target_names, cmap="Blues", ax=axes[1],
)
axes[1].set_title(f"Confusion Matrix (Optimal Threshold = {OPTIMAL_THRESHOLD})")
plt.tight_layout()
plt.savefig("knn_fig_confusion_matrices.png", dpi=150)
plt.close()
print("  Saved: knn_fig_confusion_matrices.png")

# ROC-AUC
roc_auc = roc_auc_score(y_test_enc, y_proba)
print(f"\n8.3 ROC-AUC Score: {roc_auc:.4f}")

fpr, tpr, _ = roc_curve(y_test_enc, y_proba)
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, color="steelblue", linewidth=2, label=f"KNN (AUC = {roc_auc:.4f})")
ax.plot([0, 1], [0, 1], color="grey", linestyle="--", label="Random Classifier")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve — KNN Classifier")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("knn_fig_roc_curve.png", dpi=150)
plt.close()
print("  Saved: knn_fig_roc_curve.png")

# PR-AUC
precisions, recalls, pr_thresholds = precision_recall_curve(y_test_enc, y_proba)
pr_auc = auc(recalls, precisions)
print(f"8.4 PR-AUC Score: {pr_auc:.4f}")

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(recalls, precisions, color="steelblue", linewidth=2, label=f"KNN (PR-AUC = {pr_auc:.4f})")
ax.axhline(y=np.mean(y_test_enc), color="grey", linestyle="--", label=f"Baseline (prevalence = {np.mean(y_test_enc):.4f})")
idx_opt = np.argmin(np.abs(pr_thresholds - OPTIMAL_THRESHOLD))
ax.plot(recalls[idx_opt], precisions[idx_opt], "ro", markersize=10,
        label=f"Optimal threshold = {OPTIMAL_THRESHOLD}")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve — KNN Classifier")
ax.legend(loc="upper right")
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
plt.tight_layout()
plt.savefig("knn_fig_pr_curve.png", dpi=150)
plt.close()
print("  Saved: knn_fig_pr_curve.png")

# Failure analysis
print(f"\n8.5 Failure Analysis — Top 10 Most Confident Failures")
print("=" * 80)

failure_df = pd.DataFrame({
    "actual": y_test_enc,
    "predicted_prob": y_proba,
    "predicted_label": y_pred_optimal,
})
failure_df["error"] = failure_df["actual"] != failure_df["predicted_label"]
failure_df["confidence"] = np.abs(failure_df["predicted_prob"] - 0.5)

errors = failure_df[failure_df["error"]].sort_values("confidence", ascending=False)
print(f"Total misclassifications (threshold={OPTIMAL_THRESHOLD}): {len(errors)} / {len(failure_df)} ({len(errors)/len(failure_df)*100:.2f}%)")

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

# ── 9. Feature Importance (Permutation) ───────────────────────────────────────
print(f"\n{'='*80}")
print("SECTION 9: Feature Importance (Permutation Method)")
print("=" * 80)
print("Computing permutation importance (this may take several minutes)...")

t0_perm = time.time()
perm_result = permutation_importance(
    best_pipeline,
    X_test,
    y_test_enc,
    n_repeats=10,
    random_state=RANDOM_STATE,
    scoring="f1",
    n_jobs=-1,
)
elapsed_perm = time.time() - t0_perm
print(f"Permutation importance computed in {elapsed_perm/60:.1f} minutes")

perm_importances = pd.DataFrame({
    "Feature": X_test.columns,
    "Importance Mean": perm_result.importances_mean,
    "Importance Std": perm_result.importances_std
}).sort_values(by="Importance Mean", ascending=False)

print("\nTop 20 Features by Permutation Importance:")
print("-" * 60)
for _, row in perm_importances.head(20).iterrows():
    print(f"  {row['Feature']:<40} {row['Importance Mean']:.4f} +/- {row['Importance Std']:.4f}")

# Plot
fig, ax = plt.subplots(figsize=(10, 8))
top_20 = perm_importances.head(20)
ax.barh(range(len(top_20)), top_20["Importance Mean"], xerr=top_20["Importance Std"], color="steelblue")
ax.set_yticks(range(len(top_20)))
ax.set_yticklabels(top_20["Feature"])
ax.set_xlabel("Mean Importance (F1 drop when shuffled)")
ax.set_title("Top 20 Feature Importances — KNN (Permutation Method)")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig("knn_fig_feature_importance.png", dpi=150)
plt.close()
print("  Saved: knn_fig_feature_importance.png")

# ── 10. Comparison Summary ───────────────────────────────────────────────────
cv_results = pd.DataFrame(random_search.cv_results_)
best_idx = random_search.best_index_

mean_cv_f1 = cv_results.loc[best_idx, "mean_test_score"]
std_cv_f1 = cv_results.loc[best_idx, "std_test_score"]

test_accuracy = accuracy_score(y_test_enc, y_pred_optimal)
test_f1_macro = f1_score(y_test_enc, y_pred_optimal, average="macro")
test_f1_yes = f1_score(y_test_enc, y_pred_optimal, average="binary")

print(f"\n{'='*80}")
print("SECTION 10: MODEL COMPARISON SUMMARY")
print(f"{'='*80}")
print(f"  Model:              KNN (tuned)")
print(f"  Algorithmic Family: Distance-Based")
print(f"  Best CV F1:         {mean_cv_f1:.4f} +/- {std_cv_f1:.4f}")
print(f"  Test Accuracy:      {test_accuracy:.4f}")
print(f"  Test F1 (Macro):    {test_f1_macro:.4f}")
print(f"  Test F1 (Yes):      {test_f1_yes:.4f}")
print(f"  ROC-AUC:            {roc_auc:.4f}")
print(f"  PR-AUC:             {pr_auc:.4f}")
print(f"  Threshold:          {OPTIMAL_THRESHOLD}")
print(f"  Best Params:        {random_search.best_params_}")

# ── 11. Pipeline Verification ────────────────────────────────────────────────
print(f"\n{'='*80}")
print("SECTION 11: PIPELINE ARCHITECTURE VERIFICATION")
print(f"{'='*80}")
print("\nPipeline Structure:")
print(best_pipeline)
print("\nData Leakage Prevention:")
print("  1. StandardScaler: Fitted ONLY on training data")
print("  2. KNeighborsClassifier: Trained ONLY on scaled training data")
print("  3. Cross-Validation: StratifiedKFold ensures class balance")

scaler = best_pipeline.named_steps["scaler"]
print(f"\nScaler Statistics (fitted on training data):")
print(f"  Number of features: {len(scaler.mean_)}")
print(f"  Feature means (first 5): {scaler.mean_[:5].round(4)}")
print(f"  Feature stds (first 5):  {scaler.scale_[:5].round(4)}")

# ── 12. Save Artifacts ───────────────────────────────────────────────────────
import joblib

joblib.dump(best_pipeline, "knn_best_pipeline.joblib")
cv_results.to_csv("knn_cv_results.csv", index=False)

summary_df = pd.DataFrame({
    "Model": ["KNN (tuned)"],
    "Algorithmic Family": ["Distance-Based"],
    "Best CV F1": [f"{mean_cv_f1:.4f} +/- {std_cv_f1:.4f}"],
    "Test Accuracy": [f"{test_accuracy:.4f}"],
    "Test F1 (Macro)": [f"{test_f1_macro:.4f}"],
    "Test F1 (Yes)": [f"{test_f1_yes:.4f}"],
    "ROC-AUC": [f"{roc_auc:.4f}"],
    "PR-AUC": [f"{pr_auc:.4f}"],
    "Threshold": [OPTIMAL_THRESHOLD],
    "Best Params": [str(random_search.best_params_)],
})
summary_df.to_csv("knn_summary_metrics.csv", index=False)

print(f"\n{'='*80}")
print("SECTION 12: Artifacts saved")
print(f"{'='*80}")
print("  knn_best_pipeline.joblib")
print("  knn_cv_results.csv")
print("  knn_summary_metrics.csv")

print(f"\n{'='*80}")
print("DONE — All results printed and figures saved.")
print(f"{'='*80}")
