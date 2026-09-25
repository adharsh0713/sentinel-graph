import sys
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "experiments"))

from preprocessing import load_split


def load_all():
    X_train, y_train, intensity_train = load_split("train")
    X_val, y_val, intensity_val = load_split("val")
    X_test, y_test, intensity_test = load_split("test")

    return (
        X_train,
        y_train,
        intensity_train,
        X_val,
        y_val,
        intensity_val,
        X_test,
        y_test,
        intensity_test,
    )


def evaluate(name, predictions, probabilities, y_test):
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    roc_auc = roc_auc_score(y_test, probabilities)
    pr_auc = average_precision_score(y_test, probabilities)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    ).ravel()

    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    print(f"\n{name}")
    print("-" * 50)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1       : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print(f"PR-AUC   : {pr_auc:.4f}")
    print(f"FPR      : {fpr:.4f}")
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")


def main():
    (
        X_train,
        y_train,
        intensity_train,
        X_val,
        y_val,
        intensity_val,
        X_test,
        y_test,
        intensity_test,
    ) = load_all()

    print("Original shapes:")
    print("Train:", X_train.shape)
    print("Val:  ", X_val.shape)
    print("Test: ", X_test.shape)

    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    print("\nTraining baseline models...")

    # ---------------------------------------------------------
    # 1. Majority baseline
    # ---------------------------------------------------------
    majority = DummyClassifier(strategy="most_frequent")
    majority.fit(X_train_flat, y_train)

    majority_pred = majority.predict(X_test_flat)
    majority_prob = majority.predict_proba(X_test_flat)[:, 1]

    evaluate(
        "Majority Baseline",
        majority_pred,
        majority_prob,
        y_test,
    )

    # ---------------------------------------------------------
    # 2. Persistence baseline
    # ---------------------------------------------------------
    #
    # The target represents the FUTURE state immediately after
    # the 10-state input sequence.
    #
    # We use the attack status of the final observed state as
    # the prediction for the future state.
    #
    # The final state itself is not stored as y in the sequence,
    # so reconstruct its attack status from the sequence's
    # final state's corresponding target relationship.
    #
    # For this dataset, use the final observed state's attack
    # intensity threshold from the sequence construction.
    # ---------------------------------------------------------

    # Recover the final observed attack state by using the
    # sequence target alignment:
    #
    # y[i] = attack state AFTER the 10 input states.
    #
    # Therefore persistence needs the attack label immediately
    # before that target. We reconstruct it from consecutive
    # sequence targets wherever possible.

    def persistence_predictions(split):
        files = sorted((PROJECT_ROOT / "data" / "splits" / split).glob("*.npz"))

        predictions = []
        probabilities = []
        targets = []

        for file in files:
            data = np.load(file)

            y = data["y_attack"]
            last_attack = data["last_attack"]

            # Exact persistence:
            # predict the next state will have the same attack
            # status as the final observed state in the input sequence.
            pred = last_attack.astype(int)

            predictions.extend(pred)
            probabilities.extend(pred.astype(float))
            targets.extend(y)

        return (
            np.array(predictions),
            np.array(probabilities),
            np.array(targets),
        )

    persistence_pred, persistence_prob, persistence_y = persistence_predictions(
        "test"
    )

    evaluate(
        "Persistence Baseline",
        persistence_pred,
        persistence_prob,
        persistence_y,
    )

    # ---------------------------------------------------------
    # 3. Logistic Regression
    # ---------------------------------------------------------
    logistic = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    logistic.fit(X_train_flat, y_train)

    logistic_pred = logistic.predict(X_test_flat)
    logistic_prob = logistic.predict_proba(X_test_flat)[:, 1]

    evaluate(
        "Logistic Regression",
        logistic_pred,
        logistic_prob,
        y_test,
    )



    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # 4. XGBoost
    # ---------------------------------------------------------

    print("XGBoost")

    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    xgb.fit(X_train_flat, y_train)

    # Validation is used for diagnostics only.
    # The classification threshold is fixed at 0.50 so that
    # the test evaluation does not depend on validation-set
    # class prevalence.
    val_prob = xgb.predict_proba(X_val_flat)[:, 1]
    val_pred = (val_prob >= 0.50).astype(int)
    val_f1 = f1_score(y_val, val_pred)

    # Final test evaluation with a fixed threshold.
    test_prob = xgb.predict_proba(X_test_flat)[:, 1]
    test_pred = (test_prob >= 0.50).astype(int)

    print("Classification threshold: 0.50")
    print(f"Validation F1:            {val_f1:.4f}")

    evaluate(
        "XGBoost",
        test_pred,
        test_prob,
        y_test,
    )



if __name__ == "__main__":
    main()
