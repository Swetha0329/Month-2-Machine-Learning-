"""Retrain MailGuard AI using the corrections collected in feedback.csv.

Usage (stop the Flask app first, then run in the project folder):
    python retrain.py

What it does:
1. Splits spam.csv into 80% train / 20% test (the same split the dashboard uses).
2. Trains the model design twice on the train split only: without and with your feedback.
3. Compares both on the unseen test messages.
4. Only if the feedback version is not worse, trains the final model on ALL data plus
   your feedback, backs up the old model, and saves the new one.
"""

import csv
import shutil
import sys
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "svm_spam_classifier.joblib"
DATASET_PATH = BASE_DIR / "spam.csv"
FEEDBACK_PATH = BASE_DIR / "feedback.csv"

F1_TOLERANCE = 0.001  # the new model may be at most 0.1 point worse on F1


def load_dataset():
    frame = pd.read_csv(DATASET_PATH, encoding="latin-1")[["v1", "v2"]]
    frame.columns = ["label", "message"]
    frame["label"] = frame["label"].astype(str).str.strip().str.lower()
    frame = frame.dropna()
    frame = frame[frame["label"].isin(["ham", "spam"])]
    return frame.drop_duplicates(subset="message")


def load_feedback():
    if not FEEDBACK_PATH.exists():
        return pd.DataFrame(columns=["message", "label"])

    rows = []
    with open(FEEDBACK_PATH, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            message = (row.get("message") or "").strip()
            label = (row.get("true_label") or "").strip().lower()
            if message and label in ("ham", "spam"):
                rows.append({"message": message, "label": label})

    frame = pd.DataFrame(rows, columns=["message", "label"])
    return frame.drop_duplicates(subset="message", keep="last")  # newest feedback wins


def evaluate(model, X_test, y_test):
    actual = (y_test == "spam").astype(int).to_numpy()
    predicted = (model.predict(X_test) == "spam").astype(int)

    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual, predicted, average="binary", zero_division=0
    )

    return {
        "accuracy": (tn + tp) / len(actual), "precision": precision,
        "recall": recall, "f1": f1, "fp": int(fp), "fn": int(fn),
    }


def show(name, scores):
    print(f"{name:<26} accuracy {scores['accuracy'] * 100:6.2f}%   precision {scores['precision'] * 100:6.2f}%   "
          f"recall {scores['recall'] * 100:6.2f}%   F1 {scores['f1'] * 100:6.2f}%   "
          f"false positives {scores['fp']}   false negatives {scores['fn']}")


def main():
    if not MODEL_PATH.exists() or not DATASET_PATH.exists():
        sys.exit("svm_spam_classifier.joblib and spam.csv must be in the same folder as retrain.py.")

    feedback = load_feedback()
    if feedback.empty:
        sys.exit("No usable feedback found in feedback.csv. Use the 👍 / 👎 buttons in the app first.")

    current = joblib.load(MODEL_PATH)
    data = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        data["message"], data["label"], test_size=0.2, random_state=42, stratify=data["label"]
    )

    # Feedback that is also in the test set is ignored so the comparison stays fair.
    test_texts = set(X_test)
    usable = feedback[~feedback["message"].isin(test_texts)]
    skipped = len(feedback) - len(usable)

    print(f"Feedback rows: {len(feedback)} ({skipped} skipped because they are in the test set)")
    print(f"Spam in feedback: {(usable['label'] == 'spam').sum()}, ham: {(usable['label'] == 'ham').sum()}\n")

    baseline = clone(current).fit(X_train, y_train)
    with_feedback = clone(current).fit(
        pd.concat([X_train, usable["message"]]),
        pd.concat([y_train, usable["label"]]),
    )

    base_scores = evaluate(baseline, X_test, y_test)
    new_scores = evaluate(with_feedback, X_test, y_test)

    print("Evaluation on unseen test messages:")
    show("Without feedback", base_scores)
    show("With feedback", new_scores)
    print()

    not_worse = (new_scores["f1"] >= base_scores["f1"] - F1_TOLERANCE
                 and new_scores["fp"] <= base_scores["fp"] + 1)

    if not not_worse:
        print("The feedback version is worse, so your current model was NOT changed.")
        print("Collect more feedback and try again.")
        return

    final = clone(current).fit(
        pd.concat([data["message"], usable["message"]]),
        pd.concat([data["label"], usable["label"]]),
    )

    backup = MODEL_PATH.with_name(f"svm_spam_classifier_backup_{datetime.now():%Y%m%d_%H%M%S}.joblib")
    shutil.copy2(MODEL_PATH, backup)
    joblib.dump(final, MODEL_PATH)

    print(f"Saved the retrained model to {MODEL_PATH.name}.")
    print(f"The old model was backed up as {backup.name}.")
    print("Start the app again to use the new model.")


if __name__ == "__main__":
    main()