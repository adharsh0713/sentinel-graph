import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "experiments"))

from preprocessing import load_split


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

SEED = 42
BATCH_SIZE = 64
HIDDEN_SIZE = 64
NUM_LAYERS = 1
DROPOUT = 0.0
LEARNING_RATE = 1e-3
EPOCHS = 30
PATIENCE = 5


# ---------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------

np.random.seed(SEED)
torch.manual_seed(SEED)


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size=24,
        hidden_size=64,
        num_layers=1,
        dropout=0.0,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output, _ = self.lstm(x)

        # Use the final observed state.
        final_hidden = output[:, -1, :]

        return self.classifier(final_hidden).squeeze(1)


# ---------------------------------------------------------
# Scaling
# ---------------------------------------------------------

def scale_sequences(X_train, X_val, X_test):
    """
    Standardize each of the 24 features using TRAIN only.
    """

    mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
    std = X_train.reshape(-1, X_train.shape[-1]).std(axis=0)

    std[std == 0] = 1.0

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    return X_train, X_val, X_test


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

def evaluate(name, predictions, probabilities, y_true):
    accuracy = accuracy_score(y_true, predictions)
    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )
    recall = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )
    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0,
    )
    roc_auc = roc_auc_score(y_true, probabilities)
    pr_auc = average_precision_score(y_true, probabilities)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
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


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------

@torch.no_grad()
def predict(model, loader, device):
    model.eval()

    probabilities = []

    for X_batch, _ in loader:
        X_batch = X_batch.to(device)

        logits = model(X_batch)
        probs = torch.sigmoid(logits)

        probabilities.extend(
            probs.cpu().numpy()
        )

    return np.array(probabilities)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    print("Device:", device)

    X_train, y_train, _ = load_split("train")
    X_val, y_val, _ = load_split("val")
    X_test, y_test, _ = load_split("test")

    print("\nOriginal shapes:")
    print("Train:", X_train.shape)
    print("Val:  ", X_val.shape)
    print("Test: ", X_test.shape)

    # -----------------------------------------------------
    # Train-only normalization
    # -----------------------------------------------------

    X_train, X_val, X_test = scale_sequences(
        X_train,
        X_val,
        X_test,
    )

    # -----------------------------------------------------
    # Tensor datasets
    # -----------------------------------------------------

    X_train_tensor = torch.tensor(
        X_train,
        dtype=torch.float32,
    )

    y_train_tensor = torch.tensor(
        y_train,
        dtype=torch.float32,
    )

    X_val_tensor = torch.tensor(
        X_val,
        dtype=torch.float32,
    )

    y_val_tensor = torch.tensor(
        y_val,
        dtype=torch.float32,
    )

    X_test_tensor = torch.tensor(
        X_test,
        dtype=torch.float32,
    )

    y_test_tensor = torch.tensor(
        y_test,
        dtype=torch.float32,
    )

    train_loader = DataLoader(
        TensorDataset(
            X_train_tensor,
            y_train_tensor,
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        TensorDataset(
            X_val_tensor,
            y_val_tensor,
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    test_loader = DataLoader(
        TensorDataset(
            X_test_tensor,
            y_test_tensor,
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    model = LSTMClassifier(
        input_size=24,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------

    print("\nTraining LSTM...")

    for epoch in range(1, EPOCHS + 1):

        model.train()

        train_losses = []

        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            logits = model(X_batch)

            loss = criterion(
                logits,
                y_batch,
            )

            loss.backward()

            optimizer.step()

            train_losses.append(
                loss.item()
            )

        # Validation loss
        model.eval()

        val_losses = []

        with torch.no_grad():

            for X_batch, y_batch in val_loader:

                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                logits = model(X_batch)

                loss = criterion(
                    logits,
                    y_batch,
                )

                val_losses.append(
                    loss.item()
                )

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

            patience_counter = 0

        else:

            patience_counter += 1

            if patience_counter >= PATIENCE:
                print("Early stopping.")
                break

    # -----------------------------------------------------
    # Restore best validation model
    # -----------------------------------------------------

    model.load_state_dict(best_state)

    # -----------------------------------------------------
    # Validation evaluation
    # -----------------------------------------------------

    val_prob = predict(
        model,
        val_loader,
        device,
    )

    val_pred = (
        val_prob >= 0.50
    ).astype(int)

    val_f1 = f1_score(
        y_val,
        val_pred,
        zero_division=0,
    )

    print(
        f"\nValidation F1: {val_f1:.4f}"
    )

    # -----------------------------------------------------
    # Final test evaluation
    # -----------------------------------------------------

    test_prob = predict(
        model,
        test_loader,
        device,
    )

    test_pred = (
        test_prob >= 0.50
    ).astype(int)

    print("\nProbability diagnostics:")
    print(
        f"Val  min={val_prob.min():.4f}, "
        f"max={val_prob.max():.4f}, "
        f"mean={val_prob.mean():.4f}, "
        f"median={np.median(val_prob):.4f}"
    )
    print(
        f"Test min={test_prob.min():.4f}, "
        f"max={test_prob.max():.4f}, "
        f"mean={test_prob.mean():.4f}, "
        f"median={np.median(test_prob):.4f}"
    )
    print(
        f"Test predictions at 0.50: "
        f"{test_pred.sum()} / {len(test_pred)}"
    )

    evaluate(
        "LSTM",
        test_pred,
        test_prob,
        y_test,
    )


if __name__ == "__main__":
    main()
