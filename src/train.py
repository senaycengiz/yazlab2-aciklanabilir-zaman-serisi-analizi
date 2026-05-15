import torch
import torch.nn as nn
import torch.optim as optim

from config.config import (
    EPOCHS,
    LEARNING_RATE,
    EARLY_STOPPING_PATIENCE
)


def train_model(
    model,
    train_loader,
    validation_loader,
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    device="cpu",
    patience=EARLY_STOPPING_PATIENCE,
    return_history=False
):

    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    model.to(device)

    best_validation_loss = float("inf")
    patience_counter = 0

    history = {
        "epoch": [],
        "train_loss": [],
        "validation_loss": []
    }

    for epoch in range(epochs):

        model.train()

        train_loss = 0.0

        for inputs, labels in train_loader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(
                outputs.squeeze(),
                labels.float()
            )

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

        average_train_loss = train_loss / len(train_loader)

        model.eval()

        validation_loss = 0.0

        with torch.no_grad():

            for inputs, labels in validation_loader:

                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)

                loss = criterion(
                    outputs.squeeze(),
                    labels.float()
                )

                validation_loss += loss.item()

        average_validation_loss = validation_loss / len(validation_loader)

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(average_train_loss)
        history["validation_loss"].append(average_validation_loss)

        print(
            f"Epoch {epoch+1} | "
            f"Train Loss: {average_train_loss:.4f} | "
            f"Validation Loss: {average_validation_loss:.4f}"
        )

        if average_validation_loss < best_validation_loss:
            best_validation_loss = average_validation_loss
            patience_counter = 0
        else:
            patience_counter += 1

            print(
                f"Validation loss iyileşmedi. "
                f"Early stopping sayacı: {patience_counter}/{patience}"
            )

        if patience_counter >= patience:
            print(
                f"Early stopping çalıştı. "
                f"Eğitim {epoch+1}. epochta durduruldu."
            )
            break

    if return_history:
        return model, history

    return model
