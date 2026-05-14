import torch
import torch.nn as nn
import torch.optim as optim


def train_model(
    model,
    train_loader,
    validation_loader,
    epochs=10,
    learning_rate=0.001,
    device="cpu"
):

    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    model.to(device)

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

        average_validation_loss = (
            validation_loss / len(validation_loader)
        )

        print(
            f"Epoch {epoch+1} | "
            f"Train Loss: {average_train_loss:.4f} | "
            f"Validation Loss: {average_validation_loss:.4f}"
        )

    return model