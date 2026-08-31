import torch

def train_model(model, X_train, y_train, criterion,optimizer, epochs=200):
    losses = []

    for epoch in range(epochs):
        predctions = model(X_train)
        loss = criterion(predctions, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return losses


def train_with_early_stopping(
        model, 
        train_loader, 
        val_loader, 
        criterion, 
        optimizer,
        device ,
        epochs=20,
        patience=3
):

    print("Device being used for training:", device)
    print("Model device:" , next(model.parameters()).device)
    train_losses = []
    val_accuracies = []


    best_val_accuracy = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):

        # Training 

        model.train()
        running_loss = 0.0

        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()

            outputs = model(X)
            loss = criterion(outputs, y)

            loss.backward()
            optimizer.step()


            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(device)
                y = y.to(device)

                outputs = model(X)
                predictions = outputs.argmax(dim=1)

                total += y.size(0)
                correct += (predictions == y).sum().item()

        val_accuracy = correct / total
        val_accuracies.append(val_accuracy)

        print(
            f"Epoch [{epoch + 1}/{epochs}], "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Validation Accuracy: {val_accuracy:.4f}"
        )

        # Early stopping
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy


            #Clone tensors so this is a real snapshot

            best_state = {
                k: v.clone()
                for k, v in model.state_dict().items()
            }

            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    #Restore best model

    model.load_state_dict(best_state)

    return train_losses, val_accuracies

    