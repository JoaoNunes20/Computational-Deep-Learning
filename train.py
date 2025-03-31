import os

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from models.baseline_classification_model import Baseline

path_to_root = os.getcwd()


def train_classifier():
    mlflow.set_experiment("cnn_training_experiment")
    epochs = 10
    learning_rate = 0.001
    batch_size = 32

    with mlflow.start_run():

        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("batch_size", batch_size)

        transform = transforms.Compose(
            [
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        # Load the dataset
        train_dataset = datasets.ImageFolder(
            root=os.path.join(path_to_root, "data/train"),
            transform=transform,
        )
        val_dataset = datasets.ImageFolder(
            root=os.path.join(path_to_root, "data/val"),
            transform=transform,
        )

        # Randomly select indices for a small subset
        train_sample_size = len(train_dataset) // 2
        val_sample_size = len(val_dataset) // 2

        # Get total number of samples in the dataset
        train_total = len(train_dataset)
        val_total = len(val_dataset)

        # Randomly select indices for train and validation datasets
        train_indices = torch.randint(
            0, train_total, (train_sample_size,)
        ).tolist()  # Random indices for train
        val_indices = torch.randint(
            0, val_total, (val_sample_size,)
        ).tolist()  # Random indices for val

        # Create subsets based on the randomly selected indices
        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # train the classifier's model
        model = Baseline()
        criterion = nn.CrossEntropyLoss()  # For binary classification
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            train_loader_tqdm = tqdm(
                train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training]", leave=False
            )
            running_loss = 0.0
            model.train()
            for images, labels in train_loader_tqdm:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                train_loader_tqdm.set_postfix(loss=loss.item())

            avg_train_loss = running_loss / len(train_loader)
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)

            print(
                f"Epoch {epoch+1}/{epochs}, Avg Training Loss: {running_loss / len(train_loader)}"
            )

            model.eval()
            val_loader_tqdm = tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{epochs} [Validation]", leave=False
            )

            val_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for images, labels in val_loader_tqdm:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    val_loader_tqdm.set_postfix(loss=loss.item())

            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = 100 * correct / total

            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)

            print(f"Validation Loss: {val_loss / len(val_loader)}")
            print(f"Validation Accuracy: {100 * correct / total}%")
            print("=====================================")

        model_path = "model.pth"
        torch.save(model.state_dict(), model_path)
        mlflow.pytorch.log_model(model, "model")

        print("Training complete. Model saved and logged to MLflow.")


def train_generator():
    raise NotImplementedError("Generator training not implemented yet")


def main():
    print("Training classifier...")
    train_classifier()


if __name__ == "__main__":
    main()
