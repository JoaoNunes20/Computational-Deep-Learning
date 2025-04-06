import os

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from models.baseline_classification_model import Baseline

path_to_root = os.getcwd()


# Helper function to sample stratified indices
def stratified_subset_indices(dataset, sample_size):
    targets = [label for _, label in dataset.imgs]
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=sample_size, random_state=42
    )
    for _, subset_idx in splitter.split(X=targets, y=targets):
        return subset_idx


def train_classifier():
    mlflow.set_experiment("cnn_training_experiment_take_2")
    epochs = 10
    learning_rate = 0.001
    batch_size = 32

    with mlflow.start_run():

        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("batch_size", batch_size)

        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
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

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # train the classifier's model
        model = Baseline()
        # Count class frequencies
        labels = sorted([label for _, label in train_dataset.imgs])
        class_counts = torch.bincount(torch.tensor(labels))
        real, fake = class_counts[0].item() / len(train_dataset.imgs), class_counts[
            1
        ].item() / len(train_dataset.imgs)
        class_weights = torch.tensor([1.0 / real, 1.0 / fake], dtype=torch.float)
        print(f"Class weights: {class_weights}")
        criterion = nn.CrossEntropyLoss(
            weight=class_weights
        )  # For binary classification
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
            y_true = []
            y_pred = []

            with torch.no_grad():
                for images, labels in val_loader_tqdm:
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    y_true.extend(labels.numpy())
                    y_pred.extend(predicted.numpy())
                    val_loader_tqdm.set_postfix(loss=loss.item())

            report = classification_report(
                y_true, y_pred, target_names=val_dataset.classes, output_dict=True
            )
            # Log key metrics to MLflow
            for label, metrics in report.items():
                if isinstance(metrics, dict):
                    for metric_name, value in metrics.items():
                        mlflow.log_metric(f"{label}_{metric_name}", value, step=epoch)
                else:
                    mlflow.log_metric(f"{label}", metrics, step=epoch)

            avg_val_loss = val_loss / len(val_loader)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
            print(report)
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
