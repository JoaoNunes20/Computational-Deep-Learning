"""the goal of this script is to organize the dataset into a train/val split"""

import argparse
import os
import shutil

import pandas as pd
from sklearn.model_selection import train_test_split

# Define paths
path_to_root = __file__.split("scripts", maxsplit=1)[0]

source_root = os.path.join(path_to_root, "images")
target_root = os.path.join(path_to_root, "data")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Organize dataset into train/val split"
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.8,
        help="Ratio of images to be used for training",
    )
    return parser.parse_args()


# Function to move images to structured directories
def move_images(image_list, label_list, split):
    for img_path, label in zip(image_list, label_list):
        class_name = "fake" if label == 1 else "real"
        dest_dir = os.path.join(target_root, split, class_name)
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy(img_path, os.path.join(dest_dir, os.path.basename(img_path)))


def main(args):

    # clear data in target_root
    shutil.rmtree(os.path.join(target_root, "train"), ignore_errors=True)
    shutil.rmtree(os.path.join(target_root, "val"), ignore_errors=True)

    split_ratio = args.ratio

    dataset_df = pd.read_csv(os.path.join(path_to_root, "data/dataset.csv"))

    print(f"Total images found: {len(dataset_df)}")
    print(dataset_df["image_type"].value_counts())

    # perform stratified split on image_type
    train_images, val_images, train_labels, val_labels = train_test_split(
        dataset_df["image_path"],
        dataset_df["label"],
        train_size=split_ratio,
        random_state=42,
        stratify=dataset_df["label"],
    )

    # print split details
    print(f"Training images: {len(train_images)}")
    print(f"Training labels: {train_labels.value_counts()}")

    print(f"Validation images: {len(val_images)}")
    print(f"Validation labels: {val_labels.value_counts()}")

    # Move images to train and val folders
    move_images(train_images, train_labels, "train")
    move_images(val_images, val_labels, "val")

    print("Dataset organized successfully!")


if __name__ == "__main__":
    parsed_args = parse_args()
    main(parsed_args)
