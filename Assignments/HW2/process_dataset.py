"""Read, split and save the kaggle dataset for our model"""

import csv
import os
import sys


def load_dataset(path_csv):
    """(TODO) Loads dataset into memory from csv file"""

    return dataset


def save_dataset(dataset, save_dir):
    """ (TODO) Writes sentences.txt and labels.txt files in save_dir from dataset
    See data/example for the format

    Args:
        dataset: ([(["a", "cat"], ["O", "O"]), ...])
        save_dir: (string)
    """
    # (TODO) Create directory if it doesn't exist

    # (TODO) Export the dataset

    print(" Done ")


if __name__ == "__main__":
    # Check that the dataset exists (you need to make sure you haven't downloaded the `ner.csv`)
    path_dataset = 'data/new_data/dataset.csv'

    # Load the dataset into memory
    print("Loading dataset into memory...")
    dataset = load_dataset(path_dataset)
    print(" Done ")

    # Split the dataset into train, val and split (dummy split with no shuffle)
    train_dataset = # TODO
    val_dataset = # TODO
    test_dataset = # TODO

    # Save the datasets to files
    save_dataset(train_dataset, 'data/new_data/train')
    save_dataset(val_dataset, 'data/new_data/val')
    save_dataset(test_dataset, 'data/new_data/test')