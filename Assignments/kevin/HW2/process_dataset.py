"""Read, split and save the kaggle dataset for our model"""

import csv
import os
import sys
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# def load_dataset(path_csv):
#     """
#        TODO: DONE
#        Loads dataset into memory from csv file
#     """
#     dataset = pd.read_csv(path_csv,encoding='windows-1252',dtype=str)
#     return dataset


# def save_dataset(dataset, save_dir):
#     """ 
#        TODO: DONE
#        Writes sentences.txt and labels.txt files in save_dir from dataset
#        See data/example for the format

#        Args:
#         dataset: ([(["a", "cat"], ["O", "O"]), ...])
#         save_dir: (string)
#     """
#     # TODO: DONE
#     # Create directory if it doesn't exist
#     Path(save_dir).mkdir(parents=True, exist_ok=True)

#     # TODO: DONE
#     # Export the dataset
#     df = dataset
#     df.fillna(method='ffill', inplace=True)
#     df.loc['Sentence #'] = df['Sentence #'].apply(str)
#     df[['sentence', 'number']] = df['Sentence #'].str.split(':', expand=True)
#     df.loc['number'] = pd.to_numeric(df['number'], errors='coerce')
#     groups = df.groupby('number')

#     sentences = groups['Word'].apply(' '.join)
#     sentences.to_csv(f'{save_dir}/sentences.txt', index=False, header=False)

#     labels = groups['Tag'].apply(' '.join)
#     labels.to_csv(f'{save_dir}/labels.txt', index=False, header=False)
#     print(" Done ")

def load_dataset(path_csv):
    """
       TODO: DONE
       Loads dataset into memory from csv file
    """
    with open(path_csv, encoding="windows-1252") as f:
        csv_file = csv.reader(f, delimiter=',')
        dataset = []
        sentences = []
        labels = []

        # Iterate through each row
        for i, row in enumerate(csv_file):
            if i == 0: 
                continue
            sentence, word, pos, tag = row
            # Add to dataset
            if len(sentence) > 0 and len(sentences) > 0:
                dataset.append((sentences, labels))
                sentences = []
                labels = []
            # Parse
            word = str(word)
            tag = str(tag)
            sentences.append(word)
            labels.append(tag)
            
    return dataset


def save_dataset(dataset, save_dir):
    """ 
       TODO: DONE
       Writes sentences.txt and labels.txt files in save_dir from dataset
       See data/example for the format

       Args:
        dataset: ([(["a", "cat"], ["O", "O"]), ...])
        save_dir: (string)
    """
    # TODO: DONE
    # Create directory if it doesn't exist
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # TODO: DONE
    # Export the dataset
    with open(f'{save_dir}/sentences.txt', 'w') as sfile:
        with open(f'{save_dir}/labels.txt', 'w') as lfile:
            for sentences, labels in dataset:
                sfile.write("{}\n".format(" ".join(sentences)))
                lfile.write("{}\n".format(" ".join(labels)))
    print("saving")


if __name__ == "__main__":
    # Check that the dataset exists (you need to make sure you haven't downloaded the `ner.csv`)
    path_dataset = 'data/new_data/dataset.csv'

    # Load the dataset into memory
    print("Loading dataset into memory...")
    dataset = load_dataset(path_dataset)
    print(" Done ")

    # Split the dataset into train, val and split (dummy split with no shuffle)
    train_ind = int(len(dataset)*0.6)
    test_ind = int(len(dataset)*0.8)
    train_dataset = dataset[:train_ind]       # TODO: DONE
    val_dataset = dataset[train_ind:test_ind] # TODO: DONE
    test_dataset = dataset[test_ind:]         # TODO: DONE

    # Save the datasets to files
    save_dataset(train_dataset, 'data/new_data/train')
    save_dataset(val_dataset, 'data/new_data/val')
    save_dataset(test_dataset, 'data/new_data/test')