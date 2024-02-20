import pandas as pd
import numpy as np
import json

class VocabDict:
    """
    Class for creating a vocabulary dictionary from a given dataset.
    """

    def __init__(self, root_dir, label='text', out_dir=''):
        """
        Initialize the VocabDict instance.

        Args:
            root_dir (str): The root directory containing 'train.csv' and 'test.csv'.
            label (str, optional): The column name containing text data in the CSV files. Defaults to 'text'.
            out_dir (str, optional): Output directory for the generated vocabulary JSON file.
        """
        self.train_csv = root_dir + '/train.csv'
        self.test_csv = root_dir + '/test.csv'
        self.label = label
        self.train_values = pd.read_csv(self.train_csv)[self.label].values
        self.test_values = pd.read_csv(self.test_csv)[self.label].values
        self.all_text = np.concatenate([self.train_values, self.test_values])
        self.out_dir = out_dir
        
    def process(self):
        """
        Processes the text data and generates a vocabulary dictionary.

        Returns:
            dict: Vocabulary dictionary with characters as keys and indices as values.
        """
        words = []
        for line in self.all_text:
            words += line.split(' ')
        words = list(set(words))
        chars = []
        for word in words:
            chars += [*word]

        vocab_dict = {v: k for k, v in enumerate(chars)}
        vocab_dict["|"] = len(vocab_dict) 
        vocab_dict["[UNK]"] = len(vocab_dict)
        vocab_dict["[PAD]"] = len(vocab_dict)

        with open(self.out_dir + '/vocab.json', 'w') as vocab_file:
            json.dump(vocab_dict, vocab_file, ensure_ascii=False, indent=4, sort_keys=True)
        
        return vocab_dict
