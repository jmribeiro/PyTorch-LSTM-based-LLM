import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import Dataset

import numpy as np


class Vocabulary:

    """ Encapsulates a vocabulary of characters for a given language """

    def __init__(self, language):
        self.language = language
        self.characters = ["$START", "$STOP", "$UNK", "$PAD"]

    def add_character(self, character):
        """ Adds a character to the vocabulary if not in it already. """
        if character not in self.characters:
            self.characters.append(character)

    def __len__(self):
        """
        Returns the total number of characters in the vocabulary.
        """
        return len(self.characters)

    def __repr__(self):
        """
        Returns the words in the vocabulary (including special symbols).
        """
        return "| " + " | ".join(self.characters) + " |"

    def __getitem__(self, item):
        if isinstance(item, str):
            try:
                return self.characters.index(item)
            except ValueError:
                print(f"WARN: Character '{item}' does not belong to {self.language} vocabulary, returning $UNK.")
                return self.characters.index("$UNK")

        elif isinstance(item, int):
            return self.characters[item]
        else:
            raise IndexError("Invalid item for indexation. Pass only int or str.")


class Ar2EnDataset(Dataset):

    """ Encapsulates the ar2en dataset """

    def __init__(self, path, reverse_source_string):

        # ############# #
        # Load from txt #
        # ############# #

        training = np.loadtxt(f"{path}/ar2en-train.txt",    dtype=np.str)
        validation = np.loadtxt(f"{path}/ar2en-eval.txt",   dtype=np.str)
        test = np.loadtxt(f"{path}/ar2en-test.txt",         dtype=np.str)

        # ################### #
        # Create Vocabularies #
        # ################### #

        self.english_vocabulary = Vocabulary("english")
        self.arabic_vocabulary = Vocabulary("arabic")

        for ar_word, en_word in list(training) + list(validation) + list(test):
            [self.arabic_vocabulary.add_character(ar_char) for ar_char in ar_word]
            [self.english_vocabulary.add_character(en_char) for en_char in en_word]

        self.n_inputs = len(self.arabic_vocabulary)
        self.n_outputs = len(self.english_vocabulary)
        print(f"Arabic vocabulary ({self.n_inputs} unique characters):\n\t {self.arabic_vocabulary}\n", flush=True)
        print(f"English vocabulary ({self.n_outputs} unique characters):\n\t {self.english_vocabulary}\n", flush=True)

        # ######## #
        # Features #
        # ######## #

        self.X_train, self.y_train = self.to_index_features(training, reverse_source_string)
        self.X_dev, self.y_dev = self.to_index_features(validation, reverse_source_string)
        self.X_test, self.y_test = self.to_index_features(test, reverse_source_string)

    def to_index_features(self, data, reverse_source_string):

        X, y = [], []

        for ar_word, en_word in data:

            source_seq = [self.arabic_vocabulary[ar_char] for ar_char in ar_word]
            source_seq.reverse() if reverse_source_string else source_seq
            xseq = [self.arabic_vocabulary["$START"]] + source_seq + [self.arabic_vocabulary["$STOP"]]
            xseq = torch.tensor(xseq, dtype=torch.float)

            target_seq = [self.english_vocabulary[en_char] for en_char in en_word]
            yseq = [self.english_vocabulary["$START"]] + target_seq + [self.english_vocabulary["$STOP"]]
            yseq = torch.tensor(yseq, dtype=torch.float)

            X.append(xseq)
            y.append(yseq)

        X = pad_sequence(X, batch_first=True, padding_value=self.arabic_vocabulary["$PAD"])
        y = pad_sequence(y, batch_first=True, padding_value=self.english_vocabulary["$PAD"])

        X = X.long()
        y = y.long()

        return X, y

    def __len__(self):
        return len(self.X_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]
