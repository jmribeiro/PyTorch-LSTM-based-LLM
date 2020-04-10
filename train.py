#!/usr/bin/env python

import argparse
import sys

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader

from dataset import Ar2EnDataset
from model import EncoderDecoder, train_batch, evaluate


def plot(epochs, plottable, ylabel, title, name):
    plt.clf()
    plt.xlabel('Epoch')
    if isinstance(plottable, tuple):
        assert isinstance(ylabel, tuple) and len(plottable) == len(ylabel)
        for i in range(len(plottable)):
            plt.plot(epochs, plottable[i], label=ylabel[i])
        plt.legend()
        plt.ylabel("Accuracy")
    else:
        plt.ylabel(ylabel)
        plt.plot(epochs, plottable)
    plt.title(title)
    plt.savefig('%s.pdf' % name, bbox_inches='tight')
    plt.close()


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('-data', help="Path to ar2en dataset.",                             default='./ar2en_dataset')

    parser.add_argument('-embeddings_size', type=int,                                       default=300)
    parser.add_argument('-layers',          type=int,                                       default=2)
    parser.add_argument('-hidden_sizes',    type=int,                                       default=300)
    parser.add_argument('-dropout',         type=float,                                     default=0.1)
    parser.add_argument('-epochs',          type=int,                                       default=20)
    parser.add_argument('-optimizer',       choices=['sgd', 'adam'],                        default='adam')
    parser.add_argument('-learning_rate',   type=float,                                     default=0.001)
    parser.add_argument('-l2_decay',        type=float,                                     default=0.0)
    parser.add_argument('-batch_size',      type=int,                                       default=64)

    parser.add_argument('-cuda', action='store_true',
                        help='Whether or not to use cuda for parallelization (if devices available)')

    parser.add_argument('-name', type=str, required=False, default=None,
                        help="Filename for the plot")

    parser.add_argument('-quiet', action='store_true',
                        help='No execution output.')

    parser.add_argument('-tqdm', action='store_true',
                        help='Whether or not to use TQDM progress bar in training.')

    parser.add_argument('-display_vocabularies', action="store_true",
                        help="Only display the vocabularies (no further execution).")

    parser.add_argument('-reverse_source_string', action="store_true",
                        help="Whether or not to reverse the source arabic string.")

    parser.add_argument('-bidirectional', action="store_true",
                        help="Whether or not to use a bidirectional encoder LSTM.")

    parser.add_argument('-attention', type=str, choices=["dot", "general"], required=False, default=None,
                        help="Attention mechanism in the decoder.")

    opt = parser.parse_args()

    # ############# #
    # 1 - Load Data #
    # ############# #

    dataset = Ar2EnDataset(opt.data, opt.reverse_source_string)

    if opt.display_vocabularies:
        sys.exit(0)

    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)
    X_dev, y_dev = dataset.X_dev, dataset.y_dev
    X_test, y_test = dataset.X_test, dataset.y_test

    # ################ #
    # 2 - Create Model #
    # ################ #

    device = torch.device("cuda:0" if torch.cuda.is_available() and opt.cuda else "cpu")
    if not opt.quiet: print(f"Using device '{device}'", flush=True)

    model = EncoderDecoder(dataset.n_inputs, dataset.n_outputs,
                           opt.embeddings_size, opt.attention, opt.bidirectional,
                           opt.hidden_sizes, opt.layers, opt.dropout,
                           dataset.arabic_vocabulary, dataset.english_vocabulary, device)

    # ############# #
    # 3 - Optimizer #
    # ############# #

    optimizer = {
        "adam": torch.optim.Adam,
        "sgd": torch.optim.SGD
    }[opt.optimizer](
        model.parameters(),
        lr=opt.learning_rate,
        weight_decay=opt.l2_decay
    )
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.english_vocabulary["$PAD"])

    # ###################### #
    # 4 - Train and Evaluate #
    # ###################### #

    epochs = torch.arange(1, opt.epochs + 1)
    train_mean_losses = []
    val_word_acc = []
    val_char_acc = []
    train_losses = []

    for epoch in epochs:

        if not opt.quiet: print('\nTraining epoch {}'.format(epoch), flush=True)

        if opt.tqdm:
            from tqdm import tqdm
            dataloader = tqdm(dataloader)

        for X_batch, y_batch in dataloader:
            loss = train_batch(X_batch, y_batch, model, optimizer, criterion)
            train_losses.append(loss)

        mean_loss = torch.tensor(train_losses).mean().item()
        word_acc, char_acc = evaluate(model, X_dev, y_dev)

        train_mean_losses.append(mean_loss)
        val_word_acc.append(word_acc)
        val_char_acc.append(char_acc)

        if not opt.quiet:
            print('Training loss: %.4f' % mean_loss, flush=True)
            print('Valid word acc: %.4f' % val_word_acc[-1], flush=True)
            print('Valid char acc: %.4f' % val_char_acc[-1], flush=True)

    final_test_accuracy_words, final_test_accuracy_chars = evaluate(model, X_test, y_test)
    if not opt.quiet:
        print('\nFinal Test Word Acc: %.4f' % final_test_accuracy_words, flush=True)
        print('Final Test Char Acc: %.4f' % final_test_accuracy_chars, flush=True)

    # ######## #
    # 5 - Plot #
    # ######## #

    name = opt.name if opt.name is not None else "encoder_decoder"
    plot(epochs, train_mean_losses, ylabel='Loss',         name=name+"_loss",   title="Training Loss")
    plot(epochs, val_word_acc,      ylabel='Word Val Acc', name=name+"_acc",    title=f"Word Validation Accuracy\n(Final Word Test Accuracy: {round(final_test_accuracy_words,3)})")

    return final_test_accuracy_words


if __name__ == '__main__':
    main()
