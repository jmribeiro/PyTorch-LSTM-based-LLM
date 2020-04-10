# PyTorch-Sequence-to-Sequence
Arabic to English Translation using Encoder-Decoder Sequence-to-Sequence Model

#### To run

1) Install PyTorch from https://pytorch.org/get-started/locally/ (default below) and other dependencies

        $ pip install torch torchvision matplotlib

2) [Optional] Install latest PyTorch-NLP if using attention

        $ pip install git+https://github.com/PetrochukM/PyTorch-NLP

3) Run train.py

        $ python train.py -h
        usage: train.py [-h] [-data DATA] [-embeddings_size EMBEDDINGS_SIZE]
                    [-layers LAYERS] [-hidden_sizes HIDDEN_SIZES]
                    [-dropout DROPOUT] [-epochs EPOCHS] [-optimizer {sgd,adam}]
                    [-learning_rate LEARNING_RATE] [-l2_decay L2_DECAY]
                    [-batch_size BATCH_SIZE] [-cuda] [-name NAME] [-quiet] [-tqdm]
                    [-display_vocabularies] [-reverse_source_string]
                    [-bidirectional] [-attention {dot,general}]

##### train.py
> Main script, loads the dataset and trains the model. Plots both the training losses and validation accuracies and returns the final test accuracy.

##### model.py
> Encoder-Decoder model implemented in PyTorch. Also contains the training and evaluation utilities.

##### dataset.py
> Ar2EnDataset and simple Vocabulary class.

##### Optional Arguments
    $ python train.py

    -h, --help            show this help message and exit
    -data DATA            Path to ar2en dataset.

    -embeddings_size EMBEDDINGS_SIZE
    -layers          LAYERS
    -hidden_sizes    HIDDEN_SIZES
    -dropout         DROPOUT
    -epochs          EPOCHS
    -optimizer       {sgd,adam}
    -learning_rate   LEARNING_RATE
    -l2_decay        L2_DECAY
    -batch_size      BATCH_SIZE
    -attention       {dot,general} -> Requires installing the latest version of PyTorch-NLP from git

    -cuda                     Whether or not to use cuda for parallelization (if devices available)
    -name NAME                Filename for the plot 
    -quiet                    No execution output.
    -tqdm                     Whether or not to use TQDM progress bar in training.
    -display_vocabularies     Only display the vocabularies (no further execution).
    -reverse_source_string    Whether or not to reverse the source arabic string.
    -bidirectional            Whether or not to use a bidirectional LSTM encoder.
