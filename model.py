import torch
from torch import nn


class EncoderDecoder(nn.Module):

    def __init__(self, n_inputs, n_outputs,
                 embeddings_size, attention, bidirectional,
                 hidden_sizes, layers, dropout, input_vocab, output_vocab, device):

        super(EncoderDecoder, self).__init__()

        self.input_vocab = input_vocab
        self.output_vocab = output_vocab

        self.padding_value = output_vocab["$PAD"]
        self.bidirectional = bidirectional

        self.encoder_embeddings = nn.Embedding(n_inputs, embeddings_size, padding_idx=input_vocab["$PAD"])
        self.encoder = nn.LSTM(embeddings_size, hidden_sizes, layers, dropout=dropout, batch_first=True, bidirectional=bidirectional)

        self.decoder_embeddings = nn.Embedding(n_inputs, embeddings_size, padding_idx=output_vocab["$PAD"])
        if bidirectional: hidden_sizes *= 2
        self.decoder = nn.LSTM(embeddings_size, hidden_sizes, layers, dropout=dropout, batch_first=True)

        if attention is not None:
            from torchnlp.nn import Attention
            self.attention = Attention(hidden_sizes, attention_type=attention)

        self.hidden_to_output = nn.Linear(hidden_sizes, n_outputs)

        self.device = device
        self.to(device)

    # ############## #
    # Main Interface #
    # ############## #

    def forward(self, X, y):
        """ Uses full teacher forcing when training (i.e., uses y to decode instead re-feeding the generated tokens)"""
        encodings, hidden = self.encode(X)
        y = y[:, :-1]   # Start at $START, but don't decode the last token (since it has no followup)
        scores, last_hidden = self.decode(y, encodings, hidden)
        return scores

    def predict(self, X, max_length):
        """ Vectorized computation without teaching forcing """
        encodings, hidden = self.encode(X)
        Z = torch.zeros(X.shape[0], 1, len(self.output_vocab)).to(self.device)
        tokens = torch.full((X.shape[0], 1), fill_value=self.output_vocab["$START"], dtype=torch.long).to(self.device)
        for i in range(max_length):
            scores, hidden = self.decode(tokens, encodings, hidden)
            tokens = scores.argmax(dim=2)
            Z = torch.cat((Z, scores), 1)
        return Z[:, 1:]  # Zeroed scores for $START are ignored (from Z = torch.zeros before concat)

    # ################# #
    # Auxiliary Methods #
    # ################# #

    def encode(self, tokens):
        embeddings = self.encoder_embeddings(tokens)
        encodings, hidden = self.encoder(embeddings)
        if self.bidirectional: hidden = self.bidirectional_reshape(hidden)
        return encodings, hidden

    def decode(self, tokens, encodings, previous_hidden):
        embeddings = self.decoder_embeddings(tokens)
        decodings, hidden = self.decoder(embeddings, hx=previous_hidden)
        if hasattr(self, "attention"): decodings, attention_weights = self.attention(decodings, encodings)
        scores = self.hidden_to_output(decodings)
        return scores, hidden

    def bidirectional_reshape(self, hidden):
        hn, cn = hidden
        H, B, E = hn.shape
        H = int(H/2)
        hn_new = torch.zeros((H, B, E * 2)).to(self.device)
        cn_new = torch.zeros((H, B, E * 2)).to(self.device)
        for h in range(H):
            i = h*2
            h_lr, h_rl = hn[i], hn[i + 1]
            c_lr, c_rl = cn[i], cn[i + 1]
            hn_new[h] = torch.cat((h_lr, h_rl), dim=1)
            cn_new[h] = torch.cat((c_lr, c_rl), dim=1)
        return hn_new, cn_new


def train_batch(X, y, model, optimizer, criterion):

    model.train()

    optimizer.zero_grad()

    X = X.to(model.device)
    y = y.to(model.device)

    scores = model(X, y)

    # Drop $START token since model doesn't return its score
    y = y[:, 1:]

    y_flat = y.contiguous().view(-1)
    scores_flat = scores.reshape(scores.shape[0] * scores.shape[1], -1)

    loss = criterion(scores_flat, y_flat)
    loss.backward()

    optimizer.step()

    return loss.detach().item()


def evaluate(model, X, y):

    model.eval()

    X = X.to(model.device)
    y = y.to(model.device)

    scores = model.predict(X, max_length=y.shape[1])

    # Drop $START token since model doesn't return its score
    y = y[:, 1:]

    n_correct_words, n_possible_words = 0, 0
    n_correct_chars, n_possible_chars = 0, 0

    """ Im sure this can be written in a more clean an efficient way... 
    But all main computation is vectorized already so I didn't bother"""

    for b in range(y.shape[0]):

        word = y[b]
        word_scores = scores[b]
        predicted_chars = word_scores.argmax(dim=1)

        correct_word = True
        for c, char in enumerate(word):
            if char == model.output_vocab["$PAD"]: break
            predicted_char = predicted_chars[c]
            n_possible_chars += 1
            correct_char = (char == predicted_char).item()
            if correct_char: n_correct_chars += 1
            else: correct_word = False

        n_possible_words += 1
        n_correct_words += correct_word

    char_acc = n_correct_chars / n_possible_chars
    word_acc = n_correct_words / n_possible_words

    return word_acc, char_acc
