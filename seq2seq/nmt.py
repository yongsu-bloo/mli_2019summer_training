### IMPORT MODULES
# basic
import numpy as np
# torch
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
# text tools
from torchtext.data import Field, BucketIterator
from torchtext.datasets import Multi30k
# utils
import random, time, spacy, os
from argparse import ArgumentParser
from tqdm import tqdm

from beam_search import *
### Utils
tt = time.time
def tokenize(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]
def tokenize_reverse(text):
    return [tok.text for tok in spacy_de.tokenizer(text)][::-1]
def init_weights(m):
    if (type(m) == nn.LSTM) or (type(m) == nn.GRU):
        for name, param in m.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)

### Seq2Seq Model
class Encoder(nn.Module):
    """
    Encode sentences to context vectors
    """
    def __init__(self, input_dim, emd_dim, hidden_dim, num_layers, dropout=0.5, rnn_type="LSTM", bidirectional=False):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.num_directions = 2 if bidirectional else 1
        # Layers
        self.emb = nn.Embedding(input_dim, emd_dim)
        self.dropout = nn.Dropout(dropout)
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(emd_dim, hidden_dim, num_layers, bidirectional=bidirectional)
        else:
            self.rnn = nn.GRU(emd_dim, hidden_dim, num_layers, bidirectional=bidirectional)

    def forward(self, input):
        """
        input: (batch of) input sentence tensors
        output: (batch of) context vectors
        """
        output = self.emb(input)
        output = self.dropout(output)
        output, hidden = self.rnn(output)
        return hidden # Context Vector

class Decoder(nn.Module):
    """
    Decode the context vector ONE STEP
    """
    def __init__(self, hidden_dim, output_dim, num_layers, dropout=0.5, rnn_type="LSTM", bidirectional=False):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        # Layers
        self.emb = nn.Embedding(output_dim, emd_dim) # output_dim: vocabulary size of target
        self.dropout = nn.Dropout(dropout)
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(emd_dim, hidden_dim, num_layers, bidirectional=bidirectional)
        else:
            self.rnn = nn.GRU(emd_dim, hidden_dim, num_layers, bidirectional=bidirectional)
        self.fc = nn.Linear(emd_dim, output_dim)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        """
        input1: (batch of) words. size: [batch_size]
        input2: (batch of) hidden from last layer(decoder) or context vector from encoder.
        output1: (batch of) translated words. size: [batch_size, output_dim]
        output2: same type of input2
        """
        output = self.emb(input.unsqueeze(0))
        output = self.dropout(output)
        # output = F.relu(output)
        output, hidden = self.rnn(output, hidden)
        output = self.log_softmax(self.fc(output.squeeze(0)))
        return output, hidden

class Seq2Seq(nn.Module):
    """
    Combine Encoder and Decoder
    """
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, source, target):
        """
        input: (batch of) pairs of source and target sequences. size: [sentence_length, batch_size]
        """
        max_length = target.shape[0]
        batch_size = target.shape[1]
        outputs = torch.zeros(max_length, args.batch_size, self.decoder.output_dim, device=device)

        hidden = self.encoder(source)

        if self.training:
            input = target[0,:] # a batch of <sos>'s'
            for i in range(1, max_length):
                # Teacher forcing
                output, hidden = self.decoder(input, hidden)
                outputs[i] = output
                input = target[i]
        else:
            # Beam Search: top 2
            beam_width = 2
            n_sen = 1
            t1 = tt()
            decode_batch = beam_decode(self.decoder, target, hidden, beam_width, n_sen)
            t2 = tt()
            print("Beam Search: {:.3f} sec\n".format(t2-t1))
            outputs = torch.tensor(decode_batch)

        return outputs
### Train and Evaluation
def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    clip = 1
    for i, batch in tqdm(enumerate(iterator), desc="a epoch (# batches)"):
        optimizer.zero_grad()

        src = batch.src
        trg = batch.trg

        output = model(src, trg)

        trg = trg[1:].view(-1)
        output = output[1:].view(-1, output.shape[-1])

        loss = criterion(output, trg)
        loss.backward()
        # clip
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):

            src = batch.src
            trg = batch.trg

            output = model(src, trg) #turn off teacher forcing

            trg = trg[1:].view(-1)
            output = output[1:].view(-1, output.shape[-1])

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-seed', type=int, default=9)
    parser.add_argument('-b', "--batch_size", type=int, default=128, help='batch size(default=128)')
    parser.add_argument('-num-layers', type=int, default=4)
    parser.add_argument('-emd-dim', type=int, default=1000)
    parser.add_argument('-hidden-dim', type=int, default=1000)
    parser.add_argument('--no-reverse', help='not to reverse input seq', action='store_true')
    parser.add_argument('--bidirectional', help='bidirectional rnn', action='store_true')
    parser.add_argument('-lr', type=float, default=1e-3)
    parser.add_argument('-rnn-type', choices=['LSTM', 'GRU'], default="LSTM", help="LSTM or GRU")
    parser.add_argument('-opt', choices=['adam', 'sgd'], default='sgd')
    parser.add_argument('-epochs', type=int, default=10)
    parser.add_argument('--cpu', help='forcing to use cpu', action='store_true')
    parser.add_argument('-dropout', type=float, help='dropout rate', default=0.5)
    parser.add_argument('-resume', type=str, help='load model from checkpoint(input: path of ckpt)')

    global args
    args = parser.parse_args()

    global device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    # Random Seed
    random_seed = args.seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True

    # Hyperparameters loading
    epochs = args.epochs
    batch_size = args.batch_size
    rnn_type = args.rnn_type
    reverse = not args.no_reverse
    bidirectional = args.bidirectional
    num_layers = args.num_layers
    emd_dim = args.emd_dim
    hidden_dim = args.hidden_dim
    lr = args.lr

    params = [batch_size, rnn_type, reverse, bidirectional, num_layers, emd_dim, hidden_dim,
                lr]
    PATH = os.path.join("models", "seq2seq-{}".format("-".join(str(params))))
    # Preparing data
    t1 = tt()
    spacy_de = spacy.load('de')
    spacy_en = spacy.load('en')
    source_field = Field(sequential = True,
                        use_vocab = True,
                        tokenize = tokenize_reverse if reverse else tokenize,
                        init_token = '<sos>',
                        eos_token = '<eos>',
                        lower = True,
                        batch_first = False)
    target_field = Field(sequential = True,
                        use_vocab = True,
                        tokenize = tokenize,
                        init_token = '<sos>',
                        eos_token = '<eos>',
                        lower = True,
                        batch_first = False)
    train_data, valid_data, test_data = Multi30k.splits(exts = ('.de', '.en'),
                                                        fields = (source_field, target_field))
    source_field.build_vocab(train_data, min_freq = 2)
    target_field.build_vocab(train_data, min_freq = 2)

    train_iterator, valid_iterator, test_iterator = \
                        BucketIterator.splits((train_data, valid_data, test_data),
                                                batch_size = batch_size,
                                                device = device)
    t2 = tt()
    print("Data ready: {:.3f} sec\n".format(t2-t1))

    # model inputs
    input_dim = len(source_field.vocab)
    output_dim = len(target_field.vocab)

    encoder = Encoder(input_dim=input_dim, emd_dim=emd_dim, hidden_dim=hidden_dim,
                        num_layers=num_layers, rnn_type=rnn_type, bidirectional=bidirectional).to(device)
    decoder = Decoder(hidden_dim=hidden_dim, output_dim=output_dim,
                        num_layers=num_layers, rnn_type=rnn_type, bidirectional=bidirectional).to(device)
    model = Seq2Seq(encoder, decoder).to(device)
    model.apply(init_weights) # weight initialization

    # Training
    # @TODO: Existing Model load
    print("Model Training Starts\n")
    t1 = tt()
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=1e-3) if args.opt == "sgd" else optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.NLLLoss(ignore_index=target_field.vocab.stoi['<pad>'])

    loss = 0
    best_eval_loss = float("inf")
    for epoch in tqdm(range(epochs), desc="Total train (# epochs)"):
        train_loss = train(model, train_iterator, optimizer, criterion)
        eval_loss = evaluate(model, valid_iterator, criterion)
        print("Train loss: {:.4f}, Eval loss: {:.4f}\n".format(train_loss, eval_loss))
        if eval_loss < best_eval_loss:
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    'args': args
                    }, PATH + "-{}.ckpt".format(epoch))
            best_eval_loss = eval_loss
            print("Best Model Updated\n")

    t2 = tt()
    print("Model Training ends ({:.3f} min)\n".format((t2-t1) / 60))
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'args': args
            }, PATH + "-final.ckpt")
    print("Model Saved\n")
    # Evaluation
    # @TODO load existing model
    model.eval()
