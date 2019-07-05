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


from beam_search import *
from bleu import *
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
        self.fc = nn.Linear(hidden_dim * self.num_directions, output_dim)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        """
        input1: (batch of) words. size: [batch_size]
        input2: (batch of) hidden from last layer(decoder) or context vector from encoder. [num_layers*num_directions, batch_size, output_dim]
        output1: (batch of) translated words. size: [batch_size, output_dim]
        output2: same type of input2
        """
        # print("Input shape: {}\n".format(input.shape))
        output = self.emb(input).unsqueeze(0)
        output = self.dropout(output)
        # print("Embeded shape: {}\n".format(output.shape))
        # print("Hidden shape: {}\n".format(" and ".join([ str(x.shape) for x in hidden])))
        # output = F.relu(output)
        output, hidden = self.rnn(output, hidden)
        # print("After RNN shape: {}\n".format(output.shape))
        # output: [1, batch_size, hidden_dim]
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
        output: (batch of) translated sentences. size: (train)[max_sentence_length, batch_size, target_vocab_size], (test)[max_sentence_length, batch_size]
        """
        max_length = target.shape[0]
        batch_size = target.shape[1]

        hidden = self.encoder(source)
        if self.training:
            outputs = torch.ones(max_length, batch_size, self.decoder.output_dim, device=device) * target_field.vocab.stoi['<sos>']
            input = target[0,:] # a batch of <sos>'s'
            for i in range(1, max_length):
                # Teacher forcing
                output, hidden = self.decoder(input, hidden)
                outputs[i] = output
                input = target[i]
        else:
            # Beam Search: top 2
            outputs = torch.zeros(max_length, batch_size, device=device)
            beam_width = 2
            n_sen = 1
            t1 = tt()
            decode_batch = beam_decode(self.decoder, target, hidden, beam_width, n_sen) # returns: python list of sentence(list of str). size: [batch_size, sentence_length]
            t2 = tt()
            # print("Beam Search: {:.3f} sec\n".format(t2-t1))
            for i in range(batch_size):
                if len(decode_batch[i]) < max_length:
                    output = F.pad(torch.tensor(decode_batch[i], dtype=torch.int), (0, max_length - len(decode_batch[i])), 'constant', target_field.vocab.stoi['<eos>'])
                else:
                    output = torch.tensor(decode_batch[i], dtype=torch.int)
                outputs[:,i] = output

        return outputs
### Train and Evaluation
def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    clip = 1
    for i, batch in enumerate(iterator):
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
        # break # for debugging
    return epoch_loss / len(iterator)

def evaluate(model, iterator):
    #@TODO bleu score
    model.eval()
    bleu_score = 0
    with torch.no_grad():
        for i, batch in enumerate(iterator):

            src = batch.src
            trg = batch.trg

            output = model(src, trg)
            trg = trg.transpose(0,1)
            output = output.transpose(0,1)
            bleu_score += get_bleu(output, trg)
    return bleu_score

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-seed', type=int, default=9)
    parser.add_argument('-b', "--batch_size", type=int, default=128, help='batch size(default=128)')
    parser.add_argument('-num-layers', type=int, default=4)
    parser.add_argument('-emd-dim', type=int, default=256)
    parser.add_argument('-hidden-dim', type=int, default=512)
    parser.add_argument('--no-reverse', help='not to reverse input seq', action='store_true')
    parser.add_argument('--bidirectional', help='bidirectional rnn', action='store_true')
    parser.add_argument('-lr', type=float, default=1e-3)
    parser.add_argument('-rnn-type', choices=['LSTM', 'GRU'], default="LSTM", help="LSTM or GRU")
    parser.add_argument('-opt', choices=['adam', 'sgd'], default='sgd')
    parser.add_argument('-epochs', type=int, default=10)
    # parser.add_argument('--cpu', help='forcing to use cpu', action='store_true')
    parser.add_argument('-dropout', type=float, help='dropout rate', default=0.5)
    parser.add_argument('-resume', type=str, help='load model from checkpoint(input: path of ckpt)')
    parser.add_argument('--evaluate', help='Not train, Only evaluate', action='store_true')

    global args
    args = parser.parse_args()

    # global device
    train_only = not args.evaluate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using Device: {}".format(device))
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
    PATH = os.path.join("models", "seq2seq-{}".format("-".join([ str(p) for p in params ])))
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
    if do_train:
    # @TODO: Existing Model load
    print("Model Training Start\n")
    t1 = tt()
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=1e-3) if args.opt == "sgd" else optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.NLLLoss(ignore_index=target_field.vocab.stoi['<pad>'])
        train_losses = []
        best_eval_score = -float("inf")
        for epoch in range(epochs):
            tt1 = tt()
            train_loss = train(model, train_iterator, optimizer, criterion)
            tt2 = tt()
            print("[{}/{}]Train time per epoch: {:.3f}".format(epoch, epochs, tt2-tt1))
            train_losses.append(train_loss)
            eval_score = evaluate(model, valid_iterator)
            tt3 = tt()
            print("[{}/{}]Eval time per epoch: {:.3f}".format(epoch, epochs, tt3-tt2))
            print("[{}/{}]Train loss: {:.4f}, BLEU score: {:.4f}\n".format(epoch, epochs, train_loss, eval_score))
            if eval_score >= best_eval_score:
                torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'losses': train_losses,
                        'args': args
                        }, PATH + "-{}.ckpt".format(epoch))
                best_eval_score = eval_score
                print("Best Model Updated\n")
            # break # for debugging
        t2 = tt()
        print("Model Training ends ({:.3f} min)\n".format((t2-t1) / 60))
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'losses': train_losses,
                'args': args
                }, PATH + "-final.ckpt")
        print("Model Saved\n")
    # Evaluation - Test Dataset
    # @TODO load existing model
    model.eval()
    test_score = evaluate(model, test_iterator)
    if args.resume
