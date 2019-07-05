### Code from https://github.com/budzianowski/PyTorch-Beam-Search-Decoding/blob/master/decode_beam.py
import operator
import torch
import torch.nn as nn
import torch.nn.functional as F
from queue import PriorityQueue
from itertools import count

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 2
EOS_token = 3
MAX_LENGTH = 20

class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length):
        '''
        :param hiddenstate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        '''
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward

        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward


def beam_decode(decoder, target_tensor, decoder_hiddens, beam_width=2, n_sen=1):
    '''
    :param target_tensor: target indexes tensor of shape [T, B] where B is the batch size and T is the maximum length of the output sentence
    :param decoder_hiddens: input tensor of shape [L*Bi, B, H] for start of the decoding
    :return: decoded_batch
    '''
    topk = 2  # how many sentence do you want to generate
    decoded_batch = []
    unique_id = count()
    max_length = target_tensor.size(0)
    batch_size = target_tensor.size(1)
    # decoding goes sentence by sentence
    for idx in range(batch_size):
        if decoder.num_layers * decoder.num_directions == 1:
            if isinstance(decoder_hiddens, tuple):  # LSTM case
                decoder_hidden = (decoder_hiddens[0][:,idx, :].unsqueeze(0),decoder_hiddens[1][:,idx, :].unsqueeze(0))
            else:
                decoder_hidden = decoder_hiddens[:, idx, :].unsqueeze(0)
        else:
            if isinstance(decoder_hiddens, tuple):  # LSTM case
                decoder_hidden = (decoder_hiddens[0][:,idx, :], decoder_hiddens[1][:,idx, :])
            else:
                decoder_hidden = decoder_hiddens[:, idx, :]

        # Start with the start of the sentence token
        if device == torch.device("cuda"):
            decoder_input = torch.cuda.LongTensor([[SOS_token]])
        else:
            decoder_input = torch.LongTensor([[SOS_token]])
        # Number of sentence to generate
        endnodes = []
        number_required = min((topk + 1), topk - len(endnodes))

        # starting node -  hidden vector, previous node, word id, logp, length
        node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1)
        nodes = PriorityQueue()

        # start the queue
        nodes.put((-node.eval(), next(unique_id), node))
        qsize = 1

        # First Step of Beam Search

        # fetch the best node
        score, id , n = nodes.get()
        decoder_input = n.wordid
        decoder_hidden = n.h

        # decode for one step using decoder
        if isinstance(decoder_hiddens, tuple): # LSTM
            decoder_output, decoder_hidden = decoder(decoder_input.squeeze(0), \
            (decoder_hidden[0].contiguous().view(decoder.num_layers*decoder.num_directions,1,-1), decoder_hidden[1].contiguous().view(decoder.num_layers*decoder.num_directions,1,-1)) )
        else:
            decoder_output, decoder_hidden = decoder(decoder_input.squeeze(0), decoder_hidden.contiguous().view(decoder.num_layers*decoder.num_directions,1,-1) )

        # PUT HERE REAL BEAM SEARCH OF TOP
        log_prob, indexes = torch.topk(decoder_output, beam_width)
        nextnodes = []

        for new_k in range(beam_width):
            decoded_t = indexes[0][new_k].view(1,-1)
            log_p = log_prob[0][new_k].item()

            node = BeamSearchNode(decoder_hidden, n, decoded_t, n.logp + log_p, n.leng + 1)
            score = -node.eval()
            nextnodes.append((score, next(unique_id), node))

        # put them into queue
        for i in range(len(nextnodes)):
            score, id, nn = nextnodes[i]
            nodes.put((score, id, nn))
        # increase qsize
        qsize += len(nextnodes) - 1

        # START BEAM SEARCH MAIN LOOP
        # while True:
        for _ in range(1, max_length):
            # give up when decoding takes too long
            if qsize > 2000: break
            nextnodes = []
            for x in range(beam_width):
                # fetch the best node
                score, id , n = nodes.get()
                decoder_input = n.wordid
                decoder_hidden = n.h

                if n.wordid.item() == EOS_token and n.prevNode != None:
                    endnodes.append((score, id, n))
                    # if we reached maximum # of sentences required

                # decode for one step using decoder
                if isinstance(decoder_hiddens, tuple):
                    decoder_output, decoder_hidden = decoder(decoder_input.squeeze(0), \
                    (decoder_hidden[0].contiguous().view(decoder.num_layers*decoder.num_directions,1,-1), decoder_hidden[1].contiguous().view(decoder.num_layers*decoder.num_directions,1,-1)) )
                else:
                    decoder_output, decoder_hidden = decoder(decoder_input.squeeze(0), decoder_hidden.contiguous().view(decoder.num_layers*decoder.num_directions,1,-1) )

                # PUT HERE REAL BEAM SEARCH OF TOP
                log_prob, indexes = torch.topk(decoder_output, beam_width)


                for new_k in range(beam_width):
                    decoded_t = indexes[0][new_k].view(1,-1)
                    log_p = log_prob[0][new_k].item()

                    node = BeamSearchNode(decoder_hidden, n, decoded_t, n.logp + log_p, n.leng + 1)
                    score = -node.eval()
                    nextnodes.append((score, next(unique_id), node))

            # put them into NEW queue
            nodes = PriorityQueue()
            for i in range(len(nextnodes)):
                score, id, nn = nextnodes[i]
                nodes.put((score, id, nn))
                # increase qsize
            qsize += len(nextnodes) - 1

            # we are done
            if len(endnodes) >= number_required:
                break
            else:
                continue

        # choose nbest paths, back trace them
        if len(endnodes) == 0:
            endnodes = [nodes.get() for _ in range(topk)]

        utterances = []
        for score, _, n in sorted(endnodes, key=operator.itemgetter(0)):
            utterance = []
            utterance.append(n.wordid.item())
            # back trace
            while n.prevNode != None:
                n = n.prevNode
                utterance.append(n.wordid.item())

            utterance = utterance[::-1]
            utterances.append(utterance[:max_length])

        decoded_batch.append(utterances) # -> list(list(list(str)))

        # # use the best one
        # score, _, n = sorted(endnodes, key=operator.itemgetter(0))[0]
        # utterance = []
        # utterance.append(n.wordid.item())
        # # back trace
        # while n.prevNode != None:
        #     n = n.prevNode
        #     utterance.append(n.wordid.item())
        # utterance = utterance[::-1]
        # decoded_batch.append(utterance[:max_length])
        # -> python list of sentence(list of str)
    return decoded_batch
