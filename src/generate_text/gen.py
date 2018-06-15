import argparse
import copy

import numpy as np
import torch
from torch.autograd import Variable
from train.dataUtil import load_pickle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokens = ['<SOS>', '<EOS>', '<PAD>', '<UNK>']


def is_end(c):
    end_tokens = ['。', '？', '！', '.', '?', '!', '』', '」', ')', '）']
    return c in end_tokens


def to_prob(vec):
    s = sum(vec)
    return [v / s for v in vec]


def gen_text(model, trainData, target_to_int, int_to_target, targets, temperature=0.8, n_sent=10, restart_seq=False):
    patterns = copy.deepcopy(trainData)  # for list pass by reference issue
    n_patterns = len(patterns)

    # Randomly choose a pattern to start text generation
    current_seq = patterns[np.random.randint(0, n_patterns - 1)]
    # print(start, end='')
    # Start generation until n_sent sentences generated
    cnt = 0
    while cnt < n_sent:
        # Format input pattern
        seq_in = np.array(current_seq)
        seq_in = seq_in.reshape(1, -1)  # batch_size = 1
        seq_in = Variable(torch.LongTensor(seq_in)).to(device)
        # Predict next target
        pred = model(seq_in)
        '''
        pred = to_prob(F.softmax(pred, dim=1).data[0].numpy()) # turn into probability distribution
        target = np.random.choice(targets, p=pred)             # pick char based on probability instead of always picking the highest value
        target_idx = target_to_int[target]
        '''
        pred = pred.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(pred, 1)[0]
        target = targets[top_i]
        target_idx = target_to_int[target]
        if target not in tokens:
            print(target, end='')

        # Append predicted character to pattern, truncate to usual pattern size, use as new pattern
        current_seq.append(target_idx)
        current_seq = current_seq[1:]

        if target == '<EOS>':
            if restart_seq:
                current_seq = patterns[np.random.randint(0, n_patterns - 1)]
                print()

            cnt += 1

    if not restart_seq:
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate text')
    parser.add_argument('corpus', type=str, metavar='F',
                        help='corpus-related data file')
    parser.add_argument('model', type=str, metavar='F',
                        help='model for text generation')
    parser.add_argument('--n-sent', type=int, default=10, metavar='N',
                        help='number of sentences to generate (default: 10)')
    parser.add_argument('--restart-seq', action='store_true',
                        help='whether to randomly pick a new sequence to start the next sentence generation (default: F)')

    args = parser.parse_args()

    # Load mappings & vocabularies
    dataX, dataY, target_to_int, int_to_target, targets = load_pickle(args.corpus)

    # Load model
    model = torch.load(args.model)

    # Generate text
    gen_text(model, dataX, target_to_int, int_to_target, targets, n_sent=args.n_sent, restart_seq=args.restart_seq)
