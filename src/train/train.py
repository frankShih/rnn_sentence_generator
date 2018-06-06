import argparse
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from random import shuffle
import os
from .data import parse_corpus_word, format_data
from .model import Net


def load_data(path, seq_length, batch_size, mode):
    dataX, dataY, target_to_int, int_to_target, targets = parse_corpus_word(path, mode, seq_length=seq_length)

    data = format_data(dataX, dataY, n_classes=len(targets), batch_size=batch_size)
    return data, dataX, dataY, target_to_int, int_to_target, targets

def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def train(model, optimizer, epoch, data, log_interval):            
    model.train()   # for Dropout & BatchNorm
    # model.zero_grad() # set this or optimizer to zero

    '''
    # truncated to the last K timesteps (for gradient vanishing)
    for t in range(T):
        out = model(out)
        if T - t == K:
            out.backward()
            out.detach()
    out.backward()

    # or

    modelparameter.requires_grad = False
    for t in range(T):
        out = model(out)
        if T - t == K:
            modelparameter.requires_grad = True
    out.backward()
    '''
    for batch_i, (seq_in, target) in enumerate(data):
        seq_in, target = Variable(seq_in), Variable(target)
        optimizer.zero_grad()
        
        output = model(seq_in)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        # Log training status
        if batch_i % log_interval == 0:
            print('Train epoch: {} ({:2.0f}%)\tLoss: {:.6f}'.format(epoch, 100. * batch_i / len(data), loss.data.item()))

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train seq2seq model')
    parser.add_argument('corpus', type=str, metavar='F',
                        help='training corpus file')
    parser.add_argument('--seq-length', type=int, default=200, metavar='N',
                        help='input sequence length (default: 50)')
    parser.add_argument('--batch-size', type=int, default=30, metavar='N',
                        help='training batch size (default: 1)')
    parser.add_argument('--embedding-dim', type=int, default=64, metavar='N',
                        help='embedding dimension for characters/words in corpus (default: 128)')
    parser.add_argument('--hidden-dim', type=int, default=64, metavar='N',
                        help='hidden state dimension (default: 64)')
    parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--dropout', type=float, default=0.1, metavar='DR',
                        help='dropout rate (default: 0.2)')
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='number of batches to wait before logging status (default: 10)')
    parser.add_argument('--save-interval', type=int, default=10, metavar='N',
                        help='number of epochs to wait before saving model (default: 10)')
    parser.add_argument('--mode', type=str, default='char', metavar='N',
                        help='char/word level to train model (default: char)')                        
    parser.add_argument('--output', type=str, default='model.bin', metavar='F',
                        help='output model file')
    parser.add_argument('--output-c', type=str, default='corpus.bin', metavar='F',
                        help='output corpus related file (mappings & vocab)')
    args = parser.parse_args()

    # Prepare    
    train_data, dataX, dataY, target_to_int, int_to_target, targets = load_data(args.corpus, seq_length=args.seq_length, batch_size=args.batch_size, mode=args.mode)
    
    model = Net(len(targets), args.embedding_dim, args.hidden_dim, len(targets), n_layers=2, dropout=args.dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # criterion = nn.CrossEntropyLoss()

    # Train
    try:
        for epoch in range(args.epochs):
            shuffle(train_data)
            train(model, optimizer, epoch, train_data, log_interval=args.log_interval)

            if (epoch + 1) % args.save_interval == 0:
                model.eval()
                torch.save(model, args.output)

        # Save mappings, vocabs, & model
        print("Saving...")
        save_pickle((dataX, target_to_int, int_to_target, targets), args.output_c)
        model.eval()
        torch.save(model, args.output)

    except KeyboardInterrupt:
        print("Saving before quit...")
        save_pickle((dataX, target_to_int, int_to_target, targets), args.output_c)
        model.eval()
        torch.save(model, args.output)



