import argparse
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from random import shuffle
import os
from .data import parse_corpus#, format_data
from .model import Net, BiRNN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_data(path, seq_length, batch_size, mode):
    data, dataX, dataY, target_to_int, int_to_target, targets = parse_corpus(path, mode, batch_size, seq_length=seq_length)
    # data = format_data(dataX, dataY, n_classes=len(targets), batch_size=batch_size)
    return data, dataX, dataY, target_to_int, int_to_target, targets

def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def train(data, log_interval):
    model.train().to(device)   # for Dropout & BatchNorm
    # model.zero_grad() # set this or optimizer to zero
    loss = 0
    counter = 0
    for batch_i, (seq_in, target) in enumerate(data):
        seq_in, target = Variable(seq_in), Variable(target)
        optimizer.zero_grad()

        output = model(seq_in)
        loss = F.cross_entropy(output, target)
        # print(output.size())
        # print(output)
        # print(target.size())
        # print(target)
        loss.backward()
        optimizer.step()

        # Log training status
        if batch_i % log_interval == 0:
            print('Train epoch: {} ({:2.0f}%)\tLoss: {:.6f}'.format(epoch, 100. * batch_i / len(data), loss.data.item()))

    '''
    for ind, (seq_in, target) in enumerate(data):
        # print(seq_in,target)
        seq_in, target = Variable(torch.LongTensor(seq_in)), Variable(torch.LongTensor([target]))
        optimizer.zero_grad()

        output = model(seq_in)
        # print(output.shape, target.shape)
        loss += F.cross_entropy(output, target).to(device)
        counter +=1

        if ind%(batch_size) == 0 and ind:
            loss.backward()
            optimizer.step()
            if ind%(batch_size*log_interval) == 0:
                print('Train epoch: {:02}.{:2.0f}\tLoss: {:.6f}' \
                        .format(epoch, 100. * ind / len(data), loss.data.item()/counter))
            counter = 0
            loss = 0
    '''


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train seq2seq model')
    parser.add_argument('--corpusbin', type=str, default="corpus.bin", metavar='F',
                        help='corpus bin (default: corpus.bin)')
    parser.add_argument('--modelbin', type=str, default="model.bin", metavar='F',
                        help='model bin (default: model.bin)')
    parser.add_argument('corpus', type=str, metavar='F',
                        help='training corpus file')
    parser.add_argument('--seq-length', type=int, default=200, metavar='N',
                        help='input sequence length (default: 50)')
    parser.add_argument('--batch-size', type=int, default=30, metavar='N',
                        help='training batch size (default: 1)')
    parser.add_argument('--embedding-dim', type=int, default=128, metavar='N',
                        help='embedding dimension for characters/words in corpus (default: 128)')
    parser.add_argument('--hidden-dim', type=int, default=64, metavar='N',
                        help='hidden state dimension (default: 64)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--dropout', type=float, default=0.1, metavar='DR',
                        help='dropout rate (default: 0.2)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 30)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
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

    # Load mappings & vocabularies
    print("####################################################")
    print("# loading... " + os.path.abspath(args.corpusbin))
    print("# loading... " + os.path.abspath(args.modelbin))
    print("####################################################")
    print()

    if os.path.exists(args.corpusbin) and os.path.exists(args.modelbin):
        comfirm = input("Train with existing model.bin and corpus.bin? [Y/n]")
        if comfirm == "y" or comfirm == "Y":
            dataX, dataY, target_to_int, int_to_target, targets = load_pickle(args.corpusbin)
            # train_data = format_data(dataX, dataY, n_classes=len(targets), batch_size=args.batch_size)
            model = torch.load(args.modelbin)
        else:
            print("Re-train the model ...")
            train_data, dataX, dataY, target_to_int, int_to_target, targets = load_data(args.corpus,
                                                                            seq_length=args.seq_length,
                                                                            batch_size=args.batch_size,
                                                                            mode=args.mode)
            model = Net(len(targets), args.embedding_dim, args.hidden_dim, len(targets),
                        n_layers=2,
                        dropout=args.dropout)
            # model = BiRNN(len(targets), args.embedding_dim, args.hidden_dim, len(targets),
            #                 n_layers=2,
            #                 dropout=args.dropout)
    else:
        print("Train a new model ...")
        train_data, dataX, dataY, target_to_int, int_to_target, targets = load_data(args.corpus,
                                                                        seq_length=args.seq_length,
                                                                        batch_size=args.batch_size,
                                                                        mode=args.mode)
        model = Net(len(targets), args.embedding_dim, args.hidden_dim, len(targets),
                    n_layers=2,
                    dropout=args.dropout)
        # model = BiRNN(len(targets), args.embedding_dim, args.hidden_dim, len(targets),
        #                 n_layers=2,
        #                 dropout=args.dropout)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # criterion = nn.CrossEntropyLoss()

    # Train
    try:
        for epoch in range(args.epochs):
            # shuffle(train_data)
            shuffle(train_data)
            train(train_data, log_interval=args.log_interval)

            if (epoch + 1) % args.save_interval == 0:
                model.eval()
                torch.save(model, args.output)

        # Save mappings, vocabs, & model
        print("Saving...")
        save_pickle((dataX, dataY, target_to_int, int_to_target, targets), args.output_c)
        model.eval()
        torch.save(model, args.output)

    except KeyboardInterrupt:
        print("Saving before quit...")
        save_pickle((dataX, dataY, target_to_int, int_to_target, targets), args.output_c)
        model.eval()
        torch.save(model, args.output)
