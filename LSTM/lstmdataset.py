import numpy as np
import torch
import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
from nltk.tokenize import word_tokenize
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset
import nltk
def tokenize(text):
    return word_tokenize(str(text).lower())

def to_onehot(labels, num_classes):
        y = torch.zeros(len(labels), num_classes)
        y.scatter_(1, labels.unsqueeze(1), 1)
        return y

def pad_sequences(sequences, maxlen):
    padded = torch.zeros(len(sequences), maxlen, dtype=torch.long)
    for i, seq in enumerate(sequences):
        if len(seq) > maxlen:
            padded[i] = torch.tensor(seq[-maxlen:], dtype=torch.long)
        else:
            padded[i, -len(seq):] = torch.tensor(seq, dtype=torch.long)
    return padded

def texts_to_sequences(texts,word_index):
    sequences=[]
    for text in texts:
        seqtemp=[]
        for word in tokenize(text):
            seq = word_index.get(word, word_index["<OOV>"])
            seqtemp.append(seq)
        sequences.append(seqtemp)
    return sequences



def create_labels(sequences):
    X = sequences[:, :-1]  # context
    y = sequences[:, -1]   # target
    return X, y


def dataset(data):
    filename='vocab_lstm.pkl'
    most_common=30000    ##### set most common words 


    
    datatext=data['text']
    X_temp, X_test = train_test_split(datatext, test_size=0.1, random_state=42)
    X_train, X_val = train_test_split(X_temp, test_size=0.2, random_state=42)

    X_test=X_test.reset_index()
    X_test.drop('index',axis='columns', inplace=True)

    X_train=X_train.reset_index()
    X_train.drop('index',axis='columns', inplace=True)

    X_val=X_val.reset_index()
    X_val.drop('index',axis='columns', inplace=True)

    sentence_lengths = [len(word_tokenize(str(text))) for text in datatext]
    max_seq_len = int(np.mean(sentence_lengths))


    # vocabulary
    word_counts = {}
    for text in X_train['text']: ## avoid leakage.
        for word in tokenize(text):
            word_counts[word] = word_counts.get(word, 0) + 1

    
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    sorted_words=sorted_words[0:most_common] ##most common words
    PAD_TOKEN = "<PAD>"
    OOV_TOKEN = "<OOV>"
    word_index = {
        PAD_TOKEN: 0,  # padding 0
        OOV_TOKEN: 1   # OOV 1
    }
    for i, (word, count) in enumerate(sorted_words):
        word_index[word] = i + 2  # Starting from 2 as 0 is assigned to padding and 1 to OOV


    idx_to_word = {idx: word for word, idx in word_index.items()}

    total_words = len(word_index)
    vocab_dict_lstm = {
    "word_index": word_index,
    "idx_to_word": idx_to_word,
    "total_words": total_words
    }

    # Save .pkl
    with open(filename, 'wb') as f:
        pickle.dump(vocab_dict_lstm, f)




    train_sequences = texts_to_sequences(X_train['text'],word_index)
    val_sequences = texts_to_sequences(X_val['text'],word_index)
    test_sequences = texts_to_sequences(X_test['text'],word_index)
    
    
    train_padded = pad_sequences(train_sequences, maxlen=max_seq_len-1)
    val_padded = pad_sequences(val_sequences, maxlen=max_seq_len-1)
    test_padded = pad_sequences(test_sequences, maxlen=max_seq_len-1)
    
    # Δημιουργία labels
    X_train_final, y_train_final = create_labels(train_padded)
    X_val_final, y_val_final = create_labels(val_padded)
    X_test_final, y_test_final = create_labels(test_padded)
    
    y_train_idx = torch.LongTensor(y_train_final)
    y_val_idx = torch.LongTensor(y_val_final)
    y_test_idx = torch.LongTensor(y_test_final)
    
    return (
        X_train_final, 
        X_val_final, 
        X_test_final, 
        y_train_idx, 
        y_val_idx, 
        y_test_idx, 
        total_words
    )


    __all__ = ['dataset', 'tokenize']