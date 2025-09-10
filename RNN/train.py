import numpy as np
import pandas as pd
import nltk
import string
import unidecode
import random
import torch
import torch.nn as nn
import zipfile
import os
import pickle
from nltk.tokenize import sent_tokenize
import re
import time, math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
#%matplotlib inline
from preprocess import *
from model import RNN
from test import generate_text
from sklearn.model_selection import train_test_split
from dataset import TextDataset
from torch.utils.data import Dataset, DataLoader
import wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau


n_epochs = 100
print_every = 3
plot_every = 1
hidden_size = 512
n_layers = 3
lr = 0.0001
ngrams=5
project= "rnn-diplomatiki-update"
name=f"rnn-10k-hidden{hidden_size}-nl{n_layers}-lr{lr}-ngram{ngrams}-lessvocab"


def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

if __name__ == '__main__':
  from multiprocessing import freeze_support
  freeze_support() 

  nltk.download('punkt_tab')
  data = pd.read_csv(r"C:\Users\Giannis\Desktop\diplomatikis\AI_Human.csv")

  human=data.loc[data['generated']==0.0]
  human=human.reset_index()
  human.drop('index',axis='columns', inplace=True)

  humanena=human.head(10000)

  with open('vocab.pkl', 'rb') as f:
    vocab_dict = pickle.load(f)

  word_to_ix = vocab_dict["word_to_ix"]
  idx_to_word = vocab_dict["idx_to_word"]
  voc_len=vocab_dict['voc_len']


  for i in range(0,len(humanena)):
    a = humanena.iloc[i]['text']
    a = re.sub(r'([.,\/;!?@#$%^&*()_\-+={}`~\[\]<>\"\':|\\])', r' \1 ', a)
    a= re.sub(r'\s+', ' ', a).strip()
    a=sent_tokenize(a)
    a = [sentence + " <EOS> " for sentence in a]
    a=''.join(a)
    humanena.at[i, 'text']=a

  ##########################################

  train_on_gpu = torch.cuda.is_available()
  if(train_on_gpu):
    device = 'cuda'
    print('Training on GPU!')
  else:
    device = "cpu"
    print('No GPU available, training on CPU; consider making n_epochs very small.')


  def train(decoder, batch):
    inputs_list, targets_list = batch
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    avg_loss=0
    avg_accuracy=0

    for inputs, targets in zip(inputs_list, targets_list):
        inputs, targets = inputs.to(device), targets.to(device)
        hidden = decoder.init_hidden(inputs.size(0)).to(device)

        decoder.zero_grad()
        with torch.cuda.amp.autocast():
          outputs, hidden = decoder(inputs, hidden)
          loss = criterion(outputs.view(-1, voc_len), targets.view(-1))

        scaler.scale(loss).backward()
        scaler.step(decoder_optimizer)
        scaler.update()
        total_loss += loss.item()


        predicted = outputs.argmax(dim=-1)  # [batch, seq_len]
        correct = (predicted == targets).float().sum().item()
        total_correct += correct
        total_tokens += targets.numel()

    avg_loss=total_loss / len(inputs_list)  # average loss
    avg_accuracy = total_correct / total_tokens
    return avg_loss , avg_accuracy


  def evaluate(decoder, batch):
    inputs_list, targets_list = batch
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    avg_loss=0
    avg_accuracy=0

    for inputs, targets in zip(inputs_list, targets_list):
        inputs, targets = inputs.to(device), targets.to(device)
        hidden = decoder.init_hidden(inputs.size(0)).to(device)

        decoder.zero_grad()
        with torch.cuda.amp.autocast():
          outputs, hidden = decoder(inputs, hidden)
          loss = criterion(outputs.view(-1, voc_len), targets.view(-1))

        scaler.scale(loss).backward()
        scaler.step(decoder_optimizer)
        scaler.update()
        total_loss += loss.item()


        predicted = outputs.argmax(dim=-1)  # [batch, seq_len]
        correct = (predicted == targets).float().sum().item()
        total_correct += correct
        total_tokens += targets.numel()

    avg_loss=total_loss / len(inputs_list)  # average loss
    avg_accuracy = total_correct / total_tokens
    return avg_loss , avg_accuracy


  wandb.login() # for logs and better understanding

  decoder = RNN(voc_len, hidden_size, voc_len, n_layers)
  decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
  scheduler = ReduceLROnPlateau(
    decoder_optimizer,
    mode='min',
    factor=0.5,
    patience=2
    )
  criterion = nn.CrossEntropyLoss()


  start = time.time()
  all_losses = []
  all_val_losses = []
  train_loss_avg = 0
  val_loss_avg = 0
  train_acc_avg=0
  val_acc_avg=0
  all_perplexities=[]
  all_val_perplexities=[]

  X_temp, X_test = train_test_split(humanena["text"], test_size=0.1, random_state=42)
  X_train, X_val = train_test_split(X_temp, test_size=0.2, random_state=42)  # 0.25 x 0.8 = 0.2


  if(train_on_gpu):
    decoder.to(device)
    decoder.train()
    scaler = torch.amp.GradScaler('cuda')
    train_dataset = TextDataset(X_train, word_to_ix,seq_length=ngrams) #############ngrams
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,      
        shuffle=True,       
        num_workers=4,      # Parallel load 4 threads
        pin_memory=True,     
        collate_fn=custom_collate_fn
      )
    
    val_dataset = TextDataset(X_val,word_to_ix, seq_length=ngrams)  ##############ngrams
    val_loader = DataLoader(
      val_dataset,         
      batch_size=64,      
      shuffle=True,       
      num_workers=4,      
      pin_memory=True,     
      collate_fn=custom_collate_fn
      )
    wandb.init(
      project=project,
      name=name, 
      config={
          "epochs": n_epochs,
          "hidden_size": hidden_size,
          "learning_rate": lr,
          "n_layers": n_layers,
          "device": device,
          "batch_size": 64,
          "vocab_size": voc_len,
          }
        )

    print("Start of training:")
    for epoch in range(1,n_epochs+1):
      for batch in train_loader:
        loss,acc = train(decoder, batch)
        train_loss_avg += loss
        train_acc_avg += acc

      for val_batch in val_loader:
        val_loss,val_acc = evaluate(decoder, val_batch)
        val_loss_avg += val_loss
        val_acc_avg += val_acc

      #if epoch % print_every == 0:
      train_loss_avg=train_loss_avg / len(train_loader)
      train_acc_avg=train_acc_avg / len(train_loader)

      val_loss_avg = val_loss_avg / len(val_loader)
      val_acc_avg = val_acc_avg / len(val_loader)

      scheduler.step(val_loss_avg)  ###reduceonplateau

      train_perplexity = math.exp(train_loss_avg)
      val_perplexity = math.exp(val_loss_avg)

      print(f"train perplexity:{train_perplexity}")

      print('Time :[%s (%d %d%%)]\nTrain: loss= %.4f | acc = %.4f' % (time_since(start), epoch, epoch / n_epochs * 100, train_loss_avg, train_acc_avg))
      print('Val: Loss : %.4f | Acc: %.4f' % ( val_loss_avg , val_acc_avg))


      wandb.log({
        "train_acc": train_acc_avg,
        "train_loss": train_loss_avg,
        "val_acc": val_acc_avg,
        "val_loss": val_loss_avg,
        "train_perplexity": train_perplexity,
        "val_perplexity": val_perplexity,
        "epoch": epoch
      })


      if epoch % plot_every == 0:
        all_perplexities.append(train_perplexity)
        all_val_perplexities.append(val_perplexity)

        all_losses.append(train_loss_avg / plot_every)
        all_val_losses.append(val_loss_avg / plot_every)

        train_acc_avg=0
        train_loss_avg = 0
        val_acc_avg=0
        val_loss_avg = 0


  torch.save(decoder.state_dict(), name + ".pth")



  plt.figure()
  plt.plot(all_perplexities, label='Train Perplexity')
  plt.plot(all_val_perplexities, label='Val Perplexity')
  plt.title('Perplexity Over Epochs')
  plt.xlabel('Epoch')
  plt.ylabel('Perplexity')
  plt.legend()
  plt.grid(True)


  plt.figure()
  plt.plot(all_losses, label="Train loss")
  plt.plot(all_val_losses, label="Val loss")
  plt.title('Loss Over Epochs')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend()
  plt.grid(True)

  plt.show()
  ############ quick test just to see if it works

  starting_words = ["what", "are"]

  generated_text = generate_text(
      decoder,
      starting_words,
      word_to_ix,
      idx_to_word,
      temperature=0.6
  )
  print(generated_text)
  wandb.finish()