import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd
import nltk
import math
import time
import wandb
from lstmdataset import dataset 
from lstmmodel import LSTMModel


nltk.download('punkt')
# Time function to help with progress
def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return f'{m}m {s}s'

###
projectname="lstm-diplomatiki-final"
name = "lstm150k-final"
device = "cuda" if torch.cuda.is_available() else "cpu"


batch_size = 128
epochs = 100
text_num=150000

# Load Data
data = pd.read_csv(r"C:\Users\Giannis\Desktop\diplomatikis\AI_Human.csv")
human = data.loc[data['generated'] == 0.0].reset_index(drop=True).head(text_num)

X_train, X_val, X_test, y_train, y_val, y_test, total_words = dataset(human)

X_train = torch.LongTensor(X_train)
y_train = y_train.long()
X_val = torch.LongTensor(X_val)
y_val = y_val.long()
X_test = torch.LongTensor(X_test)
y_test = y_test.long()


train_data = TensorDataset(X_train, y_train)
val_data = TensorDataset(X_val, y_val)
test_data = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)


#model
model = LSTMModel(total_words,embedding_dim=128).to(device)

#  loss , optimizer
criterion = nn.CrossEntropyLoss(ignore_index=0) ## ignore to padding=0
optimizer = optim.Adam(model.parameters(),lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, verbose=True
)
#WandB
wandb.login()
wandb.init(
    project=projectname,
    name=name, 
    config={
        "epochs": epochs,
        "device": device,
        "batch_size": batch_size
    }
)

# Training
start = time.time()

if device == "cuda":
    print("Start of training:")
    for epoch in range(epochs):
        model.train()
        train_loss=0.0
        train_correct = 0
        train_total = 0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch= y_batch.to(device).long()

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0) # sum loss over samples
            

            preds = outputs.argmax(dim=1)
            train_correct += (preds == y_batch).sum().item()
            train_total += y_batch.size(0)
        
        # Eval
        model.eval()
        val_loss=0.0
        val_total=0
        val_correct=0
        with torch.no_grad():
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch= X_val_batch.to(device)
                y_val_batch = y_val_batch.to(device).long()

                val_outputs = model(X_val_batch)
                v_loss = criterion(val_outputs, y_val_batch)

                val_loss += v_loss.item() * X_val_batch.size(0)

                val_preds = val_outputs.argmax(dim=1)
                val_correct += (val_preds == y_val_batch).sum().item()
                val_total += y_val_batch.size(0)
        # Calculations
        train_loss /= train_total
        train_acc = train_correct / train_total
        val_loss /= val_total
        val_acc = val_correct / val_total

        scheduler.step(val_loss)
        
        print(f"Time: {time_since(start)}, "
            f"Epoch {epoch}, "
            f"Train Loss: {train_loss:.4f}, "
            f"Train Acc: {train_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val Acc: {val_acc:.4f}")
        
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

    # Save model
    torch.save(model.state_dict(), f"{name}.pth")
else:
    print("de trexei se gpu")