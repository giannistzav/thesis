from torch.utils.data import Dataset, DataLoader
import pandas as pd
from preprocess import intar

class TextDataset(Dataset):
    def __init__(self, texts, word_to_ix,seq_length=3):
        self.texts = texts         
        self.seq_length = seq_length
        self.word_to_ix=word_to_ix

    def __len__(self):
        return len(self.texts) 

    def __getitem__(self, idx):
        text = self.texts.iloc[idx] if isinstance(self.texts, pd.Series) else self.texts[idx]
        inp, tar, _ = intar(text, self.seq_length, self.word_to_ix)
        return inp, tar
        