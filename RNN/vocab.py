from collections import Counter
from entropyfunc import removepunctuation
import pickle
import nltk
import numpy as np
import pandas as pd

filename='vocab-lstm.pkl'
nltk.download('punkt_tab')
data = pd.read_csv(r"C:\Users\Giannis\Desktop\diplomatikis\AI_Human.csv")
human=data.loc[data['generated']==0.0]
human=human.reset_index()
human.drop('index',axis='columns', inplace=True)

humanena=human.head(10000)

vocab=[]
all_words=[]




for i in range(0,len(humanena)):
    c=removepunctuation(humanena.iloc[i]['text'])
    c=c.split()
    all_words.extend(c)



MAX_VOCAB_SIZE = 15000 ### vocab size( most common words )

word_freq = Counter(all_words)
most_common = word_freq.most_common(MAX_VOCAB_SIZE - 3)  # -3 for tokens

vocab = [w for w, _ in most_common]
vocab += ['<PAD>', '<UNK>', '<EOS>']


punct=",./;:!?@#$%^&*()_-+={}[]<>`~'\"|\\«»“”‘’—–‐‑"
for i in punct:
    if i not in vocab:
        vocab.append(i)


voc_len=len(vocab)
word_to_ix = {word: i for i, word in enumerate(vocab)}
idx_to_word = {v: k for k, v in word_to_ix.items()}


vocab_dict = {
    "word_to_ix": word_to_ix,
    "idx_to_word": idx_to_word,
    "voc_len": voc_len
}

# save .pkl
with open(filename, 'wb') as f:
    pickle.dump(vocab_dict, f)



