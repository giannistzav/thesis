from collections import Counter
import re
import torch

def removepunctuation(string):
  string = string.lower()
  a=",./;:!?@#$%^&*()_-+={}[]<>`~'\"|\\«»“”‘’—–‐‑"
  string = re.sub(r'\s+', ' ', string).strip()
  for i in string:
    if i in a:
      string=string.replace(i," ")
  if string[0]==' ':
    string=string[1:]
  if string[-1]==' ':
    string=string[:-1]
  return string


def ngrams(test_sentence,n):
  test_sentence=test_sentence.split()
  ngram=[]
  b=[]
  c=()
  for i in range(len(test_sentence) - (n-1)):
    for k in range(i,i+(n-1)):
      b.append(test_sentence[k])

    c=(b,test_sentence[i+n-1])
    ngram.append(c)
    b=[]
  return ngram

UNK_TOKEN = "<UNK>"
def intar(sentence, n, word_to_ix):
  inp = []
  tar = []
  ngram_list = ngrams(sentence, n)
  for context, target in ngram_list:
    context_idxs = torch.tensor(
      [word_to_ix.get(w, word_to_ix[UNK_TOKEN]) for w in context],
      dtype=torch.long
    )
    inp.append(context_idxs)

    target_idx = torch.tensor(
      [word_to_ix.get(target, word_to_ix[UNK_TOKEN])],
      dtype=torch.long
    )
    tar.append(target_idx)

  inp = torch.stack(inp)
  tar = torch.stack(tar)
  return inp, tar, len(ngram_list)

def custom_collate_fn(batch):
  inputs, targets = zip(*batch)

  return list(inputs), list(targets)
