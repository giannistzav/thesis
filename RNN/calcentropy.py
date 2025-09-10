from entropyfunc import entropy, ttr
import torch
import pickle
import torch.nn as nn

from test import generate_text
from model import RNN

savefile='results\entropy-ttr-rnn.txt'
modelname='rnn-10k-hidden512-nl3-lr0.0001-ngram5-lessvocab'
path_to_sentence_file='sentencesfortextgen\sentences-5words.txt'
vocabfile='vocab.pkl'
max_length=200

# load state_dict
decoder = RNN(
    input_size=10033, 
    hidden_size=512,
    output_size=10033,
    n_layers=3
)

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

decoder.load_state_dict(torch.load(modelname+'.pth',map_location=device))
decoder.to(device)

with open(vocabfile, 'rb') as f:
    vocab_dict = pickle.load(f)

word_to_ix = vocab_dict["word_to_ix"]
idx_to_word = vocab_dict["idx_to_word"]

with open(path_to_sentence_file, 'rb') as f:
    all_words = pickle.load(f)


entr=0
typetr=0
for i in range (0, len(all_words)):
    starting_words = all_words[i].split()

    generated_text = generate_text( ## to model.eval ginetai edo mesa
        model=decoder,
        starting_words=starting_words,
        word_to_ix=word_to_ix,
        idx_to_word=idx_to_word,
        max_length=max_length,
        temperature=0.6,
        device=device
    )
    t=generated_text.split()
    if t[-1]=='<EOS>':
        t=t[:-1]
        generated_text=' '.join(t)

    entr+=entropy(generated_text)
    typetr+=ttr(generated_text)


calc_entr=entr/len(all_words)
calc_ttr=typetr/len(all_words)

calculations=f'\nmodel: {modelname} \n length of sentences:{max_length}\n calculated entropy: {calc_entr} \n calculated ttr: {calc_ttr} .'
with open(savefile, 'ab') as f:
        pickle.dump(calculations, f)

    