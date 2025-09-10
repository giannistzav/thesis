from entropyfunc import entropy, ttr, self_bleu
import torch
import pickle
import torch.nn as nn

from lstmtest import generate_text
from lstmmodel import LSTMModel

savefile='results\entropy-ttr-lstm.txt'
modelname='lstm150k-final'
path_to_sentence_file='sentencesfortextgen\sentences-5words.txt'
vocabfile='vocab_lstm.pkl'
max_length=200

with open(path_to_sentence_file, 'rb') as f:
    all_words = pickle.load(f)

with open(vocabfile, 'rb') as f:
    vocab_dict_lstm = pickle.load(f)

word_index = vocab_dict_lstm["word_index"]
idx_to_word = vocab_dict_lstm["idx_to_word"]
total_words = vocab_dict_lstm["total_words"]

# load state_dict
model = LSTMModel(total_words, 
    embedding_dim=128, 
    hidden_dim=100,
    num_layers=2,
    dropout_rate=0.5
    )

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model.load_state_dict(torch.load(modelname+'.pth',map_location=device))
model.to(device)

entr=0
typetr=0
selfbleuscore=0
all_generated_text=[]
for i in range (0, len(all_words)):

    starting_words = all_words[i]

    generated_text=generate_text(model,
        word_index,
        idx_to_word,
        starting_words,
        next_words=max_length,
        max_seq_len=10,
        device=device

    )
    all_generated_text.append(generated_text)

    t=generated_text.split()
    if t[-1]=='<EOS>':
        t=t[:-1]
        generated_text=' '.join(t)
    

    entr+=entropy(generated_text)
    typetr+=ttr(generated_text)
    

calc_entr = entr/len(all_words)
calc_ttr = typetr/len(all_words)

calculations=f'\nmodel: {modelname}\nlength of sentences:{max_length}\ncalculated entropy:{calc_entr}\ncalculated ttr:{calc_ttr}\n\n'
with open(savefile, 'ab') as f:
        pickle.dump(calculations, f)

    