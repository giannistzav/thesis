from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import pickle
from entropyfunc import entropy, ttr

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()
path_to_sentence_file='sentencesfortextgen\sentences-5words.txt'
max_length=50

savefile='results\entropy-ttr-gpt2.txt'
modelname='gpt2-update'

with open(path_to_sentence_file, 'rb') as f:
    all_words = pickle.load(f)
# input


entr=0
typetr=0


for i in range (0, len(all_words)):
    inputs = tokenizer(all_words[i], return_tensors="pt", padding=True, truncation=True)
    
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    t=generated_text.split()
    if t[-1]=='<EOS>':
        t=t[:-1]
        generated_text=' '.join(t)
    
    entr+=entropy(generated_text)
    typetr+=ttr(generated_text)


calc_entr=entr/len(all_words)
calc_ttr=typetr/len(all_words)

calculations=f'model: {modelname} \n length of sentences :{max_length} \n calculated entropy: {calc_entr} \n calculated ttr: {calc_ttr} .'
with open(savefile, 'wb') as f:
        pickle.dump(calculations, f)
