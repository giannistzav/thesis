from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pickle
from entropyfunc import entropy, ttr

model_path = r"C:\Users\Giannis\visual studio cod\gpt-neo-1.3b"
savefile='results\entropy-ttr-gptneo.txt'
modelname='gptneo-update'
path_to_sentence_file='sentencesfortextgen\sentences-5words.txt'
max_length=100

tokenizer = AutoTokenizer.from_pretrained(model_path)

tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_path)


with open(path_to_sentence_file, 'rb') as f:
    all_words = pickle.load(f)

entr=0
typetr=0

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)


for i in range (0, len(all_words)):
    inputs = tokenizer(all_words[i], return_tensors="pt", padding=True, truncation=True).to(model.device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]


    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id
        )

    out=tokenizer.decode(outputs[0], skip_special_tokens=True)

    t=out.split()
    if t[-1]=='<EOS>':
        t=t[:-1]
        out=' '.join(t)
    
    entr+=entropy(out)
    typetr+=ttr(out)


calc_entr=entr/len(all_words)
calc_ttr=typetr/len(all_words)

calculations=f'model: {modelname} \n length of sentences :{max_length} \ncalculated entropy: {calc_entr} \n calculated ttr: {calc_ttr} .'
with open(savefile, 'ab') as f:
        pickle.dump(calculations, f)
