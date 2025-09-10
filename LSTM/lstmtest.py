import torch
from entropyfunc import sample_next_token

def generate_text(model, word_to_ix, idx_to_word, seed_text, next_words=20, max_seq_len=10, device="cpu"):
    model.eval()


    words = seed_text.strip().split()

    for _ in range(next_words):
        context = words[-(max_seq_len - 1):]


        input_indices = [
            word_to_ix.get(w, word_to_ix["<OOV>"]) for w in context
        ]

        #Padding
        if len(input_indices) < max_seq_len - 1:
            input_indices = [word_to_ix["<OOV>"]] * (max_seq_len - 1 - len(input_indices)) + input_indices

        input_tensor = torch.LongTensor(input_indices).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            logits = output.squeeze(0)
            next_word_id = sample_next_token(
                logits,
                strategy="top-p", 
                p=0.9, 
                temperature=1.0
            ).item()


            predicted_word = idx_to_word.get(next_word_id, "<OOV>")

        words.append(predicted_word)

    return " ".join(words)
