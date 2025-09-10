import torch
import torch.nn.functional as F

def generate_text(model, starting_words, word_to_ix, idx_to_word, 
                  max_length=50, temperature=1.0, device="cpu"):

    model.eval()
    words = starting_words[:]

    # μετατροπή σε indices
    input_ids = [word_to_ix.get(w, word_to_ix["<UNK>"]) for w in starting_words]
    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)

    hidden = model.init_hidden(1).to(device)

    # αρχικό forward με το seed
    with torch.no_grad():
        for i in range(len(starting_words)-1):
            _, hidden = model(input_tensor[:, i].unsqueeze(1), hidden)

    next_input = input_tensor[:, -1].unsqueeze(1)

    for _ in range(max_length):
        output, hidden = model(next_input, hidden)
        
        # softmax
        probs = F.softmax(output / temperature, dim=-1).squeeze()

        # ignore <UNK>
        unk_idx = word_to_ix.get("<UNK>")
        if unk_idx is not None:
            probs[unk_idx] = 0.0
            probs = probs / probs.sum()  # normalize

        # sampling
        next_word_idx = torch.multinomial(probs, 1).item()
        next_word = idx_to_word[next_word_idx]

        words.append(next_word)

        if next_word == "<EOS>":
            break

        next_input = torch.tensor([[next_word_idx]], dtype=torch.long).to(device)

    return " ".join(words)
