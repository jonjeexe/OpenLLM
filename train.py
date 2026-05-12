import torch
import torch.nn as nn
from torch.optim import AdamW
from tokenizer import Tokenizer
from transformer import Transformer
import os
import json

def load_all_data():
    from data import training_data

    # Convert built-in JSONL pairs to "instruction <SEP> response" strings
    all_data = []
    for item in training_data:
        combined = item["instruction"] + " <SEP> " + item["response"]
        all_data.append(combined)

    print(f"Built-in data:  {len(training_data)} pairs")

    # Load external JSONL file if exists
    if os.path.exists("real_data.jsonl"):
        jsonl_count = 0
        with open("real_data.jsonl", "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    instruction = item.get("instruction", "").strip()
                    response = item.get("response", "").strip()
                    if instruction and response:
                        combined = instruction + " <SEP> " + response
                        all_data.append(combined)
                        jsonl_count += 1
                except json.JSONDecodeError:
                    continue
        print(f"JSONL data:     {jsonl_count} pairs")

    # Also support legacy real_data.txt (plain text fallback)
    if os.path.exists("real_data.txt"):
        txt_count = 0
        with open("real_data.txt", "r") as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line and len(line) > 5:
                all_data.append(line)
                txt_count += 1
        print(f"TXT data:       {txt_count} sentences")

    print(f"Total data:     {len(all_data)} sequences")
    return all_data


def train():
    training_data = load_all_data()

    # Setup tokenizer — add <SEP> as special token
    tok = Tokenizer()
    tok.special_tokens['<SEP>'] = 4
    tok.word2idx['<SEP>'] = 4
    tok.idx2word[4] = '<SEP>'

    tok.build_vocab(training_data)
    tok.save()

    vocab_size = len(tok.word2idx)
    print(f"Vocab size: {vocab_size}")

    # Build model
    model = Transformer(
        vocab_size=vocab_size,
        embed_dim=256,
        num_heads=8,
        num_layers=6,
        max_seq_len=256
    )

    if os.path.exists('model.pt'):
        checkpoint = torch.load('model.pt')
        model.load_state_dict(checkpoint['model_state'], strict=False)
        print("✅ Loaded existing model, old knowledge preserved")

    optimizer = AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss(
        ignore_index=tok.special_tokens['<PAD>']
    )

    # Prepare tokens
    all_tokens = []
    for text in training_data:
        tokens = tok.encode(text)
        if len(tokens) >= 2:
            all_tokens.append(tokens)

    print(f"Training sequences: {len(all_tokens)}")
    print("-" * 40)

    # Training loop
    epochs = 500
    model.train()
    best_loss = float('inf')

    for epoch in range(epochs):
        total_loss = 0
        count = 0

        for tokens in all_tokens:
            t = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
            inputs = t[:, :-1]
            targets = t[:, 1:]

            logits = model(inputs)
            loss = criterion(
                logits.reshape(-1, vocab_size),
                targets.reshape(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            count += 1

        avg_loss = total_loss / count

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model_state': model.state_dict(),
                'vocab_size': vocab_size,
                'embed_dim': 256,
                'num_heads': 8,
                'num_layers': 6,
            }, 'model.pt')

        if epoch % 50 == 0:
            print(f"Epoch {epoch:4d} | Loss: {avg_loss:.4f} | Best: {best_loss:.4f}")

    print("-" * 40)
    print(f"Training done! Best loss: {best_loss:.4f}")
    print("Model saved to model.pt ✅")

    # Test generation
    print("\n--- Testing Generation ---")
    model.eval()

    def generate(prompt, max_tokens=30, temperature=0.7):
        # Append <SEP> to prompt so model knows to generate a response
        full_prompt = prompt + " <SEP>"
        tokens = tok.encode(full_prompt)
        if tokens[-1] == tok.special_tokens['<EOS>']:
            tokens = tokens[:-1]
        generated = list(tokens)

        with torch.no_grad():
            for _ in range(max_tokens):
                t = torch.tensor(generated).unsqueeze(0)
                logits = model(t)
                last_logits = logits[0, -1] / temperature
                probs = torch.softmax(last_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                if next_token == tok.special_tokens['<EOS>']:
                    break
                generated.append(next_token)

        # Only return tokens AFTER <SEP>
        sep_id = tok.special_tokens.get('<SEP>', None)
        if sep_id and sep_id in generated:
            sep_pos = generated.index(sep_id)
            generated = generated[sep_pos + 1:]

        return tok.decode(generated)

    prompts = [
        "hello",
        "who created you",
        "what is python",
        "who is jonje",
        "hii bro",
        "i need help",
    ]

    for prompt in prompts:
        result = generate(prompt)
        print(f"Input:  '{prompt}'")
        print(f"Output: '{result}'")
        print()

if __name__ == "__main__":
    train()
