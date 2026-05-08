import torch
import torch.nn as nn
from torch.optim import AdamW
from tokenizer import Tokenizer
from transformer import Transformer
import os

def load_all_data():
    from data import training_data
    all_data = list(training_data)

    if os.path.exists("real_data.txt"):
        with open("real_data.txt", "r") as f:
            lines = f.readlines()
        real_sentences = [
            l.strip().strip('",').strip()
            for l in lines
            if l.strip() and len(l.strip()) > 5
        ]
        all_data.extend(real_sentences)
        print(f"Personal data:  {len(training_data)} sentences")
        print(f"Real data:      {len(real_sentences)} sentences")
    
    print(f"Total data:     {len(all_data)} sentences")
    return all_data

def train():
    # Load all data
    training_data = load_all_data()

    # Setup tokenizer
    tok = Tokenizer()
    tok.build_vocab(training_data)
    tok.save()

    vocab_size = len(tok.word2idx)
    print(f"Vocab size: {vocab_size}")

    # Build model (bigger for more data!)
    model = Transformer(
        vocab_size=vocab_size,
        embed_dim=512,
        num_heads=8,
        num_layers=12,
        max_seq_len=512
    )

    # Optimizer
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
    epochs = 1000
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

        # Save best model automatically!
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model_state': model.state_dict(),
                'vocab_size': vocab_size,
                'embed_dim': 512,
                'num_heads': 8,
                'num_layers': 12,
            }, 'model.pt')  # always overwrites ✅

        if epoch % 50 == 0:
            print(f"Epoch {epoch:4d} | Loss: {avg_loss:.4f} | Best: {best_loss:.4f}")

    print("-" * 40)
    print(f"Training done! Best loss: {best_loss:.4f}")
    print("Model saved to model.pt ✅")

    # Test generation
    print("\n--- Testing Generation ---")
    model.eval()

    def generate(prompt, max_tokens=20, temperature=0.7):
        tokens = tok.encode(prompt)
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

        return tok.decode(generated)

    prompts = [
        "hello",
        "who created you",
        "what is python",
        "who is jonje",
        "hii bro",
    ]

    for prompt in prompts:
        result = generate(prompt)
        print(f"Input:  '{prompt}'")
        print(f"Output: '{result}'")
        print()

if __name__ == "__main__":
    train()
