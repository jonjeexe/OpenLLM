import torch
import torch.nn as nn
from torch.optim import AdamW
from tokenizer import Tokenizer
from transformer import Transformer
from data import training_data

def train():
    # Setup
    tok = Tokenizer()
    tok.build_vocab(training_data)
    tok.save()

    vocab_size = len(tok.word2idx)
    print(f"Vocab size: {vocab_size}")

    # Build model
    model = Transformer(
        vocab_size=vocab_size,
        embed_dim=128,
        num_heads=4,
        num_layers=4,
        max_seq_len=256
    )

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss(
        ignore_index=tok.special_tokens['<PAD>']
    )

    # Prepare training data
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

    for epoch in range(epochs):
        total_loss = 0
        count = 0

        for tokens in all_tokens:
            # Convert to tensor
            t = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)

            # Input = all except last, Target = all except first
            inputs = t[:, :-1]
            targets = t[:, 1:]

            # Forward pass
            logits = model(inputs)

            # Calculate loss
            loss = criterion(
                logits.reshape(-1, vocab_size),
                targets.reshape(-1)
            )

            # Real backpropagation!
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            count += 1

        avg_loss = total_loss / count

        if epoch % 50 == 0:
            print(f"Epoch {epoch:4d} | Loss: {avg_loss:.4f}")

    print("-" * 40)
    print(f"Training done! Final loss: {avg_loss:.4f}")

    # Save model
    torch.save({
        'model_state': model.state_dict(),
        'vocab_size': vocab_size,
        'embed_dim': 128,
        'num_heads': 4,
        'num_layers': 4,
    }, 'model.pt')
    print("Model saved to model.pt")

    # Test generation
    print("\n--- Testing Generation ---")
    model.eval()

    def generate(prompt, max_tokens=15, temperature=0.7):
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
        "python is",
        "neural networks",
        "artificial intelligence",
        "machine learning",
        "programming is",
    ]

    for prompt in prompts:
        result = generate(prompt)
        print(f"Input:  '{prompt}'")
        print(f"Output: '{result}'")
        print()

if __name__ == "__main__":
    train()
