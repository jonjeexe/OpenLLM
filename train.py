import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from tokenizer import Tokenizer
from transformer import Transformer
import os
import glob
import json
import random

# Use all CPUs!
torch.set_num_threads(4)

# ==========================================
# DATASET CLASS
# ==========================================
class LLMDataset(Dataset):
    def __init__(self, all_tokens, max_len=64):
        self.data = []
        for tokens in all_tokens:
            # Truncate long sequences
            if len(tokens) > max_len:
                tokens = tokens[:max_len]
            if len(tokens) >= 2:
                self.data.append(tokens)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    # Pad sequences to same length in batch
    max_len = max(len(x) for x in batch)
    padded = []
    for tokens in batch:
        padded.append(tokens + [0] * (max_len - len(tokens)))
    return torch.tensor(padded, dtype=torch.long)

# ==========================================
# LOAD ALL DATA
# ==========================================
def load_all_data():
    all_data = []

    # Load from data.py (JSONL format)
    try:
        from data import training_data
        for item in training_data:
            if isinstance(item, dict):
                instruction = item.get('instruction', '')
                response = item.get('response', '')
                if instruction and response:
                    all_data.append(f"{instruction} <SEP> {response}")
            else:
                all_data.append(str(item))
        print(f"Personal data:  {len(training_data)} pairs")
    except Exception as e:
        print(f"Warning: {e}")

    # Load JSONL files
    jsonl_files = glob.glob("*.jsonl")
    for file in jsonl_files:
        count = 0
        with open(file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    instruction = item.get('instruction', '')
                    response = item.get('response', item.get('output', ''))
                    if instruction and response:
                        # Clean
                        instruction = instruction.strip().lower()[:100]
                        response = response.strip().lower()[:100]
                        all_data.append(f"{instruction} <SEP> {response}")
                        count += 1
                except:
                    continue
        print(f"Loaded {file}: {count} pairs")

    # Load TXT files
    txt_files = glob.glob("*.txt")
    for file in txt_files:
        with open(file, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        sentences = [
            l.strip().lower()
            for l in lines
            if l.strip() and len(l.strip().split()) > 2
        ]
        all_data.extend(sentences)
        print(f"Loaded {file}: {len(sentences)} sentences")

    # Remove duplicates
    all_data = list(dict.fromkeys(all_data))
    print(f"Total unique:   {len(all_data)} sequences")
    return all_data

# ==========================================
# TRAIN
# ==========================================
def train():
    training_data = load_all_data()

    # Build tokenizer
    tok = Tokenizer()
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

    # Load existing if available
    if os.path.exists('model.pt'):
        try:
            checkpoint = torch.load('model.pt', weights_only=True)
            if checkpoint['vocab_size'] == vocab_size:
                model.load_state_dict(checkpoint['model_state'], strict=False)
                print("✅ Loaded existing model!")
            else:
                print("⚠️ Vocab changed - fresh start!")
        except:
            print("⚠️ Fresh start!")

    # Prepare tokens
    all_tokens = []
    for text in training_data:
        tokens = tok.encode(text)
        if len(tokens) >= 2:
            all_tokens.append(tokens)

    print(f"Training sequences: {len(all_tokens)}")

    # DataLoader with batching!
    dataset = LLMDataset(all_tokens, max_len=64)
    dataloader = DataLoader(
        dataset,
        batch_size=32,        # 32 sentences at once!
        shuffle=True,         # shuffle every epoch
        num_workers=2,        # parallel loading
        collate_fn=collate_fn
    )

    print(f"Batches per epoch: {len(dataloader)}")
    print("-" * 40)

    # Optimizer + scheduler
    epochs = 100
    optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    model.train()
    best_loss = float('inf')

    for epoch in range(epochs):
        total_loss = 0
        count = 0

        for batch in dataloader:
            inputs = batch[:, :-1]
            targets = batch[:, 1:]

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

        scheduler.step()
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

        if epoch % 10 == 0:
            lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Best: {best_loss:.4f} | LR: {lr:.6f}")

    print("-" * 40)
    print(f"Training done! Best loss: {best_loss:.4f}")
    print("Model saved to model.pt ✅")

    # Test generation
    print("\n--- Testing Generation ---")
    model.eval()

    def generate(prompt, max_tokens=30, temperature=0.6):
        full_prompt = f"{prompt.lower()} <SEP>"
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

                # Top-P sampling
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=0)
                mask = cumsum - sorted_probs > 0.95
                sorted_probs[mask] = 0
                if sorted_probs.sum() == 0:
                    sorted_probs = probs
                    sorted_idx = torch.arange(len(probs))
                sorted_probs = sorted_probs / sorted_probs.sum()
                next_token = sorted_idx[torch.multinomial(sorted_probs, 1)].item()

                if next_token == tok.special_tokens['<EOS>']:
                    break
                generated.append(next_token)

        # Extract only response part (after <SEP>)
        full_text = tok.decode(generated)
        if '<sep>' in full_text:
            response = full_text.split('<sep>')[-1].strip()
        else:
            response = full_text

        return response

    prompts = ["hello", "who created you", "what is python", "hii bro", "who is jonje"]
    for prompt in prompts:
        result = generate(prompt)
        print(f"Input:  '{prompt}'")
        print(f"Output: '{result}'")
        print()

if __name__ == "__main__":
    train()
                
