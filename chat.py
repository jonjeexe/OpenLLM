import torch
import torch.nn as nn
from tokenizer import Tokenizer
from transformer import Transformer

def load_model():
    tok = Tokenizer()
    tok.load('tokenizer.json')

    checkpoint = torch.load('model.pt', weights_only=True)

    model = Transformer(
        vocab_size=checkpoint['vocab_size'],
        embed_dim=checkpoint['embed_dim'],
        num_heads=checkpoint['num_heads'],
        num_layers=checkpoint['num_layers'],
    )
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model, tok

def generate(model, tok, prompt, max_tokens=20, temperature=0.7):
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

if __name__ == "__main__":
    print("Loading your LLM...")
    model, tok = load_model()

    print("=" * 40)
    print("  YOUR PERSONAL LLM IS READY! 🤖")
    print("  Type 'quit' to exit")
    print("=" * 40)

    while True:
        user_input = input("\nYou: ").strip()

        if not user_input:
            continue

        if user_input.lower() == 'quit':
            print("Bye! 👋")
            break

        response = generate(model, tok, user_input)
        print(f"LLM: {response}")
