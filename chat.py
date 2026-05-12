import torch
import torch.nn as nn
from tokenizer import Tokenizer
from transformer import Transformer

def load_model():
    tok = Tokenizer()
    tok.load('tokenizer.json')

    # Register <SEP> token if not present
    if '<SEP>' not in tok.special_tokens:
        tok.special_tokens['<SEP>'] = 4
        tok.word2idx['<SEP>'] = 4
        tok.idx2word[4] = '<SEP>'

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

def generate(model, tok, prompt, max_tokens=30, temperature=0.7):
    # Add <SEP> so model knows to generate response
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

    # Only decode tokens AFTER <SEP>
    sep_id = tok.special_tokens.get('<SEP>', None)
    if sep_id and sep_id in generated:
        sep_pos = generated.index(sep_id)
        generated = generated[sep_pos + 1:]

    return tok.decode(generated)

if __name__ == "__main__":
    print("Loading Yomi...")
    model, tok = load_model()

    print("=" * 40)
    print("  YOMI - Personal AI by jonje")
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
        print(f"Yomi: {response}")
