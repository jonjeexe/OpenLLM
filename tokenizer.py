import re
import json
from collections import Counter

class Tokenizer:
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.word2idx = {}
        self.idx2word = {}
        # Special tokens
        self.special_tokens = {
            '<PAD>': 0,   # padding
            '<UNK>': 1,   # unknown word
            '<BOS>': 2,   # beginning of sentence
            '<EOS>': 3,   # end of sentence
        }
        self.word2idx.update(self.special_tokens)
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def clean_text(self, text):
        # lowercase everything
        text = text.lower()
        # keep letters, numbers, basic punctuation
        text = re.sub(r"[^a-z0-9\s\.,!?']", '', text)
        # remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def build_vocab(self, texts):
        print("Building vocabulary...")
        all_words = []

        for text in texts:
            text = self.clean_text(text)
            words = text.split()
            all_words.extend(words)

        # Count word frequencies
        word_counts = Counter(all_words)

        # Take most common words up to vocab_size
        most_common = word_counts.most_common(
            self.vocab_size - len(self.special_tokens)
        )

        # Build vocab
        idx = len(self.special_tokens)
        for word, count in most_common:
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            idx += 1

        print(f"Vocabulary built! Size: {len(self.word2idx)} words")

    def encode(self, text):
        text = self.clean_text(text)
        words = text.split()
        tokens = [self.special_tokens['<BOS>']]
        for word in words:
            if word in self.word2idx:
                tokens.append(self.word2idx[word])
            else:
                tokens.append(self.special_tokens['<UNK>'])
        tokens.append(self.special_tokens['<EOS>'])
        return tokens

    def decode(self, tokens):
        words = []
        for token in tokens:
            if token in self.idx2word:
                word = self.idx2word[token]
                if word not in self.special_tokens:
                    words.append(word)
        return ' '.join(words)

    def save(self, path='tokenizer.json'):
        data = {
            'vocab_size': self.vocab_size,
            'word2idx': self.word2idx,
            'idx2word': {str(k): v for k, v in self.idx2word.items()}
        }
        with open(path, 'w') as f:
            json.dump(data, f)
        print(f"Tokenizer saved to {path}")

    def load(self, path='tokenizer.json'):
        with open(path, 'r') as f:
            data = json.load(f)
        self.vocab_size = data['vocab_size']
        self.word2idx = data['word2idx']
        self.idx2word = {int(k): v for k, v in data['idx2word'].items()}
        print(f"Tokenizer loaded! Vocab size: {len(self.word2idx)}")

# ---- TEST IT ----
if __name__ == "__main__":
    tokenizer = Tokenizer(vocab_size=10000)

    sample_texts = [
        "Python is a programming language used for AI and machine learning",
        "Neural networks are the brain of artificial intelligence",
        "Deep learning is a subset of machine learning",
        "What is Python? Python is an easy programming language",
        "How does AI work? AI learns from data and finds patterns",
    ]

    tokenizer.build_vocab(sample_texts)

    test = "Python is used for AI"
    encoded = tokenizer.encode(test)
    decoded = tokenizer.decode(encoded)

    print(f"\nOriginal:  {test}")
    print(f"Encoded:   {encoded}")
    print(f"Decoded:   {decoded}")

    tokenizer.save()
    print("\nTokenizer works perfectly!")
