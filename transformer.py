import torch
import torch.nn as nn
import math

class Transformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_heads=4, 
                 num_layers=4, max_seq_len=256, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim

        # Token + position embeddings
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(num_layers)
        ])

        # Output projection
        self.ln_final = nn.LayerNorm(embed_dim)
        self.output = nn.Linear(embed_dim, vocab_size, bias=False)

        # Count parameters
        total = sum(p.numel() for p in self.parameters())
        print(f"Total parameters: {total:,}")

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)

        # Embeddings
        x = self.token_embed(x) + self.pos_embed(positions)
        x = self.dropout(x)

        # Causal mask
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len)

        # Pass through layers
        memory = torch.zeros_like(x)
        for layer in self.layers:
            x = layer(x, memory, tgt_mask=mask)

        x = self.ln_final(x)
        return self.output(x)
