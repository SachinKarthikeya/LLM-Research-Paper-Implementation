import torch
import torch.nn as nn
import torch.nn.functional as F

# === Data Setup ===
text = "hello world! welcome to the world of language models."
chars = sorted(list(set(text)))
vocab_size = len(chars)

char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

def encode(s): return [char_to_idx[ch] for ch in s]
def decode(l): return ''.join([idx_to_char[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)

# === Model Definition ===
class TinyLLM(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.linear = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.linear(x)
        return x

model = TinyLLM(vocab_size, embed_dim=32)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
loss_fn = nn.CrossEntropyLoss()

# === Training Setup ===
block_size = 4

def get_batch():
    ix = torch.randint(0, len(data) - block_size - 1, (1,))
    x = data[ix:ix + block_size]
    y = data[ix + 1:ix + block_size + 1]
    return x.unsqueeze(0), y.unsqueeze(0)

# === Training Loop ===
for step in range(1000):
    x_batch, y_batch = get_batch()
    logits = model(x_batch)
    loss = loss_fn(logits.view(-1, vocab_size), y_batch.view(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 100 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}")

# === Text Generation ===
context = torch.tensor([char_to_idx['h']], dtype=torch.long).unsqueeze(0)

for _ in range(100):
    logits = model(context)
    last_logits = logits[0, -1]
    probs = F.softmax(last_logits, dim=0)
    next_char_idx = torch.multinomial(probs, num_samples=1)
    context = torch.cat([context, next_char_idx.unsqueeze(0)], dim=1)

print("Generated text:", decode(context[0].tolist()))
