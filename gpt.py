import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import Parameter
import sys
import argparse
import os

parser = argparse.ArgumentParser(description='GPT Language Model')
parser.add_argument('--path', type=str, help='Path to the input file')
parser.add_argument('--parameters', type=int, help='Hyperparameters set')
parser.add_argument('--model-path', type=str, help='Path to the model file')
parser.add_argument('--no-train', type=str, help='Skip training')
parser.add_argument('--log-path', type=str, help='Path to the log file')
parser.add_argument('--melody-path', type=str, help='Path to the melody file')
parser.add_argument('--data-type', type=str, help='Type of data')
args = parser.parse_args()

if args.no_train != "True":
    log_file = args.log_path
    with open(log_file, 'w') as f:
        f.write('')

# hyperparameters
# Original set of hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

# Hyperparameters set 1
if args.parameters == 6:
    batch_size = 128
    block_size = 16
    max_iters = 500
    eval_interval = 50
    learning_rate = 2e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_iters = 20
    n_embd = 128
    n_head = 4 
    n_layer = 4 
    dropout = 0.05
# ------------

# Hyperparameters set 2
if args.parameters == 5:
    batch_size = 128
    block_size = 64
    max_iters = 500
    eval_interval = 50
    learning_rate = 2e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_iters = 20
    n_embd = 128
    n_head = 3
    n_layer = 3
    dropout = 0.05

# Hyperparameters set 3
if args.parameters == 4:
    batch_size = 128
    block_size = 32
    max_iters = 500
    eval_interval = 50
    learning_rate = 2e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_iters = 50
    n_embd = 172
    n_head = 2
    n_layer = 2
    dropout = 0.05
    
# Hyperparameters set 4
if args.parameters == 1:
    batch_size = 64
    block_size = 4
    max_iters = 20
    eval_interval = 2
    learning_rate = 1e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_iters = 1
    n_embd = 8
    n_head = 1
    n_layer = 1
    dropout = 0.1
    
# Hyperparameters set 5
if args.parameters == 2:
    batch_size = 64
    block_size = 4
    max_iters = 100
    eval_interval = 10
    learning_rate = 1e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_iters = 10
    n_embd = 32
    n_head = 4
    n_layer = 1
    dropout = 0.1
    
# Hyperparameters set 6
if args.parameters == 3:
    batch_size = 64
    block_size = 32
    max_iters = 400
    eval_interval = 40
    learning_rate = 1e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_iters = 40
    n_embd = 32
    n_head = 4
    n_layer = 1
    dropout = 0.1
    
if args.data_type == "fine_tune":
    learning_rate *= 2

def add_noise(data, chars):
    augmented = []
    for char in data:
        if torch.rand(1).item() < 0.2:
            continue 
        augmented.append(char)
        if torch.rand(1).item() < 0.2:
            augmented.append(stoi[chars[torch.randint(len(chars), (1,)).item()]])
    return torch.tensor(augmented, dtype=torch.long)

def pattern_stretch_shrink(data):
    stretched = []
    patterns = {}
    pattern_length = 2
    for i in range(len(data)):
        if i != len(data) - 1:
            pattern = data[i:i + pattern_length - 1]
            if pattern in patterns:
                patterns[pattern] += 1
            else:
                patterns[pattern] = 1
    
    pattern_sum = sum(patterns.values())        
    pattern_avg = pattern_sum / len(patterns)
    patterns_to_remove = []
    for pattern in patterns:
        if patterns[pattern] < pattern_avg:
            patterns_to_remove.append(pattern)
    for pattern in patterns_to_remove:
        patterns.pop(pattern)
        
    for i in range(len(data)):
        stretched.append(data[i])
        if i != len(data) - 1:
            pattern = data[i:i + pattern_length - 1]
            if pattern in patterns:
                stretched.append(data[i + 1])
                stretched.append(data[i])
            
    return stretched
    
    
        
        
        
    
torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

input_file = args.path

with open(input_file, 'r', encoding='utf-8') as f:
    text = f.read()

text = pattern_stretch_shrink(text)

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
data = torch.cat((data, add_noise(data, chars)))
# data = torch.cat((data, add_noise(data, chars)))

n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
        
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=False),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=n_embd, out_channels=n_embd, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=n_embd, out_channels=n_embd, kernel_size=5, padding=2)
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            if args.data_type != 'timing':
                loss_a = F.cosine_embedding_loss(logits, F.one_hot(targets, num_classes=vocab_size).float(), torch.ones(logits.size(0), device=device))
                loss_b = F.cross_entropy(logits, targets)
                loss = loss_a * 0.7 + loss_b * 0.3
            else:
                loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            
            # Add temperature to the logits
            logits /= 0.3
            
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx_next = torch.clamp(idx_next, 0, vocab_size - 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


if args.no_train == None:    
    model = GPTLanguageModel()
    
    
    if os.path.exists(args.model_path) and args.data_type == "fine_tune":
        print(f"Loading model from {args.model_path}...")
        # Load state_dict
        checkpoint = torch.load(args.model_path)
        
        # Adjust model to fit checkpoint vocabulary size
        checkpoint_vocab_size = checkpoint['token_embedding_table.weight'].size(0)
        model_vocab_size = model.token_embedding_table.weight.size(0)
        
        if checkpoint_vocab_size != model_vocab_size:
            print(f"Adjusting model vocabulary size from {model_vocab_size} to {checkpoint_vocab_size}")
            # Update embedding and output layers
            model.token_embedding_table = nn.Embedding(checkpoint_vocab_size, model.token_embedding_table.weight.size(1))
            model.lm_head = nn.Linear(model.lm_head.weight.size(1), checkpoint_vocab_size)

        # Load updated state_dict
        model.load_state_dict(checkpoint)
        print("Model loaded successfully.")
    else:
        print("Training model...")
    m = model.to(device)
    
    # print the number of parameters in the model
    with open(log_file, 'a') as f:
        f.write(f"{sum(p.numel() for p in m.parameters())/1e6} M parameters\n")

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Write the vocabulary size to the log
    with open(log_file, 'a') as f:
        f.write(f"Vocabulary size: {vocab_size}\n")
        
    for iter in range(max_iters):

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()

            # Append losses during training
            with open(log_file, 'a') as f:
                f.write(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}\n")

        # sample a batch of data
        xb, yb = get_batch('train')

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    if args.melody_path != None:
        generated_text = decode(m.generate(context, max_new_tokens=5000)[0].tolist())
        # print(generated_text)
        with open(args.melody_path, 'w') as f:
            f.write(generated_text)

    # Save the model
    torch.save(model.state_dict(), args.model_path)
    print(f"Model saved to {args.model_path}")
    
    most_frequent_char = max(set(text), key=text.count)
    most_frequent_char_idx = stoi[most_frequent_char]
    
    melody_path = args.melody_path
    melody_leaf = melody_path.split('/')[-1]
    melody_leaf = 'baseline_' + melody_leaf
    baseline_path = os.path.join('/'.join(melody_path.split('/')[:-1]), melody_leaf)

    with open(baseline_path, 'w') as f:
        baseline_context = torch.zeros((1, 1), dtype=torch.long, device=device).fill_(most_frequent_char_idx)
        baseline_generated_text = decode([baseline_context.squeeze().item()])
        f.write(baseline_generated_text)



if args.no_train == "True":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = GPTLanguageModel().to(device)

    # Load the checkpoint
    checkpoint = torch.load(args.model_path, map_location=device)

    # Retrieve embedding and output weights from checkpoint
    embedding_weight = checkpoint['token_embedding_table.weight']
    output_weight = checkpoint['lm_head.weight']
    output_bias = checkpoint['lm_head.bias']

    # Check for vocabulary size mismatch
    checkpoint_vocab_size = embedding_weight.shape[0]
    if checkpoint_vocab_size != vocab_size:
        print("Adjusting model for different vocab size...")

        # Adjust token embedding layer
        new_embedding_weight = torch.zeros((vocab_size, n_embd), device=device)
        new_embedding_weight[:min(vocab_size, checkpoint_vocab_size), :] = embedding_weight[:min(vocab_size, checkpoint_vocab_size), :]
        model.token_embedding_table = nn.Embedding(vocab_size, n_embd).to(device)
        model.token_embedding_table.weight = Parameter(new_embedding_weight)

        # Adjust lm_head output layer weights
        new_output_weight = torch.zeros((vocab_size, n_embd), device=device)
        new_output_weight[:min(vocab_size, checkpoint_vocab_size), :] = output_weight[:min(vocab_size, checkpoint_vocab_size), :]
        model.lm_head = nn.Linear(n_embd, vocab_size).to(device)
        model.lm_head.weight = Parameter(new_output_weight)

        # Adjust lm_head output layer biases
        new_output_bias = torch.zeros(vocab_size, device=device)
        new_output_bias[:min(vocab_size, checkpoint_vocab_size)] = output_bias[:min(vocab_size, checkpoint_vocab_size)]
        model.lm_head.bias = Parameter(new_output_bias)
    else:
        # No mismatch, load state dictionary directly
        model.load_state_dict(checkpoint)

    model.eval()  # Set model to evaluation mode
    print(f"Model loaded from {args.model_path}")

    # Prepare the validation dataset
    with open(args.path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Process the text data
    data = torch.tensor(encode(text), dtype=torch.long)
    val_data = data

    loss = estimate_loss()
    val_loss = loss['val'].item()
  
    results_log = args.log_path
    with open(results_log, 'w') as f:
        f.write('Vocabulary size: {}\n'.format(vocab_size))
        f.write("Trained Model:\n")
        f.write(f"Validation Loss: {val_loss:.4f}\n")

    val_data = val_data.to(device)

    # Identify the most frequent character and its index
    most_frequent_char = max(set(text), key=text.count)
    most_frequent_char_idx = stoi[most_frequent_char]

    # Generate dummy predictions (all predictions set to the index of the most frequent character)
    dummy_predictions = torch.full((len(val_data),), most_frequent_char_idx, dtype=torch.float, device=device)

    # Calculate dummy loss
    # Ensure all tensors (both predictions and targets) are on the same device and of compatible types
    dummy_loss = F.cross_entropy(dummy_predictions.unsqueeze(0), val_data.unsqueeze(0).float()).item()

    # Log the dummy classifier loss
    with open(results_log, 'a') as f:
        f.write("Baseline Dummy Classifier:\n")
        f.write(f"Validation Loss: {dummy_loss:.4f}\n")
