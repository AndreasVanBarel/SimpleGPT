# Parameters
batch_size = B = 64 # Number of examples to process in parallel
block_size = T = 128 # Context length for predictions
n_eds = C = 256 # number of embedding dimensions (C for channels)
n_heads = 8 # Each head then has headsize equal to C//n_heads = 256/8 = 32
n_layers = 6
dropout = 0.2 # To prevent overfitting
opt_steps = 5000 # number of training steps
learning_rate = 4e-4
estimation_evals = 250 # Nb of samples for estimation of loss
eval_interval = 250 # Every eval_interval training steps, the training and validation loss are estimated
n_gen = 2500 # Nb of tokens to generate and print to illustrate the model

# timestamps
import time

# Pytorch
import torch
import torch.nn as nn
from torch.nn import functional
torch.manual_seed(1604)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# Open data to learn from
with open("data.txt", "r", encoding='utf-8') as f:
    text = f.read()

print(f"Data length: {len(text)}")

# Tokens that we will work with
tokens = sorted(list(set(text)));
n_tokens = len(tokens);
print(f"Number of tokens: {n_tokens}")

# Encoding and decoding the characters
token_to_int = {t:i for i,t in enumerate(tokens)}
int_to_token = {i:t for i,t in enumerate(tokens)}
encode = lambda s : [token_to_int[t] for t in s]
decode = lambda l : ''.join([int_to_token[i] for i in l])

# Data will be the encoded text, wrapped in a Pytorch Tensor.
data = torch.tensor(encode(text))

# split data into training and validation
n = int(0.9*len(data))
data_train = data[:n]
data_test = data[n:]

# Produce a batch of data
def get_batch(split='train'):
    match split:
        case 'train':
            data = data_train 
        case 'val' | 'test':
            data = data_test
        case _:
            raise Exception("split should be 'train' or 'val'")

    inds = torch.randint(len(data)-T, (B,))
    x = torch.stack([data[i:i+T] for i in inds])
    y = torch.stack([data[i+1:i+1+T] for i in inds])
    return x.to(device), y.to(device)

# Self-attention head
class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_eds, head_size, bias=False)
        self.query = nn.Linear(n_eds, head_size, bias=False)
        self.value = nn.Linear(n_eds, head_size, bias=False)
        # mask such that information does not flow towards past
        self.register_buffer('mask', torch.triu(torch.ones(T,T), 1)) 
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape # (batch_size, block_size, n_eds)
        k = self.key(x) # (B,T,H) (H: head_size)
        q = self.query(x) # (B,T,H)
        v = self.value(x) # (B,T,H)
        _,_,H = k.shape

        # compute affinities
        W = q @ k.transpose(-2, -1) # (B,T,H) * (B,H,T) -> (B,T,T)
        W = W * H**-0.5
        W = W.masked_fill(self.mask[:T,:T] == 1, float('-inf'))
        W = functional.softmax(W, dim=-1)
        W = self.dropout(W)

        out = W @ v # (B,T,T) * (B,T,H) -> (B,T,H)
        return out
    
class MultiHead(nn.Module):

    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_heads*head_size, n_heads*head_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        x = torch.cat([h(x) for h in self.heads], dim=-1)
        x = self.dropout(self.proj(x))
        return x
    
class FeedForward(nn.Module):

    def __init__(self, size):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(size, 4*size),
            nn.ReLU(),
            nn.Linear(4*n_eds, n_eds),
            nn.Dropout(dropout) )
        
    def forward(self, x):
        return self.ff(x)
    
class Block(nn.Module):
    # Transformer block (self-attention followed by computation)

    def __init__(self, n_eds, n_heads):
        super().__init__()
        self.sa_heads = MultiHead(n_heads, n_eds//n_heads)
        self.ff = FeedForward(n_eds)
        self.ln1 = nn.LayerNorm(n_eds)
        self.ln2 = nn.LayerNorm(n_eds)

    def forward(self,x):
        x = x + self.sa_heads(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


# Transformer model
class Transformer(nn.Module):
    
    def __init__(self, n_tokens):
        super().__init__()
        # Embedding table contains logits for next token given the current token
        self.token_embedding_table = nn.Embedding(n_tokens, C)
        self.position_embedding_table = nn.Embedding(T, C)
        # self.transformer_blocks = nn.Sequential(
        #     Block(n_eds, n_heads=4),
        #     Block(n_eds, n_heads=4),
        #     Block(n_eds, n_heads=4),
        #     nn.LayerNorm(n_eds),
        # )
        self.transformer_blocks = nn.Sequential(
            *[Block(n_eds, n_heads) for _ in range(n_layers)]
        )
        self.ln = nn.LayerNorm(n_eds)
        self.lm_head = nn.Linear(C, n_tokens)

    def forward(self, idx, targets=None):
        # idx and targets are (B,T)-tensors
        B, T = idx.shape

        # The token_embeddings are (B,T,C)-tensors
        # The logits are (B,T,n_tokens)-tensors
        token_embeddings = self.token_embedding_table(idx) # (B,T,C)
        r = torch.arange(T, device=device) # makes tensor([0, 1, 2, 3, 4, 5, 6, 7], device='cuda:0')
        position_embeddings = self.position_embedding_table(r) # (T,C)
        x = token_embeddings + position_embeddings # (B,T,C)
        x = self.transformer_blocks(x)
        x = self.ln(x)
        logits = self.lm_head(x) # (B,T,n_tokens)

        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits_view = logits.view(B*T,C)
            targets = targets.view(B*T)
            # print(f'logits_view have shape {logits_view.shape}')
            # print(f'targets have shape {targets.shape}')
            loss = functional.cross_entropy(logits_view, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B,T)
        for _ in range(max_new_tokens):
            B,T = idx.shape
            # idx_trunc = (idx if T <=block_size else idx[:,-block_size:]) # truncates
            idx_trunc = idx[:,-block_size:]

            # Generate logits 
            logits, loss = self(idx_trunc) # logits (B,T,C)
            logits = logits[:,-1,:] # (B,C)
            probs = functional.softmax(logits, dim=1) # (B,C)
            idx_drawn = torch.multinomial(probs, 1) # (B,1)
            idx = torch.cat((idx, idx_drawn), dim = 1) # (B,T+1)

        return idx

m = Transformer(n_tokens)
m = m.to(device)

# counting the number of parameters
def count_parameters(model): 
    return sum(p.numel() for p in model.parameters() if p.requires_grad) 

print(f"Number of parameters: {count_parameters(m)}")

# let the model generate text
def generate_text(length, start = torch.zeros((1,1), dtype=torch.long).to(device)):
    gen = m.generate(start, length)
    text = decode(gen[0].tolist())
    return text

print(f"\nUntrained model says: {generate_text(250)}\n")

# Loss estimation
@torch.no_grad()
def estimate_loss():
    training_mode = m.training # store whether model was in training mode
    m.eval() # Set model to evaluation mode
    outs = {}
    for split in ['train', 'val']:
        losses = torch.zeros(estimation_evals)
        for k in range(estimation_evals):
            x,y = get_batch(split)
            logits, loss = m(x,y)
            losses[k] = loss.tolist()
        
        outs[split] = losses.mean().tolist()
    
    if training_mode==True: 
        m.train() # restore the training mode
        
    return outs
    
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

start_time = time.time()

for step in range(opt_steps):
    x,y = get_batch('train')
    logits, loss = m(x,y)
    optimizer.zero_grad(set_to_none = True)
    loss.backward()
    optimizer.step()
    if step%eval_interval == 0:
        t = time.time()
        losses = estimate_loss()
        print(f'After step {step}, train loss is {losses["train"]:.3f} and val loss is {losses["val"]:.3f} ({t - start_time:.3f}s)')

print(f"\nThe trained model says: {generate_text(n_gen)}\n")
