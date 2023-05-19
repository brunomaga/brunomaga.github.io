import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)
device = 'cpu' #torch.device("cuda" if torch.cuda.is_available() else "cpu")

# set the random seed, for reproducibility
torch.manual_seed(42)

# device: where to execute computation
device = 'cpu' #torch.device("cuda" if torch.cuda.is_available() else "cpu")

# how often to do an evaluation step
eval_interval = 100

# number of training iterations
max_iters = 500

# optimizer's learning rate
learning_rate=1e-4

# minibatch size, how many inputs to 'pack' per iteration
batch_size = 3

# block size ie max number of training sequence.
# E.g for input ABCD, we have 4 training examples: A->B, AB->C, ABC->C, ABCD->E
block_size = 4

# size of the embeddings
n_embd = 16

# number of attention heads in the Multi-Attention mechanism
n_head = 6

# number of heads. this is the $d_k$ in the paper formulation of Attn. Mech
head_size = 8

# depth of the network as number of decoder blocks.
# Each block contains a normalization, an attention and a feed forward unit
n_layer = 5

# dropout rate (variable p) for dropout units
dropout = 0.2

# Uncomment for large-scale runs
eval_interval = 100 #evaluation interval
max_iters = 5000
learning_rate=3e-4
batch_size = 128
block_size = 256
n_embd = 300
head_size = 32 #dimensionaly of the attn head
n_layer = 10
dropout = 0.2


#load input data
with open("tinyshakespeare.txt") as f:
    text = f.read()
print("input data loaded. Length of text: ", len(text))

#collect all ordered and unique characters in the text
chars = sorted(list(set(text)))
print("unique chars: ", "".join(chars))
print("length of chars: ", len(chars))

#map characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda x: torch.tensor([stoi[ch] for ch in x], dtype=torch.long) #encode text to integers
decode = lambda x: ''.join([itos[i] for i in x]) #decode integers to text
vocab_size = len(stoi)
print(encode("Hello world"))
print(decode(encode("Hello world").tolist()))
print("character zero is:", decode([0]), "<end>")


# collect input data, break dataset in train/validation
data = encode(text)
n = int(0.9*len(data))
train_data, valid_data = data[:n], data[n:]
print("Train data encoded", data.shape, train_data.shape, valid_data.shape)


def get_batch(source):
  """ get batch of size block_size from source """
  
  # generate `batch_size` random offsets on the data 
  ix = torch.randint(len(source)-block_size, (batch_size,) )
  # collect `batch_size` subsequences of length `block_size` from source, as data and target
  x = torch.stack([source[i:i+block_size] for i in ix])
  # target is just x shifted right (ie the next predicted word)
  y = torch.stack([source[i+1:i+1+block_size] for i in ix])
  return x.to(device), y.to(device)

#test temporal batches
xb, yb = get_batch(train_data)
print("input:\n",xb)
print("target:\n",yb)
for b in range(batch_size): #for every batches
  print(f"\n=== batch {b}:")
  for t in range(block_size): #for each sequence in block
    context = xb[b,:t+1]
    target = yb[b,t]
    print(f"for input {context.tolist()} target is {target.tolist()}")


## test attention mechanism

#X Bag Of Words (xbow)
B, T, C, = 4, 8, 2
x = torch.randn(B,T,C) #shape (B,T,C)

#attention matrix (lower triangular), a mask used to only show previous items to predict next item
wei = torch.tril(torch.ones((T,T), dtype=torch.float32, device=device))
#normalize mask so that it sums to one. keepdim to make broadcast work later
wei /= wei.sum(dim=1, keepdim=True)
out = wei @ x # ie dot product (B,T,T) @ (B,T,C) --> (B,T,C)

print("Uniform attention matrix, implementation 1:")
print(wei)
print(out.shape)

#alternative notation
tril = torch.tril(torch.ones(T,T))
wei = wei.masked_fill(tril==0, float('-inf'))
wei = F.softmax(wei, dim=-1) #equivalent to the normalization above
out = wei @ x

print("Uniform attention matrix, implementation 2:")
print(wei)
print(out.shape)
print(x.shape)


print("NON-Uniform attention matrix:")
#or we can do a self attention like in Transformers
head_size = 16 #number of heads this is the $d_k$ in the paper
key   = nn.Linear(C, head_size, bias=False) 
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)

#every input produces a key and a query (independent of the other inputs)
k = key(x) #shape (B,T, head_size)
q = query(x) #shape (B,T, head_size)
wei = q @ k.transpose(-2, -1) #shape (B,T, head_size) @ (B,head_size,T) --> (B,T,T)
print("before:", k.var(), k.var(), q.var(), wei.var())
wei *= head_size**-0.5 #scale by sqrt(d_k) as per paper, so that variance of the wei is 1. minute 1:17:00
print("after:", k.var(), k.var(), q.var(), wei.var())

tril = torch.tril(torch.ones(T,T))
wei = wei.masked_fill(tril==0, float('-inf')) #tokens only "talk" to previous tokens
wei = F.softmax(wei, dim=-1) #equivalent to the normalization above (-inf in upper diagonal will be 0)
v = value(x) #shape (B,T, head_size)
out = wei @ v
print(out.shape)
wei[0]




class Head(nn.Module):
  
  def __init__(self, head_size=16):
    super().__init__()
    self.key   = nn.Linear(n_embd, head_size, bias=False) 
    self.query = nn.Linear(n_embd, head_size, bias=False)
    self.value = nn.Linear(n_embd, head_size, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    self.dropout = nn.Dropout(dropout) #randomly prevents some tokens from communicating with each other
    
  def forward(self, x):
    B,T,C = x.shape
    k = self.key(x) #shape (B,T, head_size)
    q = self.query(x) #shape (B,T, head_size)
    v = self.value(x) #shape (B,T, head_size)
    
    #compute self-attention scores
    wei = q @ k.transpose(-2, -1) #shape (B,T, head_size) @ (B,head_size,T) --> (B,T,T)
    wei *= head_size**-0.5 #scale by sqrt(d_k) as per paper, so that variance of the wei is 1
    wei = wei.masked_fill(self.tril[:T,:T]==0, float('-inf')) # (B,T,T)
    wei = F.softmax(wei, dim=-1) # (B, T, T)
    wei = self.dropout(wei)
    
    #perform weighted aggregation of values
    v = self.value(x) #shape (B,T, head_size)
    out = wei @ v # (B, T, T) @ (B, T, head_size) --> (B, T, head_size)
    return out


class MultiHeadAttention(nn.Module):
  """ Multi-head attention as described in the paper. Simply a collection of heads with concatenated outputs."""
  def __init__(self, num_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    self.proj  = nn.Linear(n_embd, n_embd)
    self.dropout = nn.Dropout(dropout)
    
  def forward(self, x):
    out = torch.cat([head(x) for head in self.heads], dim=-1)
    out = self.proj(out)
    out = self.dropout(out)
    return out

class FeedForward(nn.Module):
  """ the feed forward network (FFN) in the paper"""
  def __init__(self, n_embd):
    super().__init__()
    # Note: in the paper (section 3.3) we have d_{model}=512 and d_{ff}=2048.
    # Therefore the inner layer is 4 times the size of the embedding layer
    self.net = nn.Sequential(
        nn.Linear(n_embd, n_embd*4),
        nn.ReLU(),
        nn.Linear(n_embd*4, n_embd),
        nn.Dropout(dropout)
      )
  def forward(self, x):
    return self.net(x)
  
class Block(nn.Module):
  """ Transformer block: comunication (attention) followed by computation (FFN) """
  def __init__(self, n_embd, n_head):
    # n_embd: embedding dimension
    # n_heads : the number of heads we'd like to use
    super().__init__()
    head_size = n_embd // n_head
    self.sa = MultiHeadAttention(n_head, head_size)
    self.ffwd = FeedForward(n_embd)
    self.ln1 = nn.LayerNorm(n_embd)
    self.ln2 = nn.LayerNorm(n_embd)
    
  def forward(self, x):
    x = x + self.sa(self.ln1(x))
    x = x + self.ffwd(self.ln2(x))
    return x
  
    
class MyModel(nn.Module):
  def __init__(self, num_head=4, n_layer=3):
    super().__init__()
    # vocabulary embedding and positional embedding
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    self.position_embedding_table = nn.Embedding(block_size, n_embd)
    
    #sequence of attention heads and feed forward layers
    self.blocks = nn.Sequential( *[Block(n_embd, n_head=4) for _ in range(n_layer)])

    #one layer normalization layer after transformer blocs and before linear layer that oututs the vocabulary
    self.ln = nn.LayerNorm(n_embd)
    self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)      

  def forward(self, idx, targets=None):
    """ call the model with idx and targets (training) or without targets (generation)"""
    #idx and targets are both of shape (B,T)
    B, T = idx.shape
    tok_emb = self.token_embedding_table(idx) #shape (B,T,C)
    pos_emb = self.position_embedding_table(torch.arange(T, device=device)) #shape (T,C)
    x = tok_emb + pos_emb #shape (B,T,C)
    x = self.blocks(x)
    x = self.ln(x)
    logits = self.lm_head(x) #shape (B,T,V)
        
    if targets is None: #when calling generate()
      loss = None
    else:
      B, T, C  = logits.shape
      logits = logits.view(B*T, C) #shape (B*T,C)
      targets = targets.view(-1) #shape (B*T)
      loss = F.cross_entropy(logits, targets)
    return logits, loss

  def generate(self, idx, max_new_tokens):
    """ given a context idx, generate max_new_tokens tokens and append them to idx """
    for _ in range(max_new_tokens):
      idx_cond = idx[:, -block_size:] #we can never have any idx longer than block_size
      logits, _ = self(idx_cond) #call fwd without targets
      logits = logits[:, -1, :] # shape (B, C)
      #convert logits to probabilities
      probs = F.softmax(logits, dim=-1) # shape (B, C)
      #randomly sample the next tokens, 1 for each of the previous probability distributions
      #(one could take instead the argmax, but that would be deterministic and boring)
      idx_next = torch.multinomial(probs, num_samples=1) # shape (B, 1)
      #append next token ix to the solution sequence so far
      idx = torch.cat([idx, idx_next], dim=-1) # shape (B, T+1)
    return idx  

m  = MyModel().to(device)

# train the model
optimizer = torch.optim.Adam(m.parameters(), lr=learning_rate)
for steps in range(max_iters):
  xb, yb = get_batch(train_data)   #get a batch of training data
  logits, loss = m(xb, yb)   #forward pass
  loss.backward()   #backward pass
  optimizer.step()   #update parameters
  optimizer.zero_grad(set_to_none=True)  #sets to None instead of 0, to save memory

  #print progress
  if steps % 100 == 0: print(f"step {steps}, loss {loss.item():.2f}")
    
  @torch.no_grad()
  # eval loop: np backprop on this data, to avoid storing all intermediatte variables
  def eval_loss():
    xb, yb = get_batch(valid_data)   #get a batch of validation data
    _, loss = m(xb, yb)
    print(f"step {steps}, eval loss {loss.item():.2f}")
    return loss
  
  if steps % eval_interval == 0: eval_loss().item()
      

#a 1x1 tensor with batch size 1 and sequence length 1 and starting value 0 (0 is the \n character)
idx = torch.zeros((1,1), dtype=torch.long, device=device)

# test the same generate() function, now with the trained model
print(decode(m.generate(idx, max_new_tokens=500).tolist()[0]))

