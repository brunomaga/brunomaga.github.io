import os
import torch
import torch.nn as nn
import torch.nn.functional as F

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


# load tiny shakespeare
current_dir = os.path.dirname(os.path.realpath(__file__))
txt_path = os.path.join(current_dir, 'tinyshakespeare.txt')
with open(txt_path) as f:
  text = f.read()

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
v = value(x) #shape (B,T, head_size)
wei = q @ k.transpose(-2, -1) #shape (B,T, head_size) @ (B,head_size,T) --> (B,T,T)
print("before:", k.var(), k.var(), q.var(), wei.var())
wei *= head_size**-0.5 #scale by sqrt(d_k) as per paper, so that variance of the wei is 1. minute 1:17:00
print("after:", k.var(), k.var(), q.var(), wei.var())

tril = torch.tril(torch.ones(T,T))
wei = wei.masked_fill(tril==0, float('-inf')) #tokens only "talk" to previous tokens
wei = F.softmax(wei, dim=-1) #equivalent to the normalization above (-inf in upper diagonal will be 0)
out = wei @ v
print(out.shape)
wei[0]


## end of tutorial. Load full model an trained it from scratch
import gptlite
gptlite.n_layer = n_layer
gptlite.n_embd = n_embd
gptlite.n_head = n_head
gptlite.block_size = block_size
m  = gptlite.GPTlite(vocab_size).to(device)

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

