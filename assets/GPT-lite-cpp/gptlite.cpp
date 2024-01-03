#include <torch/torch.h>
#include "gptlite.h"

using namespace torch;
namespace nn = torch::nn;
namespace F = torch::nn::functional;
using Tensor = torch::Tensor;

const float inf = std::numeric_limits<float>::infinity();


Head::Head(int head_size): head_size(head_size){
	key   = nn::Linear( nn::LinearOptions(n_embd, head_size).bias(false) );
	query = nn::Linear( nn::LinearOptions(n_embd, head_size).bias(false) );
	value = nn::Linear( nn::LinearOptions(n_embd, head_size).bias(false) );
	tril = torch::tril(torch::ones( {block_size, block_size} ));
	this->dropout = nn::Dropout(dropout_p);

	register_module("key", key);
	register_module("query", query);
	register_module("value", value);
	register_buffer("tril", tril); //buffer is a tensor that is not a parameter, but is a state. Does not record gradients
	register_module("dropout", this->dropout);
}

Tensor Head::forward(Tensor x){
	int B=x.size(0), T=x.size(1), C=x.size(2);
	Tensor k = key(x);   //shape (B,T, head_size)
	Tensor q = query(x); //shape (B,T, head_size)
	Tensor v = value(x); //shape (B,T, head_size)

	// compute self-attention scores
	Tensor wei = torch::matmul(q, k.transpose(-2, -1)); //shape (B,T, head_size) @ (B,head_size,T) --> (B,T,T)
	wei = wei * std::pow(C,-0.5); //scale by sqrt(d_k) as per paper, so that variance of the wei is 1
	wei = wei.masked_fill(tril.slice(0, 0, T).slice(1, 0, T) == 0, -inf);
	wei = F::softmax(wei, -1); // (B, T, T)
	wei = this->dropout(wei);

	// perform weighted aggregation of values
	Tensor out = torch::matmul(wei, v); // (B, T, T) @ (B, T, head_size) --> (B, T, head_size)
	return out;
}



MultiHeadAttention::MultiHeadAttention(int num_heads, int head_size){

	heads = torch::nn::ModuleList();
	for (int i=0; i<num_heads; i++)
		heads->push_back( Head(head_size) );
    proj = nn::Linear(n_embd, n_embd);
    this->dropout = nn::Dropout(dropout_p);

	register_module("heads", heads);
	register_module("proj", proj);
	register_module("dropout", this->dropout);
}


Tensor MultiHeadAttention::forward(Tensor x){

	//Concatenate the outputs of the heads along the last dimension
	Tensor outputs[n_head];
	for (int i=0; i<n_head; i++){
		Head* head = heads[i]->as<Head>();
		outputs[i] = head->forward(x);
	}

    torch::Tensor out = torch::cat(outputs, -1);
    out = proj(out);
    out = this->dropout(out);
    return out;
}



FeedForward::FeedForward(int n_embd) {
    net = nn::Sequential(
        nn::Linear(n_embd, n_embd*4),
        nn::ReLU(),
        nn::Linear(n_embd*4, n_embd),
        nn::Dropout(dropout_p)
	);

	register_module("net", net);
}

Tensor FeedForward::forward(Tensor x){
    return net->forward(x);
}



Block::Block(int n_embd, int n_head){
    int head_size = (int) (n_embd / n_head);
    sa = std::shared_ptr<MultiHeadAttention>( new MultiHeadAttention(n_head, head_size) );
    ffwd = std::shared_ptr<FeedForward>( new FeedForward(n_embd) );
	ln1 = nn::LayerNorm(  std::vector<int64_t> {n_embd} );
	ln2 = nn::LayerNorm(  std::vector<int64_t> {n_embd} );

	register_module("sa", sa);
	register_module("ffwd", ffwd);
	register_module("ln1", ln1);
	register_module("ln2", ln2);
}

Tensor Block::forward(Tensor x){
    x = x + sa->forward(ln1(x));
    x = x + ffwd->forward(ln2(x));
    return x;
}



GPTlite::GPTlite(int vocab_size, torch::Device device){
	token_embedding_table = nn::Embedding(vocab_size, n_embd);
	position_embedding_table = nn::Embedding(block_size, n_embd);
	blocks = nn::Sequential();
	for (int i=0; i<n_layer; i++)
		blocks->push_back( Block(n_embd, n_head) );
		
	ln = nn::LayerNorm(  std::vector<int64_t> {n_embd} );
	lm_head = nn::Linear( nn::LinearOptions(n_embd, vocab_size).bias(false)  );

	//C++ has no reflection, so we need to register all modules so that c++
	//can iterate submodules for e.g. parameter count, moving submodules to GPU
    register_module("token_embedding_table", token_embedding_table);
	register_module("position_embedding_table", position_embedding_table);
	register_module("blocks", blocks);
	register_module("ln", ln);
	register_module("lm_head", lm_head);
	this->to(device);
}


Tensor GPTlite::forward(Tensor idx){
	int T = idx.size(1);
	Tensor tok_emb = token_embedding_table(idx); //shape (B,T,C)
	Tensor pos_emb = position_embedding_table(torch::arange(T).to( idx.device() )); //shape (T,C)
	Tensor x = tok_emb + pos_emb; //shape (B,T,C)
	x = blocks->forward(x);
	x = ln(x);
	Tensor logits = lm_head(x); //shape (B,T,C)
	//Note: python implementation uses .view(B*T,C) instead: 
	return logits.permute({0,2,1}); //shape (B,C,T)
}

