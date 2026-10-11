import numpy as np
import torch
import faiss
from inference_utils import device, load_model, Timer  # also adds GPTlite to the path
from gptlite import GPTlite
from utils import get_tiny_shakespeare_data, get_gptlite_model_parameters
from main_kvcache import generate


class SemanticCache:
  """ Cache of (query, answer) pairs, looked up by the cosine similarity of the query embeddings """

  def __init__(self, embed_fn, threshold):
    self.embed_fn = embed_fn  # maps a list of strings to unit-norm float32 embeddings of shape [n, dim]
    dim = embed_fn(["probe"]).shape[1]
    self.index = faiss.IndexFlatIP(dim)  # exact search, inner product = cosine similarity of unit vectors
    self.answers = []
    self.threshold = threshold

  def __call__(self, query, generate_fn):
    """ Returns the answer to the query, the similarity to the closest cached query, and if it was a hit """
    embedding = self.embed_fn([query])
    similarity = 0.0
    if self.index.ntotal > 0:
      similarities, ids = self.index.search(embedding, 1)  # most similar cached query
      similarity = float(similarities[0, 0])
      if similarity >= self.threshold:
        return self.answers[ids[0, 0]], similarity, True
    answer = generate_fn(query)  # cache miss: call the model and store the new pair
    self.index.add(embedding)
    self.answers.append(answer)
    return answer, similarity, False


def sentence_transformer_embedder(model_name="all-MiniLM-L6-v2"):
  """ Embedding function of a pre-trained sentence encoder (downloaded on first use) """
  from sentence_transformers import SentenceTransformer
  encoder = SentenceTransformer(model_name)
  return lambda texts: encoder.encode(texts, normalize_embeddings=True).astype(np.float32)


if __name__=='__main__':
  torch.manual_seed(42) # random seed, for reproducibility
  vocab_size, _, _, encode_fn, decode_fn = get_tiny_shakespeare_data()
  seqlen = get_gptlite_model_parameters()[6]
  model = load_model(GPTlite, vocab_size)

  # GPTlite cannot answer questions: it is only a stand-in for the LLM behind the cache
  n_tokens = seqlen // 2
  def generate_fn(query):
    with torch.inference_mode():
      prompt = torch.tensor([encode_fn(query)], device=device)
      return decode_fn(generate(model, prompt, n_tokens, seqlen)[0, prompt.size(1):].tolist())

  cache = SemanticCache(sentence_transformer_embedder(), threshold=0.85)
  queries = [
    "What is the price of corn in Rome?",
    "How much does corn cost in Rome?",
    "Who is the king of Denmark?",
    "Who rules over Denmark?",
    "What is the price of corn in Rome?",
    "How do I delete a file?",
    "How do I delete a folder?",
  ]
  for query in queries:
    with Timer() as timer:
      answer, similarity, hit = cache(query, generate_fn)
    print(f"{'HIT ' if hit else 'MISS'} similarity={similarity:.2f} {timer.elapsed*1000:7.2f} ms  {query!r} -> {answer!r}")

