import math
import torch
import torch.nn.functional as F


class PagedKVCache:
    """ KV cache of one attention layer, split into fixed-size blocks that are allocated on demand from a
        shared pool. Each sequence has a block table: the list of the physical blocks holding its tokens """

    def __init__(self, n_blocks, block_size, n_heads, d_head, device='cpu', dtype=torch.float32):
        self.block_size = block_size
        self.k_pool = torch.zeros(n_blocks, n_heads, block_size, d_head, device=device, dtype=dtype)
        self.v_pool = torch.zeros_like(self.k_pool)
        self.free_blocks = list(range(n_blocks))
        self.block_tables, self.lengths = {}, {}

    def append(self, seq_id, k, v):
        """ Writes the key and value [n_heads, d_head] of a new token, allocating a new block when the last one is full """
        table = self.block_tables.setdefault(seq_id, [])
        pos = self.lengths.get(seq_id, 0)
        if pos % self.block_size == 0:
            if not self.free_blocks:
                raise RuntimeError("out of KV cache blocks: the scheduler must preempt a sequence")
            table.append(self.free_blocks.pop())
        block, offset = table[pos // self.block_size], pos % self.block_size
        self.k_pool[block, :, offset] = k
        self.v_pool[block, :, offset] = v
        self.lengths[seq_id] = pos + 1

    def free(self, seq_id):
        """ Returns the blocks of a finished sequence to the pool """
        self.free_blocks += self.block_tables.pop(seq_id)
        del self.lengths[seq_id]


def paged_decode_attention(q, cache, seq_ids):
    """ Attention of one new token per sequence (q: [B, n_heads, d_head]) over the cached tokens of its sequence.
        The blocks of each sequence are gathered into a padded tensor; a GPU kernel reads them in place instead """
    B, H, D = q.shape
    tables = [cache.block_tables[s] for s in seq_ids]
    n_blocks = max(len(t) for t in tables)
    table = torch.tensor([t + [0] * (n_blocks - len(t)) for t in tables], device=q.device)  # [B, n_blocks]
    k = cache.k_pool[table].permute(0, 2, 1, 3, 4).reshape(B, H, n_blocks * cache.block_size, D)  # [B, H, L, D]
    v = cache.v_pool[table].permute(0, 2, 1, 3, 4).reshape(B, H, n_blocks * cache.block_size, D)
    lengths = torch.tensor([cache.lengths[s] for s in seq_ids], device=q.device)
    mask = torch.arange(n_blocks * cache.block_size, device=q.device)[None, :] < lengths[:, None]  # [B, L]: real tokens
    return F.scaled_dot_product_attention(q[:, :, None], k, v, attn_mask=mask[:, None, None, :]).squeeze(2)


if __name__ == '__main__':
    torch.manual_seed(0)
    n_heads, d_head, block_size, max_seqlen = 4, 16, 16, 1024
    lengths = torch.randint(1, max_seqlen, (8,)).tolist()  # 8 sequences of different lengths
    cache = PagedKVCache(sum(math.ceil(n / block_size) for n in lengths), block_size, n_heads, d_head)

    # the sequences grow together, as in decode, so their blocks interleave in the pool
    keys, values = {s: [] for s in range(len(lengths))}, {s: [] for s in range(len(lengths))}
    for t in range(max(lengths)):
        for s, n in enumerate(lengths):
            if t < n:
                k, v = torch.randn(n_heads, d_head), torch.randn(n_heads, d_head)
                cache.append(s, k, v)
                keys[s].append(k)
                values[s].append(v)

    # check against standard attention over a contiguous copy of each sequence's keys and values
    q = torch.randn(len(lengths), n_heads, d_head)
    out = paged_decode_attention(q, cache, list(range(len(lengths))))
    ok = all(torch.allclose(out[s], F.scaled_dot_product_attention(
        q[s, :, None], torch.stack(keys[s], dim=1), torch.stack(values[s], dim=1)).squeeze(1), atol=1e-5)
        for s in range(len(lengths)))
    print(f"Same output as attention over contiguous caches: {ok}")
    print(f"Block table of sequence 0 (first 8 blocks): {cache.block_tables[0][:8]}")
    used, reserved = sum(lengths), len(lengths) * max_seqlen
    paged = sum(len(t) for t in cache.block_tables.values()) * block_size
    print(f"Tokens stored: {used}, contiguous cache reserving max_seqlen per sequence: {reserved} "
          f"({100 * used / reserved:.0f}% used), paged cache: {paged} ({100 * used / paged:.0f}% used)")
    cache.free(0)
    print(f"Free blocks after sequence 0 finished: {len(cache.free_blocks)}")

