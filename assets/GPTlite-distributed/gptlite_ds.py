import os
import sys
import argparse
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import deepspeed
from deepspeed.pipe import PipelineModule, LayerSpec

# use base GPTlite model from the GPT-lite post
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(current_dir, "..", "GPT-lite"))
from gptlite import n_layer, n_embd, n_head, block_size, Block  # noqa: E402


def _get_device_str() -> str:
    """Pick the correct CUDA device for this process (works for multi-node too)."""
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return f"cuda:{local_rank}"
    return "cpu"


class CrossEntropyLoss_FlatView(nn.Module):
    """Cross entropy for logits shaped (B, T, C) and labels shaped (B, T)."""

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        B, T, C = logits.shape
        return F.cross_entropy(logits.view(B * T, C), labels.view(-1))


################ BASE MODEL WITH ACTIVATION CHECKPOINTING ######################


class GPTlite(nn.Module):
    class EmbeddingsSum(nn.Module):
        """Converts tok_emb + pos_emb into an nn.Module (useful for pipelining/checkpointing)."""

        def __init__(self, vocab_size: int):
            super().__init__()
            self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
            self.position_embedding_table = nn.Embedding(block_size, n_embd)

        def forward(self, idx: torch.Tensor) -> torch.Tensor:
            _B, T = idx.shape
            tok_emb = self.token_embedding_table(idx)  # (B,T,C)
            pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T,C)
            return tok_emb + pos_emb  # (B,T,C)

    def __init__(self, vocab_size: int, activation_checkpoint_interval: int = 0):
        super().__init__()
        self.activation_checkpoint_interval = activation_checkpoint_interval

        self.emb_sum = GPTlite.EmbeddingsSum(vocab_size)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, targets=None) -> torch.Tensor:
        """Forward pass. idx is shape (B,T)."""
        if self.activation_checkpoint_interval and self.activation_checkpoint_interval > 0:
            x = idx
            for l, layer in enumerate(self.to_layers()):
                is_checkpoint = (l % self.activation_checkpoint_interval) == 0
                x = deepspeed.checkpointing.checkpoint(layer, x) if is_checkpoint else layer(x)
            return x

        x = self.emb_sum(idx)
        x = self.blocks(x)
        x = self.ln(x)
        return self.lm_head(x)  # (B,T,vocab)

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Given a context idx, generate max_new_tokens tokens and append them to idx."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits = self(idx_cond)  # (B,T,C)
            logits = logits[:, -1, :]  # (B,C)
            probs = F.softmax(logits, dim=-1)  # (B,C)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B,1)
            idx = torch.cat([idx, idx_next], dim=-1)
        return idx

    def to_layers(self) -> List[nn.Module]:
        # Returning modules (not lambdas) keeps PipelineModule and checkpointing happier.
        return [self.emb_sum, *self.blocks, self.ln, self.lm_head]


################ PIPELINE VERSION (LayerSpec) ######################


class GPTlitePipeSpec(PipelineModule):
    class EmbeddingsSum(nn.Module):
        """Converts tok_emb + pos_emb into an nn.Module. Required for LayerSpec."""

        def __init__(self, vocab_size: int):
            super().__init__()
            self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
            self.position_embedding_table = nn.Embedding(block_size, n_embd)

        def forward(self, idx: torch.Tensor) -> torch.Tensor:
            _B, T = idx.shape
            tok_emb = self.token_embedding_table(idx)
            pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
            return tok_emb + pos_emb

    def __init__(self, vocab_size: int, pipe_kwargs: dict):
        specs = (
            [LayerSpec(GPTlitePipeSpec.EmbeddingsSum, vocab_size)]
            + [LayerSpec(Block, n_embd, n_head) for _ in range(n_layer)]
            + [
                LayerSpec(nn.LayerNorm, n_embd),
                LayerSpec(nn.Linear, n_embd, vocab_size, bias=False),
            ]
        )
        super().__init__(layers=specs, **pipe_kwargs)


################ HELPERS ######################


def load_tiny_shakespeare_data():
    rank = dist.get_rank() if dist.is_initialized() else 0

    txt_path = os.path.join(current_dir, "..", "GPT-lite", "tinyshakespeare.txt")
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()

    if rank == 0:
        print("input data loaded. Length of text:", len(text))

    chars = sorted(list(set(text)))
    if rank == 0:
        print("unique chars:", "".join(chars))
        print("length of chars:", len(chars))

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda x: torch.tensor([stoi[ch] for ch in x], dtype=torch.long)
    decode = lambda x: "".join([itos[i] for i in x])
    vocab_size = len(stoi)

    if rank == 0:
        print("vocab size:", vocab_size)
        print(encode("Hello world"))
        print(decode(encode("Hello world").tolist()))
        print("character zero is:", decode([0]), "<end>")

    data = encode(text)
    n = int(0.9 * len(data))
    train_data, valid_data = data[:n], data[n:]
    if rank == 0:
        print("Train data encoded", data.shape, train_data.shape, valid_data.shape)
    return train_data, valid_data, vocab_size


def get_dataset():
    class GPTliteDataset(torch.utils.data.Dataset):
        def __init__(self, train_data, block_size):
            self.train_data = train_data
            self.block_size = block_size

        def __len__(self):
            # __getitem__ samples randomly anyway, but keep a reasonable bound
            return max(1, len(self.train_data) - self.block_size - 1)

        def __getitem__(self, idx):
            ix = torch.randint(len(self.train_data) - self.block_size - 1, size=())
            x = self.train_data[ix : ix + self.block_size]
            y = self.train_data[ix + 1 : ix + 1 + self.block_size]
            return x, y

    train_data, valid_data, vocab_size = load_tiny_shakespeare_data()
    train_dataset = GPTliteDataset(train_data, block_size)
    valid_dataset = GPTliteDataset(valid_data, block_size)
    return train_dataset, valid_dataset, vocab_size


def get_model(
    vocab_size: int,
    criterion=None,
    pipeline_num_stages: int = 0,
    pipeline_spec_layers: bool = False,
    activation_checkpoint_interval: int = 0,
):
    """Factory: returns either a regular model or a DeepSpeed PipelineModule."""
    if pipeline_num_stages:
        assert criterion is not None, "for pipeline runs, need to specify criterion"
        pipe_kwargs = {
            "num_stages": pipeline_num_stages,
            "activation_checkpoint_interval": activation_checkpoint_interval,
            "loss_fn": criterion,
        }

        if pipeline_spec_layers:
            return GPTlitePipeSpec(vocab_size, pipe_kwargs=pipe_kwargs)

        device_str = _get_device_str()
        base = GPTlite(vocab_size, activation_checkpoint_interval=0).to(device_str)
        return deepspeed.pipe.PipelineModule(layers=base.to_layers(), **pipe_kwargs)

    device_str = _get_device_str()
    return GPTlite(vocab_size, activation_checkpoint_interval=activation_checkpoint_interval).to(device_str)


################ PURE PYTORCH PIPELINE PARALLEL (NON-OVERLAPPED SCHEDULE) ######################


def split_layers_evenly(layers: List[nn.Module], num_stages: int) -> List[nn.Sequential]:
    """Naive equal split by number of layers (for demos)."""
    idx = torch.arange(len(layers))
    chunks = torch.chunk(idx, num_stages)
    return [nn.Sequential(*[layers[i] for i in c.tolist()]) for c in chunks]


def _pipeline_expected_shape(
    stage_id: int, num_stages: int, B: int, T: int, vocab_size: int
) -> Tuple[torch.Size, torch.dtype]:
    """Expected recv tensor shape/dtype for activations at a given stage (demo-specific)."""
    # stage 0 consumes token ids
    if stage_id == 0:
        return torch.Size([B, T]), torch.long
    # all intermediate stages (and the last stage input) consume hidden activations
    return torch.Size([B, T, n_embd]), torch.float32


def pytorch_pipeline_train_step(
    stage: nn.Module,
    stage_id: int,
    num_stages: int,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    idx: torch.Tensor,
    targets: torch.Tensor,
    micro_batches: int,
    vocab_size: int,
) -> Optional[torch.Tensor]:
    """One mini-batch step with a non-overlapped schedule (matches the 'staircase' diagram)."""

    device = next(stage.parameters()).device
    optimizer.zero_grad(set_to_none=True)

    idx_mbs = idx.chunk(micro_batches, dim=0)
    tgt_mbs = targets.chunk(micro_batches, dim=0)

    loss_out = None

    for idx_mb, tgt_mb in zip(idx_mbs, tgt_mbs):
        B_mb, T_mb = idx_mb.shape

        # ---- Forward ----
        if stage_id == 0:
            # stage 0 input is token ids (no grads needed for idx)
            act_out = stage(idx_mb.to(device))
            dist.send(act_out, dst=stage_id + 1)
            act_in = None  # no previous stage
        elif stage_id < num_stages - 1:
            # receive activation from previous stage
            recv_shape, recv_dtype = _pipeline_expected_shape(stage_id, num_stages, B_mb, T_mb, vocab_size)
            act_in = torch.empty(recv_shape, device=device, dtype=recv_dtype)
            dist.recv(act_in, src=stage_id - 1)
            act_in.requires_grad_(True)
            act_out = stage(act_in)
            dist.send(act_out, dst=stage_id + 1)
        else:
            # last stage: receive, compute loss on logits
            recv_shape, recv_dtype = _pipeline_expected_shape(stage_id, num_stages, B_mb, T_mb, vocab_size)
            act_in = torch.empty(recv_shape, device=device, dtype=recv_dtype)
            dist.recv(act_in, src=stage_id - 1)
            act_in.requires_grad_(True)
            logits = stage(act_in)  # (B,T,vocab)
            loss = criterion(logits, tgt_mb.to(device))
            loss.backward()
            dist.send(act_in.grad, dst=stage_id - 1)
            loss_out = loss.detach() if loss_out is None else (loss_out + loss.detach())
            continue

        # ---- Backward (non-last stages) ----
        # receive grad from next stage, backprop through local stage
        grad_out = torch.empty_like(act_out)
        dist.recv(grad_out, src=stage_id + 1)

        act_out.backward(grad_out)

        if stage_id > 0:
            # send grad for this stage's input activation to the previous stage
            dist.send(act_in.grad, dst=stage_id - 1)

    optimizer.step()
    return loss_out


def run_pytorch_pipeline(args):
    assert dist.is_initialized(), "Distributed must be initialized for pipeline parallelism."
    world = dist.get_world_size()
    rank = dist.get_rank()

    num_stages = args.num_stages
    if num_stages <= 0:
        raise ValueError("--num-stages must be > 0 for pytorch pipeline mode")
    if world < num_stages:
        raise ValueError(f"world_size ({world}) must be >= num_stages ({num_stages})")

    # For this demo we assume one pipeline group spanning ranks [0..num_stages-1]
    if rank >= num_stages:
        if rank == 0:
            print("Ranks >= num_stages are idle in this simple demo.")
        return

    train_dataset, _valid_dataset, vocab_size = get_dataset()

    device_str = _get_device_str()
    device = torch.device(device_str)

    torch.manual_seed(args.seed)

    # Build full model once, then split layers
    base = GPTlite(vocab_size, activation_checkpoint_interval=0).to(device)
    layers = base.to_layers()
    stages = split_layers_evenly(layers, num_stages)
    stage = stages[rank].to(device)

    optimizer = torch.optim.Adam(stage.parameters(), lr=args.lr)
    criterion = CrossEntropyLoss_FlatView()

    # Simple synthetic-ish batch: sample tokens from dataset (CPU), then move to device per stage
    for step in range(args.steps):
        # Create the SAME batch on all ranks deterministically (so last stage has matching targets)
        # This is a demo convenience.
        torch.manual_seed(args.seed + step)
        idx = torch.randint(low=0, high=vocab_size, size=(args.batch_size, args.seq_len), dtype=torch.long)
        targets = torch.randint(low=0, high=vocab_size, size=(args.batch_size, args.seq_len), dtype=torch.long)

        loss = pytorch_pipeline_train_step(
            stage=stage,
            stage_id=rank,
            num_stages=num_stages,
            optimizer=optimizer,
            criterion=criterion,
            idx=idx,
            targets=targets,
            micro_batches=args.micro_batches,
            vocab_size=vocab_size,
        )

        dist.barrier()
        if rank == num_stages - 1 and loss is not None:
            print(f"[PyTorch pipeline] step={step} loss={loss.item():.4f}")


################ DEEPSPEED RUNNER ######################


def run_deepspeed_pipeline(args):
    # DeepSpeed takes care of distributed init, but be defensive.
    if not dist.is_initialized():
        deepspeed.init_distributed()

    train_dataset, _valid_dataset, vocab_size = get_dataset()
    device_str = _get_device_str()

    criterion = CrossEntropyLoss_FlatView()

    model = get_model(
        vocab_size=vocab_size,
        criterion=criterion,
        pipeline_num_stages=args.num_stages,
        pipeline_spec_layers=args.pipeline_spec_layers,
        activation_checkpoint_interval=args.activation_checkpoint_interval,
    )

    # Minimal DS config; you can replace with ds_config.json.
    ds_config = {
        "train_batch_size": args.batch_size * args.micro_batches,
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.micro_batches,
        "optimizer": {"type": "Adam", "params": {"lr": args.lr}},
        "fp16": {"enabled": False},
        "bf16": {"enabled": False},
        "zero_optimization": {"stage": 1},
        "pipeline": {"seed_layers": True},
    }

    engine, optimizer, train_loader, _ = deepspeed.initialize(
        model=model,
        model_parameters=[p for p in model.parameters() if p.requires_grad],
        training_data=train_dataset,
        config_params=ds_config,
    )

    # For pipeline engine, train_batch() consumes gradient_accumulation_steps micro-batches.
    for step in range(args.steps):
        loss = engine.train_batch()
        if dist.get_rank() == 0:
            print(f"[DeepSpeed pipeline] step={step} loss={loss}")


def _parse_args():
    ap = argparse.ArgumentParser(description="GPTlite pipeline: PyTorch demo + DeepSpeed pipeline")
    ap.add_argument("--mode", choices=["pytorch", "deepspeed"], default="pytorch",
                    help="Which implementation to run.")
    ap.add_argument("--num-stages", type=int, default=2,
                    help="Number of pipeline stages (and pipeline ranks in the simple PyTorch demo).")
    ap.add_argument("--pipeline-spec-layers", action="store_true",
                    help="DeepSpeed only: use LayerSpec-based PipelineModule construction.")
    ap.add_argument("--activation-checkpoint-interval", type=int, default=0,
                    help="DeepSpeed pipeline / base model: activation checkpoint interval.")
    ap.add_argument("--batch-size", type=int, default=4, help="Micro-batch size per step.")
    ap.add_argument("--micro-batches", type=int, default=1, help="Gradient accumulation steps / number of micro-batches.")
    ap.add_argument("--seq-len", type=int, default=block_size, help="Sequence length T.")
    ap.add_argument("--steps", type=int, default=5, help="Number of training steps to run.")
    ap.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed.")
    return ap.parse_args()


def main():
    args = _parse_args()

    # Init distributed if needed (PyTorch demo needs it; DS can also use it).
    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)

    if args.mode == "pytorch":
        run_pytorch_pipeline(args)
    else:
        run_deepspeed_pipeline(args)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
