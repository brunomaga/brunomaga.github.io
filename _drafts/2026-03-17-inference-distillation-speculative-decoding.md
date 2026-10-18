---
layout: post
title:  "Knowledge distillation, speculative decoding and multi-token decoding"
redirect_from: /speculative-multitoken-decoding
categories: ["Efficient Inference"]
tags: ["LLM", "distillation", "speculative decoding", "multi-token prediction", "EAGLE", "Medusa"]
---

A large language model generates its output one token at a time, and every new token costs a full forward pass of the model. Two families of methods make this faster. The first one makes the model smaller: [knowledge distillation](https://en.wikipedia.org/wiki/Knowledge_distillation) trains a small model to behave like a large one. The second one keeps the large model, but lets it produce several tokens per forward pass. In speculative decoding, a cheap method guesses the next few tokens, and the large model checks all the guesses at once. These guesses, or drafts, can come from a small model (ideally a distilled one), from the context, from extra heads, from the features or the early layers of the model itself, or from a block diffusion model. Multi-token prediction goes one step further and builds the drafter into the model during pre-training.

Most methods come with a short PyTorch snippet, which assumes `import torch`, `import torch.nn as nn` and `import torch.nn.functional as F`. The models and the transformer blocks take an optional [KV cache]({{ site.baseurl }}{% post_url 2025-06-07-inference-prefill-decode-roofline-cache %}#kv-cache) and return it together with their output, with the new tokens appended.

## Knowledge distillation

Knowledge distillation (KD) trains a *student* model from a *teacher* model. The knowledge flows from a larger or already trained teacher to a smaller or untrained student, and the goal is a student that is smaller and faster than the teacher, and more accurate than it would be if it were trained alone. Why would a student learn more from a teacher than from the data? Because the *soft labels* of a trained network, i.e. its full output distribution, are a richer training signal than the *hard labels* of the dataset.

As an example, consider a classification task with two labels, dog and cat. An image of a cat that looks like a dog has the hard label `[0,1]`. A trained model, queried with the same image, outputs a soft label such as `[0.4, 0.6]`: the image is most likely a cat, but it could also be a dog. The soft label carries more information than the hard label, because it also tells how much the image resembles the other class. [Hinton et al.](https://arxiv.org/abs/1503.02531) point out that soft labels with a high entropy give much more information per training example than hard labels, and much less variance in the gradient between examples, so that the student can often be trained on less data and with a higher learning rate. In language modeling, the same holds at every position: the distribution of the teacher over the next token tells the student not only which token is correct, but also which other tokens are plausible.

KD methods differ in four choices. The loss can match the soft labels of the two models, as in the example above, or their intermediate representations, such as feature maps. The student can be a scaled-down version of the architecture of the teacher, or a different architecture. In **offline distillation**, the teacher is trained first and then frozen while the student learns from it. In **online distillation**, both models are trained at the same time, as the figure shows. Finally, the teacher can be a single model or an [ensemble](https://en.wikipedia.org/wiki/Ensemble_learning) of models.

{: .picture}
<img width="80.8%" height="80.8%" src="{{ site.assets }}/inference/imgs/model_distillation_offline_online.svg"/>

{: .caption}
Illustration of knowledge distillation (a) with a pre-trained teacher model (offline): step 1 trains the teacher, and step 2 distills its knowledge into the student; and (b) while training the teacher model simultaneously (online). Adapted from: [Knowledge distillation in deep learning and its applications](https://peerj.com/articles/cs-474/)

The methods are detailed in [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531), the paper of Hinton et al. at Google that named the method, and in two surveys, [Knowledge distillation in deep learning and its applications](https://peerj.com/articles/cs-474/) and [Knowledge Distillation: A Survey](https://arxiv.org/abs/2006.05525). The first survey sorts the methods as in the next figure: first by the signal that the student learns from, and then by the schedule of the training.

{: .picture}
<img width="99.6%" height="99.6%" src="{{ site.assets }}/inference/imgs/kd_methods.svg"/>

{: .caption}
The categories of knowledge distillation methods, and the branches within each category. This section implements offline distillation using soft labels, underlined in red in the picture. Adapted from: [Knowledge distillation in deep learning and its applications](https://peerj.com/articles/cs-474/).

### Implementing offline distillation with soft labels

In the simplest setting, the teacher is a pre-trained model, and the student is a smaller one, for example with fewer layers and a smaller embedding. The teacher is frozen: it runs in evaluation mode, so without dropout, and its forward pass runs inside `torch.no_grad()`, so that no computation graph and no gradients are created for it. At every training step, both models process the same batch, and the student learns to match the output distribution of the teacher at every position of the sequence. The soft labels of the teacher are computed on the fly instead of being stored on disk, because they are large: batch size × sequence length × vocabulary size values per batch, i.e. about 4.2 billion values, or 8.4 GB in 16-bit precision, for 8 sequences of 4,096 tokens and a vocabulary of 128,000 tokens. Storing them only pays off when the teacher is too large to run next to the student, and then usually only the top-k logits are stored, together with the inputs they belong to.

The loss measures how far the distribution of the student is from that of the teacher. The usual choice is the [Kullback-Leibler (KL) divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) between the distribution $$q$$ of the teacher and the distribution $$p$$ of the student:

$$
\begin{equation}
\begin{split}
D_{KL}(q \parallel p) & = H(q,p) - H(q) \\
 & = - \sum_i q_i \log (p_i) + \sum_i q_i \log (q_i) \\
 & = \sum_i q_i \log \frac{q_i}{p_i}
\end{split}
\end{equation}
$$

where $$H(q,p)$$ is the [cross entropy](https://en.wikipedia.org/wiki/Cross-entropy) and $$H(q)$$ the [entropy](https://en.wikipedia.org/wiki/Entropy_(information_theory)) of the distribution of the teacher. Since $$H(q)$$ does not depend on the student, minimizing the KL divergence is equivalent to minimizing the cross entropy: both losses have the same gradients with respect to the student. Their *values* differ, however. The KL divergence is zero when both distributions match, while the cross entropy then equals the entropy of the teacher. This is why the cross entropy is the usual loss for hard labels, whose entropy is zero, and the KL divergence the usual loss between two distributions.

In PyTorch, `F.kl_div` expects the log-probabilities of the student as its input. The distribution of the teacher is passed as log-probabilities too (`log_target=True`), because `log_softmax` keeps small probabilities accurate, while a softmax can round them to zero ([underflow](https://en.wikipedia.org/wiki/Arithmetic_underflow)). The logits are flattened to the shape (batch × sequence length, vocabulary size), and `reduction='batchmean'` divides the sum by the number of rows, which makes the loss the mean KL divergence per token.

How soft should the soft labels be? A trained model is often confident and puts almost all the probability on one token, so the probabilities of the other tokens, which carry the extra information, are tiny. The **temperature** $$T$$ controls the softness of the distributions. For logits $$z$$, the softened output is a [softmax](https://en.wikipedia.org/wiki/Softmax_function) with temperature:

$$
y_i (x \mid T) = \frac{ \exp\frac{z_i(x)}{T} }{ \sum_j \, \exp\frac{z_j(x)}{T} }
$$

With $$T = 1$$, this is the usual softmax. A temperature above 1 flattens the distribution and reveals the relative probabilities of the unlikely tokens. The gradients of the softened loss scale with $$1/T^2$$, so Hinton et al. multiply the loss by $$T^2$$, and the size of the gradients then does not depend on the temperature. A common variant adds the cross entropy with the hard labels, with a lower weight.

Other losses and other training sequences are possible too. [Kim et al.](https://arxiv.org/abs/2105.08919) report that the [mean squared error](https://en.wikipedia.org/wiki/Mean_squared_error) between the logits can work better than the KL divergence. For LLMs, the choice of the training sequences matters as well. A student trained on fixed sequences never sees its own mistakes, while at inference it generates its own sequences and must continue from them. On-policy distillation ([GKD](https://arxiv.org/abs/2306.13649)) removes this mismatch: it trains the student on sequences that the student generates itself, with the distributions of the teacher on these sequences as targets.

The following code implements the distillation loss and a training step of the student:

```python
def distillation_loss(logits_student, logits_teacher, temperature):
    """ KL divergence between the softened distributions of the teacher and the student, averaged per token, and
        scaled by temperature^2 so that the size of the gradients does not depend on the temperature.
        logits_student, logits_teacher: [B, S, V] """
    V = logits_student.size(-1)
    log_p_student = F.log_softmax(logits_student.reshape(-1, V) / temperature, dim=-1)  # [B, S, V] -> [B*S, V]
    log_p_teacher = F.log_softmax(logits_teacher.reshape(-1, V) / temperature, dim=-1)  # [B, S, V] -> [B*S, V]
    return F.kl_div(log_p_student, log_p_teacher, log_target=True, reduction='batchmean') * temperature ** 2


def distillation_step(student, teacher, tokens, optimizer, temperature=2.0):
    """ One training step of the student on a batch of tokens: [B, S]. The teacher is frozen, in evaluation mode, and
        runs without gradients. student(tokens) and teacher(tokens) return the logits: [B, S, V] and a KV cache """
    student.train()
    logits_student, _ = student(tokens)
    with torch.no_grad():
        logits_teacher, _ = teacher(tokens)
    loss = distillation_loss(logits_student, logits_teacher, temperature)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)  # against exploding gradients
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)  # sets the gradients to None instead of 0, to save memory
    return loss.item()


# usage, with a pre-trained teacher and a smaller student:
# teacher.eval()
# optimizer = torch.optim.Adam(student.parameters(), lr=lr)
# for tokens in batches:
#     loss = distillation_step(student, teacher, tokens, optimizer)
```

## Speculative decoding: more than one token per forward pass

### The idea

Why could a large model check several tokens in the time it takes to generate one? Because decode is [memory-bound]({{ site.baseurl }}{% post_url 2025-06-07-inference-prefill-decode-roofline-cache %}#prefill-and-decode-phases): at small batch sizes, most of the time of a forward pass goes to reading the weights from memory, not to the arithmetic, so a forward pass over a few tokens takes about as long as a forward pass over one. Speculative decoding exploits this property. A cheap method *drafts* $$K$$ tokens, and the large *target* model checks all of them in a single forward pass. Every draft token that the target accepts is then generated almost for free. At the first disagreement, the tokens before it are kept, the token of the target replaces the draft, and drafting starts again from there. With the right acceptance rule, the output has the same distribution as with the target model alone, which is why speculative decoding is called lossless.

How many tokens does one forward pass of the target produce? If each draft token is accepted independently with probability $$\alpha$$, the first $$i$$ drafts are all accepted with probability $$\alpha^i$$, and the target always adds one token of its own, after the last accepted draft. The expected number of tokens per forward pass of the target is therefore:

$$
1 + \alpha + \alpha^2 + \dots + \alpha^K = \frac{1 - \alpha^{K+1}}{1 - \alpha}
$$

Drafting has a cost, however. If drafting one token costs a fraction $$c$$ of a forward pass of the target, and a verification costs about as much as a regular decoding step, one iteration takes the time of $$cK + 1$$ forward passes of the target, and the expected speedup over decoding with the target alone is ([Leviathan et al.](https://arxiv.org/abs/2211.17192)):

$$
\frac{1 - \alpha^{K+1}}{(1 - \alpha)(cK + 1)}
$$

The speedup therefore depends on the accuracy of the drafts and on their cost. For example, with $$\alpha = 0.8$$, $$K = 4$$ and $$c = 0.05$$, the target produces 3.36 tokens per forward pass, and the speedup is 2.8. Longer drafts help only up to a point. The expected number of tokens can never exceed $$1 / (1 - \alpha)$$, which is 5 tokens for $$\alpha = 0.8$$, while the drafting cost keeps growing with $$K$$: with the same $$\alpha$$ and $$c$$, the speedup peaks at about 3.1 for $$K = 8$$, and then falls. Speculative decoding also helps most at small batch sizes, where decode is most memory-bound. At large batch sizes, the GPU is already busy, and the verification of rejected drafts takes compute away from useful work. The methods below differ in the source of the drafts.

### Draft-model speculative decoding

The original method ([Leviathan et al.](https://arxiv.org/abs/2211.17192), [Chen et al.](https://arxiv.org/abs/2302.01318)) drafts with a small language model that uses the same tokenizer as the target, so that the target can check the drafts token by token. Ideally, the draft model is a distilled version of the target, such as the student of the [distillation section](#knowledge-distillation) above, because it then picks the same tokens as the target more often. Each iteration has three steps:

1. the draft model generates $$K$$ tokens autoregressively;
2. the target model runs one forward pass over the context and the $$K$$ draft tokens, which gives its distribution at each of the $$K$$ draft positions, plus one after the last draft;
3. the draft tokens are checked from left to right.

The acceptance rule depends on the decoding strategy.

With **greedy decoding**, the draft tokens are accepted as long as they match the most likely token of the target. At the first mismatch, the most likely token of the target replaces the draft, and the remaining drafts are discarded. If all $$K$$ drafts are accepted, the prediction of the target after the last draft is appended as a *bonus* token. The output is identical to greedy decoding with the target alone. The figure shows an example. The context ends with "The cat sat on the", and the draft model proposes "mat and the dog", with one forward pass per draft. The target then runs a single forward pass over the last token of the context and the four drafts, and predicts the next token at each of these five positions. Its predictions agree with the first two drafts, "mat" and "and", but after "and" the target predicts "purred" instead of "the". The new sequence therefore ends with "mat and purred": two accepted drafts and one token of the target, three tokens for one forward pass of the target. The rejected drafts "the" and "dog" are discarded together with their entries in the KV cache, and the predictions that follow them are not used.

{: .picture}
<img width="94.6%" height="94.6%" src="{{ site.assets }}/inference/imgs/spec_decoding.svg"/>

With **sampling** (*speculative sampling*), the draft model samples each token $$x$$ from its distribution $$p$$, and the target accepts it with probability $$\min\left(1, q(x) / p(x)\right)$$, where $$q$$ is the distribution of the target. On rejection, a replacement token is sampled from the normalized residual distribution $$\max(0, q - p)$$, and the remaining drafts are discarded. If all $$K$$ drafts are accepted, a bonus token is sampled from the distribution of the target after the last draft. This modified [rejection sampling](https://en.wikipedia.org/wiki/Rejection_sampling) produces tokens distributed exactly as $$q$$.

Why exactly as $$q$$? The figure below shows an example with a vocabulary of five tokens. The draft model proposes b with probability $$p(b) = 0.5$$, more than the probability $$q(b) = 0.3$$ of the target, so the target accepts b with probability $$0.3 / 0.5 = 0.6$$, and b comes out with probability $$0.5 \times 0.6 = 0.3 = q(b)$$. The tokens a and c, on the contrary, are drafted less often than the target would choose them, so they are always accepted when drafted. A draft is rejected with probability $$0.5 - 0.3 = 0.2$$, which is exactly the probability that a and c miss together, and the residual hands it to them in equal parts: a comes out with probability $$0.1 + 0.2 \times 0.5 = 0.2 = q(a)$$, and c with $$0.2 + 0.2 \times 0.5 = 0.3 = q(c)$$. The same argument holds for any $$p$$, as long as the drafts were sampled from it. Greedy drafts are the case of a one-hot $$p$$: the target then accepts the draft $$x$$ with probability $$q(x)$$, and on rejection samples from $$q$$ without $$x$$, so the tokens are still distributed as $$q$$. The guarantee breaks, however, when the drafts are greedy but the softmax probabilities of the draft model are used as $$p$$.

{: .picture}
<img width="99.8%" height="99.8%" src="{{ site.assets }}/inference/imgs/spec_sampling.svg"/>

{: .caption}
The draft b is accepted with probability $$\min(1, q(b) / p(b)) = 0.3 / 0.5 = 0.6$$; otherwise, it is replaced by a or c, sampled from the residual.

In a batch, each sequence accepts a different number of tokens, so the sequences advance at different speeds, and inference engines track the length of each sequence separately, as in [continuous batching]({{ site.baseurl }}{% post_url 2025-06-07-inference-prefill-decode-roofline-cache %}#static-and-continuous-batching).

The following code implements greedy speculative decoding with a draft model. For simplicity, it generates one sequence at a time, and the whole sequence must fit in the context window. Both models use a KV cache. After each verification, the cache of the target is truncated to the accepted tokens, and the draft model reuses the part of its own cache that is still valid. The output is the same as that of greedy decoding with the target alone:

```python
def truncate_cache(kv_cache, n):
    """ Keeps the first n tokens of a KV cache: a list with a tuple of tensors per layer, such as the keys and the
        values: [B, H, P, h], with the tokens in dimension 2 """
    return [tuple(t[:, :, :n] for t in layer_cache) for layer_cache in kv_cache]


def generate_speculative(model, prompt, n_tokens, propose, n_drafts):
    """ Greedy speculative decoding of a single sequence, prompt: [1, S]. model(tokens, kv_cache) returns the logits of
        the new tokens: [1, S, V] and the KV cache with them appended. At every iteration, propose(tokens, k) returns
        up to k draft tokens: [1, k], and the target checks all of them in a single forward pass. The cache of the
        target holds all the accepted tokens except the last one, which is processed together with the drafts. The
        whole sequence must fit in the context window """
    n_total = prompt.size(1) + n_tokens
    tokens, kv_cache = prompt, None
    if prompt.size(1) > 1:
        _, kv_cache = model(prompt[:, :-1])
    while tokens.size(1) < n_total:
        drafts = propose(tokens, min(n_drafts, n_total - tokens.size(1) - 1))  # [1, k]

        # predictions of the target after the last accepted token and after each draft: [1, k+1, V] -> [1, k+1]
        logits, kv_cache = model(torch.cat([tokens[:, -1:], drafts], dim=1), kv_cache)
        predictions = logits.argmax(dim=-1)

        # accept the drafts while they match the predictions of the target
        n_accepted = 0
        while n_accepted < drafts.size(1) and drafts[0, n_accepted] == predictions[0, n_accepted]:
            n_accepted += 1

        # keep the accepted drafts, plus the target's own token at the first mismatch (or the bonus token, after the
        # last draft if all were accepted), and drop the cache entries of the rejected drafts
        tokens = torch.cat([tokens, drafts[:, :n_accepted], predictions[:, n_accepted:n_accepted + 1]], dim=1)
        kv_cache = truncate_cache(kv_cache, tokens.size(1) - 1)
    return tokens  # [1, S + n_tokens]


class DraftModel:
    """ Proposes greedy drafts with a small model, model(tokens, kv_cache) -> (logits, kv_cache), and its own KV cache.
        On every call, it reuses the cache of the longest prefix of the tokens it has already processed """

    def __init__(self, model):
        self.model = model
        self.kv_cache, self.processed = None, None  # the cache, and the tokens it holds: [1, P]

    def __call__(self, tokens, n_drafts):
        if n_drafts == 0:
            return tokens[:, :0]

        # number of leading tokens processed before and still valid; the last token is always processed, to predict
        # the first draft
        n_cached = 0
        if self.processed is not None:
            n = min(self.processed.size(1), tokens.size(1) - 1)
            mismatches = (self.processed[0, :n] != tokens[0, :n]).nonzero()
            n_cached = mismatches[0].item() if len(mismatches) > 0 else n
        kv_cache = truncate_cache(self.kv_cache, n_cached) if n_cached > 0 else None

        new_tokens, drafts = tokens[:, n_cached:], []
        for _ in range(n_drafts):  # greedy generation with the draft model
            logits, kv_cache = self.model(new_tokens, kv_cache)  # [1, S, V]
            new_tokens = logits[:, -1:].argmax(dim=-1)            # [1, 1]
            drafts.append(new_tokens)
        self.kv_cache = kv_cache
        self.processed = torch.cat([tokens] + drafts[:-1], dim=1)  # the last draft was not processed
        return torch.cat(drafts, dim=1)                             # [1, n_drafts]


# usage, with a small model that uses the same tokenizer as the target:
# tokens = generate_speculative(target, prompt, n_tokens, DraftModel(draft), n_drafts=4)
```

The following code implements the acceptance rule of speculative sampling, which replaces the greedy acceptance in the loop above:

```python
def accept_sampling(drafts, p, q):
    """ Speculative sampling: verifies the drafts: [k], sampled from the distributions p: [k, V] of the draft model,
        with the distributions q: [k+1, V] of the target after the last accepted token and after each draft. A draft
        x is accepted with probability min(1, q(x) / p(x)). Returns the accepted drafts, followed by a token sampled
        from the residual max(0, q - p) at the first rejection, or by a bonus token sampled from the last q if all the
        drafts are accepted """
    for i, x in enumerate(drafts.tolist()):
        if torch.rand(()) >= q[i, x] / p[i, x]:  # rejected
            residual = (q[i] - p[i]).clamp(min=0)  # [V]
            return torch.cat([drafts[:i], torch.multinomial(residual / residual.sum(), 1)])
    return torch.cat([drafts, torch.multinomial(q[-1], 1)])


# in generate_speculative, the drafter samples the drafts and returns their distributions p, the distributions of
# the target are q = softmax(logits / temperature), and accept_sampling replaces the greedy acceptance
```

### Prompt lookup decoding

The drafts do not have to come from a model. Many outputs repeat parts of their input: a summary quotes its document, an edited program keeps most of the original code, and an answer about a document copies names and phrases from it. [Prompt lookup decoding](https://github.com/apoorvumang/prompt-lookup-decoding) exploits this, and needs no draft model at all. It takes the last few generated tokens (an [n-gram](https://en.wikipedia.org/wiki/N-gram)), searches for the same n-gram earlier in the prompt or in the generated text, and proposes the tokens that followed it as drafts. In the figure, the last two tokens, "the cat", also appear at the start of the text, where they are followed by "sat on the mat", and these four tokens become the drafts.

The verification is the same as above. Drafting costs almost nothing, and the speedup is large when the output copies from the input, as in summarization, code editing, question answering over documents, or rewriting a previous answer. When the output does not repeat the context, no drafts are found, and decoding runs at about the speed of the target alone. Related methods draft from other sources: [lookahead decoding](https://arxiv.org/abs/2402.02057) collects n-grams from [Jacobi iterations](https://en.wikipedia.org/wiki/Jacobi_method) of the model itself, and [REST](https://arxiv.org/abs/2311.08252) retrieves continuations from a datastore of text.

{: .picture}
<img width="99.9%" height="99.9%" src="{{ site.assets }}/inference/imgs/prompt_lookup.svg"/>

The following code implements prompt lookup decoding. It reuses the verification loop of speculative decoding, with a different source of drafts:

```python
def prompt_lookup(tokens, n_drafts, max_ngram=3):
    """ Drafts the tokens that followed the most recent earlier occurrence of the last n-gram of the sequence, trying
        the longest n-grams first. tokens: [1, S] -> drafts: [1, k], with k <= n_drafts, and k = 0 if the n-gram never
        occurred before """
    sequence = tokens[0]  # [S]
    for n in range(max_ngram, 0, -1):
        if sequence.size(0) <= n:
            continue
        ngram = sequence[-n:]                     # [n]
        windows = sequence[:-1].unfold(0, n, 1)  # the earlier n-grams: [S-n, n]
        matches = (windows == ngram).all(dim=1).nonzero()
        if len(matches) > 0:
            start = matches[-1].item() + n  # first token after the most recent match
            return sequence[start:start + n_drafts].unsqueeze(0)
    return tokens[:, :0]


# usage: the verification loop of speculative decoding, with a different source of drafts
# tokens = generate_speculative(model, prompt, n_tokens, prompt_lookup, n_drafts=4)
```

### Medusa

A separate draft model has its costs: it must be trained, stored and run next to the target, and it must use the same tokenizer. [Medusa](https://arxiv.org/abs/2401.10774) avoids them by adding a few extra *decoding heads* on top of the last hidden state of the target model itself. The original output head predicts the next token, and head $$i$$ predicts the token $$i+1$$ positions ahead. The heads read the hidden state that the forward pass of the target computes anyway, so drafting costs little more than the heads themselves. The top candidates of each head are combined into a tree of possible continuations. In the figure, the output head predicts "the", Medusa head 1 proposes "cat" and "dog", and Medusa head 2 proposes "sat" and "ran", which gives a tree with four paths. The target verifies all the candidates in a single forward pass with *tree attention*, a mask that lets each candidate attend only to its own ancestors in the tree, so that each path is processed as if it were a separate sequence. The longest accepted path is kept, here "the cat sat".

Training only the heads, with the model frozen (Medusa-1), gives speedups above 2.2 times, and training the heads together with the model (Medusa-2) reaches 2.3 to 3.6 times. However, each head guesses its token independently, without seeing the tokens guessed by the other heads: in the figure, head 2 proposes "sat" and "ran" without knowing whether head 1 chose "cat" or "dog". The accuracy of the heads therefore drops quickly for later positions. For sampling, Medusa also proposes a faster *typical acceptance* rule, which is not exactly lossless.

{: .picture}
<img width="94.6%" height="94.6%" src="{{ site.assets }}/inference/imgs/medusa.svg"/>

{: .caption}
The output head and the Medusa heads read the last hidden state $$h$$. Their top candidates combine into a tree, which the target verifies in one forward pass with a tree attention mask, in which each node sees its ancestors. Inspired by: [Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774)

The following code implements the Medusa heads, the tree of candidates and its attention mask, and one step of greedy decoding that keeps only the KV cache of the accepted path:

```python
class MedusaHeads(nn.Module):
    """ Medusa decoding heads, on the last hidden states of the target: head k predicts the token k positions after the
        next one, with a residual block that starts as the identity (zero weights) and a copy of the output layer of
        the target: logits_k = W_k (SiLU(W1_k h) + h) """

    def __init__(self, lm_head, num_heads=4):
        super().__init__()
        E, V = lm_head.in_features, lm_head.out_features
        self.blocks = nn.ModuleList(nn.Linear(E, E) for _ in range(num_heads))
        self.heads = nn.ModuleList(nn.Linear(E, V, bias=lm_head.bias is not None) for _ in range(num_heads))
        for block, head in zip(self.blocks, self.heads):
            nn.init.zeros_(block.weight)
            nn.init.zeros_(block.bias)
            head.load_state_dict(lm_head.state_dict())

    def forward(self, h):
        """ h: [B, S, E] -> logits of the num_heads tokens after the next one: [B, S, num_heads, V] """
        return torch.stack([head(F.silu(block(h)) + h) for block, head in zip(self.blocks, self.heads)], dim=2)


def medusa_tree(next_token, head_logits, top_k=2):
    """ Tree of candidate continuations: the target's next token at the root, the top_k tokens of head 1 as its
        children, the top_k tokens of head 2 as the children of each of them, and so on. head_logits: [num_heads, V],
        at the last accepted token. Returns the tokens of the N nodes: [N], the parent of each node (-1 for the root,
        parents before children) and the depth of each node: [N] """
    tokens, parents, depths, level = [next_token], [-1], [0], [0]  # level: the nodes of the deepest level so far
    for depth, logits in enumerate(head_logits, start=1):
        top = logits.topk(top_k).indices.tolist()
        new_level = []
        for parent in level:
            for token in top:
                new_level.append(len(tokens))
                tokens.append(token)
                parents.append(parent)
                depths.append(depth)
        level = new_level
    return torch.tensor(tokens), parents, torch.tensor(depths)


def tree_attention_mask(parents):
    """ Tree attention: [N, N], True where node i can attend to node j, i.e. j is i or one of its ancestors """
    mask = torch.eye(len(parents), dtype=torch.bool)
    for i, parent in enumerate(parents):
        if parent >= 0:
            mask[i] |= mask[parent]  # the row of the parent already holds its ancestors
    return mask


def longest_accepted_path(tokens, parents, predictions):
    """ Greedy verification of the tree: the root is the target's own token, and a node is accepted if its parent is
        accepted and its token is the target's prediction after its parent. tokens, predictions: lists of N tokens.
        Returns the nodes of the longest accepted path, from the root """
    paths = {0: [0]}  # path from the root to every accepted node
    for i in range(1, len(parents)):
        if parents[i] in paths and tokens[i] == predictions[parents[i]]:
            paths[i] = paths[parents[i]] + [i]
    return max(paths.values(), key=len)


def medusa_step(model, heads, tokens, kv_cache, next_token, head_logits, top_k=2):
    """ One step of greedy decoding with Medusa. tokens: [1, S], the accepted tokens, all in the KV cache; next_token:
        the target's next token; head_logits: [num_heads, V], of the Medusa heads at the last accepted token.
        model(tokens, kv_cache, mask, positions) returns the logits [1, N, V] and the last hidden states [1, N, E] of
        the new tokens, and the cache with them appended """
    tree, parents, depths = medusa_tree(next_token, head_logits, top_k)  # [N]
    S, device = tokens.size(1), tokens.device

    # one forward pass over the tree: every node sees the cached tokens and its ancestors, at the position after its
    # parent: mask [N, S+N]
    mask = torch.cat([torch.ones(len(tree), S, dtype=torch.bool), tree_attention_mask(parents)], dim=1)
    logits, hidden, kv_cache = model(tree[None].to(device), kv_cache, mask.to(device), (S + depths).to(device))
    path = longest_accepted_path(tree.tolist(), parents, logits[0].argmax(dim=-1).tolist())

    # keep the accepted path, and its entries in the cache
    keep = torch.tensor(list(range(S)) + [S + i for i in path], device=device)
    kv_cache = [tuple(t[:, :, keep] for t in layer_cache) for layer_cache in kv_cache]
    tokens = torch.cat([tokens, tree[path][None].to(device)], dim=1)

    # the target's token after the path, and the Medusa heads at the last node of the path, for the next step
    next_token = logits[0, path[-1]].argmax().item()
    head_logits = heads(hidden[:, path[-1]:path[-1] + 1])[0, 0]  # [num_heads, V]
    return tokens, kv_cache, next_token, head_logits


# usage: prefill the prompt: [1, S], with the causal_mask of the post on attention, then run medusa_step:
# logits, hidden, kv_cache = model(prompt, None, causal_mask(S, S), torch.arange(S))
# tokens, next_token, head_logits = prompt, logits[0, -1].argmax().item(), heads(hidden[:, -1:])[0, 0]
# while tokens.size(1) < S + n_tokens:
#     tokens, kv_cache, next_token, head_logits = medusa_step(model, heads, tokens, kv_cache, next_token, head_logits)
```

### EAGLE

[EAGLE](https://arxiv.org/abs/2401.15077) removes this weakness of Medusa: its drafts depend on each other, and it drafts at the level of *features* instead of tokens. Its drafter is a single lightweight transformer layer, which takes the top-layer hidden states (features) of the target, together with the embeddings of the tokens one position ahead, and predicts the next feature. The output layer of the target then converts this feature into a token. Why the tokens one position ahead? With sampling, the same feature can be followed by different tokens, and the feature alone does not tell which token was sampled, so the shifted tokens give the drafter this missing information. Every new draft appends its predicted feature and its token to the inputs of the drafter, so each draft depends on the previous ones, unlike the heads of Medusa. According to the authors, feature sequences are also easier to predict than token sequences, and the drafts are therefore much more accurate than those of Medusa.

In the figure, the target has processed the tokens $$t_1$$ to $$t_4$$ and predicted the next token $$t_5$$. The drafter combines the features of $$t_1$$ to $$t_4$$ with the embeddings of $$t_2$$ to $$t_5$$, predicts the features of $$t_2$$ to $$t_5$$, and the shared output layer turns the last of them into the draft $$t_6$$.

[EAGLE-2](https://arxiv.org/abs/2406.16858) shapes the draft tree dynamically, and expands the branches where the drafter is confident. The authors of [EAGLE-3](https://arxiv.org/abs/2503.01840) (NeurIPS 2025) then observed that EAGLE barely improves with more training data, and traced this to its feature prediction loss. EAGLE-3 therefore drops feature prediction and predicts tokens directly. It fuses the low-, mid- and high-level features of the target instead of the top layer only, and it uses *training-time test*: in training, the drafter is fed its own predictions, as it is when it drafts several tokens at inference. EAGLE-3 reaches speedups of up to 6.5 times, about 1.4 times those of EAGLE-2, and 1.38 times more throughput at a batch size of 64 in SGLang. vLLM, SGLang and TensorRT-LLM support EAGLE-style drafting.

{: .picture}
<img width="99%" height="99%" src="{{ site.assets }}/inference/imgs/eagle.svg"/>

{: .caption}
For the next draft, the predicted feature of $$t_5$$ and the draft $$t_6$$ are appended to the inputs. EAGLE-3 predicts tokens directly, from the fused low-, mid- and high-level features of the target. Inspired by: [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077)

The following code implements an EAGLE drafter, and a drafting loop that the MTP modules below also use:

```python
class EagleDrafter(nn.Module):
    """ EAGLE drafter: a transformer block predicts the next feature (last hidden state) of the target from its
        features and the embeddings of the tokens one position ahead, and the output layer of the target turns the
        predicted features into tokens. embedding and lm_head are those of the target, frozen; block: a transformer
        block with causal attention, block(x, kv_cache=None) -> (x, kv_cache). EAGLE-3 predicts tokens directly, from
        the concatenated features of a low, a middle and a high layer of the target """

    def __init__(self, embedding, lm_head, block):
        super().__init__()
        E = embedding.embedding_dim
        self.embedding, self.lm_head, self.block = embedding, lm_head, block
        self.fc = nn.Linear(2 * E, E)

    def forward(self, features, tokens):
        """ features: [B, S, E] at positions 1..S, tokens: [B, S] at positions 2..S+1 -> predicted features at positions
            2..S+1: [B, S, E], and logits of the tokens at positions 3..S+2: [B, S, V] """
        x = self.fc(torch.cat([features, self.embedding(tokens)], dim=-1))  # [B, S, 2E] -> [B, S, E]
        features, _ = self.block(x)                                         # [B, S, E]
        return features, self.lm_head(features)


@torch.no_grad()
def draft_with_features(drafter, features, tokens, n_drafts):
    """ Drafts n_drafts tokens with a drafter that predicts features: an EAGLE drafter, or an MTP module (below).
        features: [1, S, E], the target's features of the tokens 1..S; tokens: [1, S], the tokens 2..S+1, the last one
        being the last accepted token. Every draft appends its predicted feature and its token. For simplicity, the
        drafter recomputes the whole sequence at every draft, instead of using a KV cache """
    drafts = tokens[:, :0]
    for _ in range(n_drafts):
        predicted, logits = drafter(features, tokens)  # [1, S, E], [1, S, V]
        draft = logits[:, -1:].argmax(dim=-1)           # [1, 1]
        features = torch.cat([features, predicted[:, -1:]], dim=1)
        tokens = torch.cat([tokens, draft], dim=1)
        drafts = torch.cat([drafts, draft], dim=1)
    return drafts                                       # [1, n_drafts]


# usage, with features(tokens) the target's features of the tokens, kept from its forward passes:
# propose = lambda tokens, k: draft_with_features(drafter, features(tokens[:, :-1]), tokens[:, 1:], k)
# tokens = generate_speculative(target, prompt, n_tokens, propose, n_drafts=4)
```

### DFlash

Autoregressive drafters such as EAGLE-3 need one forward pass per draft token, so they are kept shallow: EAGLE-3 uses a single transformer layer. According to the authors of [DFlash](https://arxiv.org/abs/2602.06036) (Chen, Liang and Liu, UC San Diego, ICML 2026), this shallow depth limits the acceptance length, and keeps practical speedups around 2 to 3 times. DFlash replaces the autoregressive drafter with a small *block [diffusion](https://en.wikipedia.org/wiki/Diffusion_model)* model, which drafts all the tokens of a block at once. It takes the last verified token, appends 15 mask tokens, and fills the whole block in a single forward pass. Its drafting cost therefore barely grows with the block size (in the speedup formula above, $$cK$$ becomes about $$c$$), so it can afford a deeper drafter: a 5-layer DFlash drafter fills a 16-token block faster than a 1-layer EAGLE-3 drafter drafts 8 tokens, and more of its tokens are accepted.

A drafter this small predicts poorly on its own: without the features of the target, a 5-layer diffusion drafter reaches a speedup of only about 2 to 3 times. DFlash therefore conditions the drafter on the target, as the figure shows. The hidden states of 5 layers of the target, spread from shallow to deep, are concatenated and projected, and the result is injected into the keys and values of every draft layer, not only into the input of the drafter as in EAGLE-3. Every draft layer then attends to the representation of the context computed by the target, and the acceptance length keeps improving as draft layers are added. In the figure, the block holds 8 tokens instead of 16: the last verified token, "the", and 7 mask tokens, which the 5 draft layers fill with "cat sat on the mat . It" in one forward pass.

The drafter reuses the frozen embedding and output head of the target, and only its transformer layers are trained, on responses generated by the target. Each training block starts at a randomly sampled anchor token, as at inference, and all the blocks of a sequence are trained in one pass with a block-sparse attention mask. The loss weights the early positions of a block more, because one early mistake invalidates the rest of the block. The verification is unchanged, so the output stays lossless. Each round verifies one drafted block, not a tree; [DDTree](https://arxiv.org/abs/2604.12989) builds a draft tree from the per-position predictions of a block diffusion drafter such as DFlash, and verifies it with tree attention.

{: .picture}
<img width="95.1%" height="95.1%" src="{{ site.assets }}/inference/imgs/dflash.svg"/>

{: .caption}
The hidden states of 5 layers of the frozen target, spread from shallow to deep, are concatenated, projected, and injected into the keys and values of every draft layer. The drafter fills all the mask positions of the block in one forward pass. Inspired by: [DFlash: Block Diffusion for Flash Speculative Decoding](https://arxiv.org/abs/2602.06036)

How large are the gains? On Qwen3-8B with greedy decoding, the paper reports about 6.5 tokens per verification step on average, and a speedup over decoding with the target alone of up to 6.1 times, up to 2.5 times that of EAGLE-3. The gains are largest on math and code, and smallest on open-ended chat (about 2.8 times). In SGLang on a B200 GPU, the speedup on Qwen3-8B drops from about 4 to 5 times for a single request to about 2.4 to 2.8 times at 32 concurrent requests, because at this load the verification of 16-token blocks becomes compute-bound. Drafters trained on 4K-token contexts also lose acceptance on longer inputs, which a short fine-tune on long contexts recovers.

Multi-token prediction, described below, only helps models pre-trained with MTP modules. A DFlash drafter, in contrast, is trained after the fact against a frozen target, so it can be added to almost any model. At the time of writing, however, its training recipe is not public, so in practice the drafters released by [z-lab](https://github.com/z-lab/dflash) are used, for Qwen3.5 and Qwen3.6, gpt-oss, Gemma 4 and Kimi K2.5, among others. DFlash runs in vLLM and SGLang, and in MLX on Apple Silicon.

### Self-speculative decoding: LayerSkip

The drafter can also be a part of the target itself. [LayerSkip](https://arxiv.org/abs/2404.16710) drafts with the early layers of the target model, and verifies with all of them. In a regular model, however, only the last layer is trained to feed the output head, so the early layers predict the next token poorly. LayerSkip therefore trains the model with two techniques. *Layer dropout* skips layers at random during training, the later layers more often than the early ones, which makes the model less dependent on its later layers. An *early exit loss* trains every layer to predict the next token through the shared output head. At inference, the drafts exit after the first few layers, after 4 of the 8 layers in the figure, and the verification runs only the remaining layers, reusing the computation and the KV cache of the first layers. No second model is stored, and the reported speedups reach 2.16 times on summarization and 1.82 times on coding. The price is paid in training: the model must be pre-trained or fine-tuned with the two techniques.

{: .picture}
<img width="99.9%" height="99.9%" src="{{ site.assets }}/inference/imgs/layerskip.svg"/>

{: .caption}
Every layer also learns to predict through the shared head (early exit loss). The layer dropout rate grows linearly with depth, as in the code; in the paper, it grows exponentially. Inspired by: [LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding](https://arxiv.org/abs/2404.16710)

The following code implements a simplified LayerSkip, with the early exit loss and layer dropout for training, and drafting with the first layers at inference:

```python
class LayerSkip(nn.Module):
    """ LayerSkip, simplified: a decoder whose layers can exit early, through the shared output head (final norm and
        output layer). In training, layer dropout skips layer l of each sequence with a probability that grows with
        depth, and the early exit loss trains every layer to predict the next token. At inference, the first layers
        draft and all the layers verify. embedding(tokens, start): token and positional embeddings of the tokens at
        positions start, start+1, ...; blocks: transformer blocks with causal attention,
        block(x, kv_cache=None) -> (x, kv_cache) """

    def __init__(self, embedding, blocks, norm, head, p_max=0.2):
        super().__init__()
        self.embedding, self.blocks, self.norm, self.head = embedding, nn.ModuleList(blocks), norm, head
        self.p_max = p_max  # layer dropout rate of the last layer

    def forward(self, tokens, kv_cache=None, exit_layer=None):
        """ tokens: [1, S], the new tokens -> logits after the first exit_layer layers (all by default): [1, S, V], and
            the KV cache of these layers, with the new tokens appended """
        blocks = self.blocks[:exit_layer]
        kv_cache = kv_cache or [None] * len(blocks)
        start = 0 if kv_cache[0] is None else kv_cache[0][0].size(2)  # number of cached tokens
        x = self.embedding(tokens, start)                              # [1, S, E]
        new_cache = []
        for block, layer_cache in zip(blocks, kv_cache):
            x, layer_cache = block(x, kv_cache=layer_cache)
            new_cache.append(layer_cache)
        return self.head(self.norm(x)), new_cache

    def loss(self, tokens, targets):
        """ Early exit loss with layer dropout: the cross-entropy of every layer through the shared output head,
            averaged with weights that grow with depth. tokens, targets: [B, S] """
        x, L = self.embedding(tokens, 0), len(self.blocks)  # [B, S, E]
        loss = 0.0
        for l, block in enumerate(self.blocks):
            p = self.p_max * l / max(L - 1, 1) if self.training else 0.0
            keep = torch.rand(x.size(0), 1, 1, device=x.device) >= p  # layer dropout, per sequence: [B, 1, 1]
            x = torch.where(keep, block(x)[0], x)
            logits = self.head(self.norm(x))                           # [B, S, V]
            loss = loss + (l + 1) * F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        return loss / (L * (L + 1) / 2)


# usage: the first layers draft and the whole model verifies. LayerSkip also reuses the KV cache and the hidden states
# of the first layers in the verification, which this sketch recomputes
# early_exit = lambda tokens, kv_cache: model(tokens, kv_cache, exit_layer=4)
# tokens = generate_speculative(model, prompt, n_tokens, DraftModel(early_exit), n_drafts=4)
```

### Multi-token prediction

Multi-token prediction (MTP) goes one step further, and trains the model itself, during pre-training, to predict several future tokens at every position instead of only the next one. This has two benefits. During training, every position receives a denser learning signal, which can improve the representations that the model learns. At inference, the extra predictions serve as drafts: the model drafts for itself, without a separate draft model, and the verification is the same as in speculative decoding.

#### Parallel heads

[Gloeckle et al.](https://arxiv.org/abs/2404.19737) (Meta, 2024) add $$n$$ output heads on top of a shared trunk. The trunk $$f_s$$ computes a representation $$z_t = f_s(x_{1:t})$$ of the context $$x_{1:t}$$, the tokens up to position $$t$$. Head $$i$$, a transformer layer $$f_{h_i}$$ followed by the unembedding $$f_u$$ shared by all heads, predicts the token $$i$$ positions ahead:

$$
P_\theta(x_{t+i} \mid x_{1:t}) = \text{softmax}\left( f_u \left( f_{h_i} ( z_t ) \right) \right)
$$

The training loss is the sum of the cross-entropy losses of the $$n$$ heads, over all the positions of the sequence:

$$
L_n = - \sum_t \sum_{i=1}^{n} \log P_\theta(x_{t+i} \mid x_{1:t})
$$

Head 1 is the usual next-token head, and at inference the other heads provide the drafts, as the figure shows for $$n = 4$$. Training these heads needs some care with memory. The vocabulary is large, so the logits of a head are large too, and the logits of all heads at once would dominate the memory footprint of training. After the forward pass of the trunk, the forward and backward passes of the heads therefore run one after the other, and their gradients accumulate at the output of the trunk, so that only the logits of one head are in memory at a time. The trunk then runs its backward pass once.

The benefits of MTP grow with the model size, and small models can even get worse. The 13B model of the paper solves 12% more problems on HumanEval and 17% more on MBPP than a comparable next-token model. At inference, the extra heads draft for self-speculative decoding, and a model trained to predict 4 tokens decodes up to 3 times faster. However, as in Medusa, each head predicts its token without seeing the tokens predicted by the other heads.

{: .picture}
<img width="87.1%" height="87.1%" src="{{ site.assets }}/inference/imgs/mtp_parallel.svg"/>

{: .caption}
In training, the heads run their forward and backward passes one at a time. Adapted from: [Better & Faster Large Language Models via Multi-token Prediction](https://arxiv.org/abs/2404.19737)

The following code implements the parallel heads, and the backward pass that keeps only the logits of one head in memory:

```python
class MultiTokenHeads(nn.Module):
    """ Multi-token prediction with parallel heads (Gloeckle et al.): head i, a transformer block followed by the output
        layer shared by all heads (final norm and unembedding), predicts the token i positions ahead from the output z
        of the trunk. Head 1 is the usual next-token head. blocks: transformer blocks with causal attention,
        block(x, kv_cache=None) -> (x, kv_cache) """

    def __init__(self, blocks, unembedding):
        super().__init__()
        self.blocks, self.unembedding = nn.ModuleList(blocks), unembedding

    def forward(self, z):
        """ z: [B, S, E] -> logits of the tokens 1, 2, ..., n positions ahead: [B, S, n, V]. At inference, the heads at
            the last position draft the next n tokens, the first of which is the model's own next token """
        return torch.stack([self.unembedding(block(z)[0]) for block in self.blocks], dim=2)

    def loss_backward(self, z, tokens):
        """ Training, instead of loss.backward(): z: [B, S, E], output of the trunk for the input tokens: [B, S]. The
            forward and backward passes of the heads run one after the other, accumulating their gradients at z, so
            that only the logits of one head are in memory at a time; then the trunk runs its backward pass once.
            Returns the loss: the sum of the cross-entropy losses of the heads """
        z_heads = z.detach().requires_grad_()  # the gradients of the heads accumulate in z_heads.grad
        total = 0.0
        for i, block in enumerate(self.blocks, start=1):
            logits = self.unembedding(block(z_heads)[0][:, :-i])  # predictions of the tokens i positions ahead
            loss = F.cross_entropy(logits.flatten(0, 1), tokens[:, i:].flatten())
            loss.backward()
            total += loss.item()
        z.backward(z_heads.grad)
        return total
```

#### Sequential modules

[DeepSeek-V3](https://arxiv.org/abs/2412.19437) predicts the extra tokens *sequentially* instead, and keeps the complete causal chain at each prediction depth. It adds $$D$$ MTP modules. Module $$k$$ has its own transformer block $$\text{TRM}_k$$ and projection matrix $$M_k$$, and shares the embedding layer and the output head with the main model. For the token $$t_i$$ at position $$i$$, module $$k$$ combines the representation $$h_i^{k-1}$$ of the previous depth (the output of the main model, for $$k=1$$) with the embedding of the token $$t_{i+k}$$:

$$
h_i^{\prime k} = M_k \left[ \text{RMSNorm}(h_i^{k-1}) ; \text{RMSNorm}(\text{Emb}(t_{i+k})) \right]
$$

where the brackets denote a concatenation. The module then runs the result through its transformer block, $$h^{k} = \text{TRM}_k(h^{\prime k})$$, and predicts the token $$t_{i+k+1}$$ with the shared output head, $$P^{k}_{i+k+1} = \text{OutHead}(h^{k}_i)$$. Each depth has its own cross-entropy loss $$L^k_\text{MTP}$$, and the average of these losses, weighted by $$\lambda$$, is added to the main next-token loss:

$$
L_\text{MTP} = \frac{\lambda}{D} \sum_{k=1}^{D} L^k_\text{MTP}
$$

In other words, the main model predicts $$t_{i+1}$$. Module 1 combines the hidden state of the main model with the embedding of $$t_{i+1}$$ to predict $$t_{i+2}$$, module 2 combines the hidden state of module 1 with the embedding of $$t_{i+2}$$ to predict $$t_{i+3}$$, and so on, as the figure shows for two modules. Each prediction therefore depends on the tokens before it, unlike the parallel heads of Gloeckle et al., which predict their tokens independently. Each module adds only its projection matrix and its transformer block. DeepSeek-V3 uses a single module ($$D=1$$), with $$\lambda = 0.3$$ for the first 10T training tokens and $$0.1$$ for the remaining 4.8T. Its published MTP weights add 14B parameters to the 671B of the main model.

The authors report better results on most benchmarks, and suggest that MTP may lead the model to pre-plan its representations for better predictions of future tokens. At inference, the module can be discarded, and the model then runs as a regular next-token model, or it can serve as a drafter. As a drafter, its token is accepted 85% to 90% of the time, which gives 1.8 times more tokens per second. An MTP module is close to an EAGLE drafter: both combine the features of the target with the embedding of the next token, run one transformer layer, and predict with the output head of the target. The difference is that the MTP module is trained together with the model, during pre-training.

{: .picture}
<img width="96.8%" height="96.8%" src="{{ site.assets }}/inference/imgs/mtp_sequential.svg"/>

{: .caption}
MTP module $$k$$ combines the hidden state of depth $$k - 1$$ with the embeddings of the input tokens shifted by $$k$$, and predicts the token $$k + 1$$ positions ahead, through the shared output head. Adapted from: [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)

The following code implements the MTP modules of DeepSeek-V3 and their loss:

```python
class MTPModule(nn.Module):
    """ DeepSeek-V3 multi-token prediction module k: combines the hidden states of depth k-1 (the output of the main
        model, for k = 1) with the embeddings of the tokens k positions ahead, runs its transformer block, and predicts
        the tokens k+1 positions ahead. embedding and out_head (final norm and output layer) are shared with the main
        model; block: a transformer block with causal attention, block(x, kv_cache=None) -> (x, kv_cache) """

    def __init__(self, embedding, block, out_head):
        super().__init__()
        E = embedding.embedding_dim
        self.embedding, self.block, self.out_head = embedding, block, out_head
        self.norm_h, self.norm_e = nn.RMSNorm(E), nn.RMSNorm(E)
        self.proj = nn.Linear(2 * E, E, bias=False)  # M_k

    def forward(self, h, tokens):
        """ h: [B, S, E], hidden states of depth k-1 at positions i; tokens: [B, S] at positions i+k -> hidden states
            of depth k at positions i: [B, S, E], and logits of the tokens at positions i+k+1: [B, S, V] """
        x = self.proj(torch.cat([self.norm_h(h), self.norm_e(self.embedding(tokens))], dim=-1))  # [B, S, E]
        h, _ = self.block(x)
        return h, self.out_head(h)


def mtp_loss(modules, h, tokens, lam=0.3):
    """ DeepSeek-V3 MTP loss, added to the next-token loss of the main model. h: [B, S, E], output of the main model
        for the input tokens: [B, S]. Module k predicts the tokens k+1 positions ahead, and the average of the
        cross-entropy losses of the modules is weighted by lam """
    losses = []
    for k, module in enumerate(modules, start=1):
        h, logits = module(h[:, :-1], tokens[:, k:])  # positions i = 0, ..., S-k-1: [B, S-k, E], [B, S-k, V]
        losses.append(F.cross_entropy(logits[:, :-1].flatten(0, 1), tokens[:, k + 1:].flatten()))
    return lam * sum(losses) / len(losses)


# drafting with an MTP module, run again on its own outputs for more drafts, uses the drafting loop of EAGLE:
# propose = lambda tokens, k: draft_with_features(module, features(tokens[:, :-1]), tokens[:, 1:], k)
```

#### Drafting with MTP modules

With a single module, the model drafts one extra token per forward pass. To draft more tokens, inference engines run the same module again on its own output hidden state and draft token, as with EAGLE. The module was trained to look only one token ahead, however, so the acceptance rate drops for the later drafts. SGLang serves DeepSeek-V3 and R1 this way, with their MTP module (called NextN) as the drafter. On DeepSeek-V3 with 16 H200 GPUs, [LMSYS](https://www.lmsys.org/blog/2025-07-17-mtp) reported an average acceptance length of 2.18 tokens with a 3-token MTP window, and of 2.44 tokens with a 4-token window.

Other models train more modules for inference. [MiMo-7B](https://arxiv.org/abs/2505.07608) (Xiaomi) uses a single MTP layer in pre-training, as more layers did not help. It then initializes two more MTP layers from copies of this layer, and fine-tunes them with the main model and the first MTP layer frozen. On AIME24, the drafts of the first MTP layer are accepted about 90% of the time, and those of the third layer more than 75%. Recent models such as Qwen3-Next and Qwen3.5 ship with MTP modules, which vLLM uses as drafters, and its [speculators](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/mtp/) library can fine-tune them for multi-step drafting.

Medusa, EAGLE and DFlash add a drafter to an existing model, and LayerSkip trains the model so that its own early layers can draft. MTP builds the drafter into the model during pre-training: its drafts are cheap and accurate, and it may even improve the model, but only models pre-trained with MTP can use it.

### Summary of speculative decoding methods

The table compares the speculative decoding methods above.

| Method | Source of the drafts | Cost of drafting $$K$$ tokens | Extra training | Extra memory | Reported speedup |
|---|---|---|---|---|---|
| Draft model | a separate small model | $$K$$ passes of the small model | none, or distilling a draft | a second model | about 2-3x |
| Prompt lookup | n-grams of the context | a search in the context | none | none | large on copy-heavy tasks |
| Medusa | extra heads | the heads, in the pass of the target | the heads (or joint fine-tuning) | the heads, with an output layer each | 2.2-3.6x |
| EAGLE-3 | a draft layer on the features of the target | $$K$$ passes of one layer | the draft layer | one layer | up to 6.5x |
| DFlash | a block diffusion drafter on the hidden states of the target | one pass of a few layers | the drafter (target frozen) | a few layers | over 6x, up to 2.5x that of EAGLE-3 |
| LayerSkip | early layers of the model | $$K$$ passes of the early layers | layer dropout and early exit loss, in pre-training or fine-tuning | none | up to about 2x |
| Multi-token prediction | the model's own MTP heads (parallel) or modules (sequential) | the heads, in the pass of the model, or one module pass per draft | pre-training | small (about 2% of DeepSeek-V3) | 1.8x (DeepSeek-V3) to 3x |

The speedups are those reported in the papers, on different models, tasks and hardware, and they shrink as the batch size grows. All these methods are lossless, except Medusa with typical acceptance.

Which method to use depends mostly on the model at hand. A model pre-trained with MTP modules already contains its drafter, which needs no extra training. For any other model, prompt lookup is free, and pays off when the output copies from the input. A separate draft model needs no training when a smaller model of the same family, with the same tokenizer, exists. Drafters trained against the frozen target, such as EAGLE-3 and DFlash, give the largest speedups, at the cost of training the drafter, unless a trained drafter is released for the target. Finally, LayerSkip suits a model that can be pre-trained or fine-tuned with its two techniques.
