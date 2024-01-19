---
layout: post
title:  "Finetuning and Inference on a GPT model"
categories: [machine learning, Transformer, GPT, DeepSpeed, inference, mixture-of-experts]
tags: [machinelearning]
---

In this we will look at the inference problem. We will look at 
- how to fine-tune very large models using LoRA,
- how to use KD to perform layer and parameter prunning, 
- how to improve accuracy via mixture of experts


# ZeRO Inference

See [ZeRO-Inference](https://www.deepspeed.ai/2022/09/09/zero-inference.html)

# ZeRO ++

Mention ZeRO++ for fine-tuning? 

# Speeding up finetuning

Mention compression and LoRA for fine-tuning and faster inference?

# LORA/QLORA

# Flash Attention

Speeding up transformer training/inference can be achieved by two memory optimization technique which does not require modification of the mode: flash attention and continuous batching. about continuous batching, see [here](https://www.anyscale.com/blog/continuous-batching-llm-inference)

