Support material for the post [Building a GPT model in C++ LibTorch, and benchmarking against PyTorch and TorchScript](https://brunomaga.github.io/GPT-lite-cpp).

# How to:

1. Run `python main.py` to benchmark the Python implementations;
   - It will output several `*.pt` files, which are the binaries of the `torch::jit::Module` models benchmarked;
2. Use `cmake` to compute the C++ executable as `./main`. Then run `./main` to benchmark the C++ implementation;
   - Optionally, run `./main [pt-files-folder]` and it will load the `*pt` models output by step 1, and benchmark the `torch::jit::script` implementation as well

Terminal output shows the runtime and throughput for both train and inference steps:
```
Wide DNN train runtime: 7.027491569519043 seconds
Wide DNN train throughput: 1.42298285256917 epochs/second
Wide DNN inference runtime: 21.002371311187744 seconds
Wide DNN inference throughput: 4.761367110328682 epochs/second
```

