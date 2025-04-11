import os
import sys
import torch
import torch.nn.functional as F
import torch.jit

device='cpu'

#use the GPTlite and Benchmark models from the folder GPT-lite
current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(current_dir, '..', 'GPT-lite'))
from gptlite import GPTlite, Block, block_size, n_embd, n_head
n_head=2

def main():
  B, T, C = 2, block_size, n_embd
  embd = torch.ones((B, T, C)).to(device)
  model = Block(n_embd, n_head).to(device)
  
  # vocab_size, batch_size = 65, 1 
  # embd = torch.randint(0, vocab_size, (batch_size, block_size)).to(device)
  # model = GPTlite(vocab_size).to(device)

  # if we set to eval, traces will differ: eg dropout is disabled, batch-norm uses running stats to normalize data
  # model.eval()

  # export the modet with dynamo (beta)
  onnx_program = torch.onnx.dynamo_export(model, embd)
  onnx_program.save("model_dynamo.onnx")

  # export the model with stable export
  def export_dynamo_onnx(model, name, mode='train'):
    export_output = torch.onnx.dynamo_export(
       model.eval() if mode=='eval' else model.train(), embd)
    export_output.save(f"{name}.onnx")

  def export_torch_jit_onnx(model, name, mode='train'):
      torch.onnx.export(
        model.eval() if mode=='eval' else model.train(), # model being run
        embd,                      # model input
        f"{name}.onnx",            # where to save the model
        export_params=True,        # store the trained parameter weights inside the model file
        opset_version=10,          # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names = ['input'],   # the model's input names
        output_names = ['output'], # the model's output names
        dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                      'output' : {0 : 'batch_size'}})

  #you can use either torch.jit or torch.compile compiler, but not both 
  #also, dynamo does not work with torch.jit
      
  # export_torch_jit_onnx(model, "model") # same as model_jit_trace
  export_torch_jit_onnx(torch.jit.trace(model, embd), "model_jit_trace") #needs input for trace
  export_torch_jit_onnx(torch.jit.script(model), "model_jit_script")

  export_dynamo_onnx(model, "model_dynamo") # new Dynamo exporter
  export_dynamo_onnx(torch.compile(model), "model_compile_dynamo")
  #same as model_dynamo, torch.compile calls TorchDynamo under the hood

  # torch compile vs torch export:
  # https://pytorch.org/docs/stable/export.html#existing-frameworks

if __name__ == "__main__":
  main()
