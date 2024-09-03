from onnxruntime.datasets import get_example

example1 = get_example("mul_1.onnx")
import os
import onnx  # noqa: E402
from onnx.tools.net_drawer import GetOpNodeProducer, GetPydotGraph  # noqa: E402

for name in ['model', 'model_dynamo', 'model_compile_dynamo', 'model_jit_script', 'model_jit_trace']:
  onnx_model = onnx.load(f"{name}.onnx")
  onnx.checker.check_model(onnx_model)

  # convert to a graph in .dot format
  pydot_graph = GetPydotGraph(
    onnx_model.graph, name=onnx_model.graph.name, rankdir="LR", node_producer=GetOpNodeProducer("docstring"))
  pydot_graph.write_dot(f"{name}.dot")

  # convert .dot into a .png image
  os.system(f"dot -Tpng {name}.dot -Gdpi=900 > {name}.png")


