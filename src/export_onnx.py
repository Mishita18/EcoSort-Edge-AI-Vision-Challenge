import argparse, os, sys
import torch
from model import EcoSortNet
MAX_PARAMS=350_000
MAX_BYTES=int(1.5*1024*1024)
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",default="outputs/best_model.pth")
    p.add_argument("--output",default="model.onnx")
    a=p.parse_args()
    ckpt=torch.load(a.checkpoint,map_location="cpu",weights_only=False)
    model=EcoSortNet(num_classes=6); model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    params=sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}/{MAX_PARAMS:,}")
    if params>MAX_PARAMS: raise SystemExit("Parameter limit failed")
    dummy=torch.randn(1,3,96,96)
    torch.onnx.export(model,dummy,a.output,export_params=True,opset_version=18,
        do_constant_folding=True,input_names=["input"],output_names=["output"],
        dynamic_axes={"input":{0:"batch_size"},"output":{0:"batch_size"}},dynamo=False)
    size=os.path.getsize(a.output)
    print(f"ONNX size: {size/(1024*1024):.3f} MiB / 1.500 MiB")
    if size>MAX_BYTES: raise SystemExit("File-size limit failed")
    try:
        import onnx
        onnx.checker.check_model(onnx.load(a.output))
        import onnxruntime as ort, numpy as np
        sess=ort.InferenceSession(a.output,providers=["CPUExecutionProvider"])
        out=sess.run(["output"],{"input":np.zeros((1,3,96,96),dtype=np.float32)})[0]
        assert out.shape==(1,6)
        print("ONNX Runtime smoke test PASS; output shape",out.shape)
    except ImportError:
        print("Install onnx and onnxruntime to run full ONNX verification.")
if __name__=="__main__": main()
