
import os
import argparse
import torch
from src_train.modeling.conenet import ConeNet
from src_train.qat.qat_utils import initialize_qat, export_qat_onnx, HAS_PYTORCH_QUANT

def main(args):
    # 1. Initialize QAT if needed BEFORE model creation
    if args.qat:
        if HAS_PYTORCH_QUANT:
            initialize_qat()
        else:
            print("ERROR: pytorch-quantization not available. Cannot export QAT ONNX.")
            return

    # 2. Load Model
    model = ConeNet(deploy=False)
    
    if args.weights:
        print(f"Loading weights from {args.weights}")
        state_dict = torch.load(args.weights, map_location='cpu')
        model.load_state_dict(state_dict)
    
    model.eval()
    
    # 3. Handle Deploy Mode (Fusing)
    if args.fuse and not args.qat:
        print("Fusing branches (RepVGG switch_to_deploy)...")
        model.switch_to_deploy()
    elif args.fuse and args.qat:
        print("WARNING: Fusing after QAT training is complex. Usually, QAT should be done on the fused model if possible, or keeping branches. DLA prefers fused 3x3.")
        # For DLA, we want a single 3x3. If we QAT the branches, we can't easily fuse 'perfectly' in INT8.
        # However, RepVGG can be fused in FP32 and THEN QAT-ed.
        # If weights are already from a multi-branch QAT training, we might need a special fusion.
        model.switch_to_deploy()

    # 4. Export
    dummy_input = torch.randn(1, 4, args.height, args.width)
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    if args.qat:
        export_qat_onnx(model, dummy_input, args.output)
    else:
        print(f"Exporting standard ONNX to {args.output}...")
        torch.onnx.export(
            model,
            dummy_input,
            args.output,
            opset_version=12,
            input_names=['input'],
            output_names=['p3', 'p4'],
            dynamic_axes={'input': {0: 'batch'}, 'p3': {0: 'batch'}, 'p4': {0: 'batch'}} if args.dynamic else None
        )
    
    print("Export complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, help='Path to .pth weights')
    parser.add_argument('--output', type=str, default='work_dirs/conenet.onnx', help='Output ONNX path')
    parser.add_argument('--height', type=int, default=640)
    parser.add_argument('--width', type=int, default=1920)
    parser.add_argument('--qat', action='store_true', help='Export with Q/DQ nodes')
    parser.add_argument('--fuse', action='store_true', help='Fuse RepVGG branches before export')
    parser.add_argument('--dynamic', action='store_true', help='Use dynamic axes')
    
    args = parser.parse_args()
    main(args)
