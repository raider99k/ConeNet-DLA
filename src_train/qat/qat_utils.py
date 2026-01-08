
import torch
import torch.nn as nn
from tqdm import tqdm

try:
    from pytorch_quantization import nn as quant_nn
    from pytorch_quantization import calib
    from pytorch_quantization import tensor_quant
    from pytorch_quantization.tensor_quant import QuantDescriptor
    HAS_PYTORCH_QUANT = True
except ImportError:
    HAS_PYTORCH_QUANT = False
    print("WARNING: pytorch-quantization not found. QAT features will be disabled.")

def initialize_qat():
    """
    Initializes quantization modules. 
    This must be called BEFORE model instantiation.
    """
    if not HAS_PYTORCH_QUANT:
        return
    
    from pytorch_quantization import quant_modules
    # This automatically replaces nn.Conv2d, nn.Linear, etc with Quant counterparts
    quant_modules.initialize()
    print("PyTorch Quantization modules initialized.")

def set_quantizer_fast(model):
    """
    Configures all quantizers in the model to use Max/Min or histogram.
    """
    if not HAS_PYTORCH_QUANT:
        return
        
    for name, module in model.named_modules():
        if isinstance(module, (quant_nn.TensorQuantizer,)):
            if module._amin is not None:
                module._amin.fill_(0)
            if module._amax is not None:
                module._amax.fill_(0)

def calibrate_model(model, dataloader, device, num_batches=8, method='percentile'):
    """
    Runs calibration on the model using a few batches of data.
    """
    if not HAS_PYTORCH_QUANT:
        return
        
    model.eval()
    model.to(device)
    
    # Configure quantizers for calibration
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_calib()
                module.disable_quant()
            else:
                print(f"Quantizer {name} has no calibrator.")

    print(f"Running calibration ({method})...")
    with torch.no_grad():
        for i, (imgs, _) in enumerate(tqdm(dataloader)):
            if i >= num_batches:
                break
            model(imgs.to(device))

    # Compute amax values
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if method == 'percentile':
                    module.set_amax(module._calibrator.compute_amax(method='percentile', percentile=99.9))
                elif method == 'mse':
                    module.set_amax(module._calibrator.compute_amax(method='mse'))
                else:
                    module.set_amax(module._calibrator.compute_amax())
                
                module.disable_calib()
                module.enable_quant()
                
    print("Calibration complete.")

def disable_quantization(model):
    if not HAS_PYTORCH_QUANT:
        return
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            module.disable_quant()

def enable_quantization(model):
    if not HAS_PYTORCH_QUANT:
        return
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            module.enable_quant()

def export_qat_onnx(model, dummy_input, path):
    """
    Exports the model with Q/DQ nodes to ONNX.
    """
    if not HAS_PYTORCH_QUANT:
        print("ERROR: Cannot export QAT ONNX without pytorch-quantization.")
        return

    model.eval()
    
    # Ensure quantizers are enabled and in inference mode
    quant_nn.TensorQuantizer.use_fb_fake_quant = True
    
    print(f"Exporting QAT ONNX to {path}...")
    torch.onnx.export(
        model, 
        dummy_input, 
        path, 
        verbose=False, 
        opset_version=13, 
        # do_constant_folding=True,
        input_names=['input'],
        output_names=['p3', 'p4']
    )
    print("Export complete.")
