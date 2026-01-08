
import sys
import os
import torch
import pytest

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src_train.modeling.conenet import ConeNet

def test_conenet_output_shape():
    model = ConeNet(deploy=False)
    # Input: (Batch, 4, H, W) -> 1920x640
    input_tensor = torch.randn(1, 4, 640, 1920)
    
    outputs = model(input_tensor)
    p3, p4 = outputs
    
    # Check P3 shape
    # Downsample factor: 2 (stem) * 2 (s1) * 2 (s2) * 2 (s3) = 16
    # H = 640 / 16 = 40
    # W = 1920 / 16 = 120
    assert p3.shape == (1, 5, 40, 120), f"Expected P3 shape (1, 5, 40, 120), got {p3.shape}"
    
    # Check P4 shape
    # Downsample factor: 16 * 2 (s4) = 32
    # H = 640 / 32 = 20
    # W = 1920 / 32 = 60
    assert p4.shape == (1, 5, 20, 60), f"Expected P4 shape (1, 5, 20, 60), got {p4.shape}"
    
    print("Output shapes verified.")

def test_conenet_3_channel_input():
    model = ConeNet(deploy=False)
    # Input: (Batch, 3, H, W)
    input_tensor = torch.randn(1, 3, 640, 1920)
    
    outputs = model(input_tensor)
    p3, p4 = outputs
    assert p3.shape == (1, 5, 40, 120)
    assert p4.shape == (1, 5, 20, 60)
    print("3-channel input padding verified.")

def test_deploy_conversion():
    model = ConeNet(deploy=False)
    model.eval()
    input_tensor = torch.randn(1, 4, 640, 1920)
    
    with torch.no_grad():
        out_train = model(input_tensor)
        
    model.switch_to_deploy()
    
    with torch.no_grad():
        out_deploy = model(input_tensor)
        
    # Check logic error: closeness
    # Note: RepVGG conversion can have small numerical differences (1e-5 range)
    for o_t, o_d in zip(out_train, out_deploy):
        diff = (o_t - o_d).abs().max()
        assert diff < 1e-4, f"Deploy conversion mismatch max diff: {diff}"
        
    print("Deploy conversion verified.")

if __name__ == "__main__":
    test_conenet_output_shape()
    test_conenet_3_channel_input()
    test_deploy_conversion()
