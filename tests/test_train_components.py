
import sys
import os
import torch
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src_train.losses.losses import ConeNetLoss
from src_train.data.encoding import ConeNetEncoder

def test_encoder_and_loss():
    # 1. Setup Encoder
    encoder = ConeNetEncoder(input_size=(640, 1920), strides=[16, 32], num_classes=1)
    
    # 2. Setup Dummy Labels (Batch of 2)
    # Box format: [cx, cy, w, h]
    # Image 1: Small cone (P3) at (100, 100), size 20x40.
    # Image 2: Large cone (P4) at (400, 400), size 100x200.
    
    # Encoder expects list of boxes, list of labels.
    # We test single image for simplicity of loop in test, but loss takes batch.
    
    boxes_1 = np.array([[100, 100, 20, 40]], dtype=np.float32)
    labels_1 = np.array([0], dtype=np.int32)
    
    boxes_2 = np.array([[400, 400, 100, 200]], dtype=np.float32)
    labels_2 = np.array([0], dtype=np.int32)
    
    # Encode
    targets_1 = encoder.encode(boxes_1, labels_1) # [P3_dict, P4_dict]
    targets_2 = encoder.encode(boxes_2, labels_2)
    
    # Check shape
    # P3 stride 16 -> 640/16=40, 1920/16=120
    assert targets_1[0]['hm'].shape == (1, 40, 120)
    assert targets_1[1]['hm'].shape == (1, 20, 60)
    
    # Check content
    # Small cone should be in P3
    assert targets_1[0]['mask'].sum() == 1
    assert targets_1[1]['mask'].sum() == 0
    
    # Large cone should be in P4
    assert targets_2[0]['mask'].sum() == 0
    assert targets_2[1]['mask'].sum() == 1
    
    print("Encoder output shapes and assignment logic verified.")
    
    # 3. Setup Loss
    criterion = ConeNetLoss()
    
    # Create batch
    # Targets needs to be list of dicts per scale.
    # Encoder returns [P3, P4] for one image.
    # Batch keys: P3_batch, P4_batch.
    
    # Stack targets
    batch_targets = []
    
    # P3
    p3_hm = torch.stack([targets_1[0]['hm'], targets_2[0]['hm']])
    p3_off = torch.stack([targets_1[0]['off'], targets_2[0]['off']])
    p3_wh = torch.stack([targets_1[0]['wh'], targets_2[0]['wh']])
    p3_mask = torch.stack([targets_1[0]['mask'], targets_2[0]['mask']])
    batch_targets.append({'hm': p3_hm, 'off': p3_off, 'wh': p3_wh, 'mask': p3_mask})
    
    # P4
    p4_hm = torch.stack([targets_1[1]['hm'], targets_2[1]['hm']])
    p4_off = torch.stack([targets_1[1]['off'], targets_2[1]['off']])
    p4_wh = torch.stack([targets_1[1]['wh'], targets_2[1]['wh']])
    p4_mask = torch.stack([targets_1[1]['mask'], targets_2[1]['mask']])
    batch_targets.append({'hm': p4_hm, 'off': p4_off, 'wh': p4_wh, 'mask': p4_mask})
    
    # 4. Dummy Model Output
    # Shape: (N, 5, H, W). 5 = 1(hm) + 2(off) + 2(size)
    p3_pred = torch.randn(2, 5, 40, 120, requires_grad=True)
    p4_pred = torch.randn(2, 5, 20, 60, requires_grad=True)
    
    outputs = [p3_pred, p4_pred]
    
    # 5. Compute Loss
    loss, stats = criterion(outputs, batch_targets)
    
    print(f"Loss computed: {loss.item()}")
    print(f"Stats: {stats}")
    
    # Backprop check
    loss.backward()
    print("Backward pass successful.")

if __name__ == "__main__":
    test_encoder_and_loss()
