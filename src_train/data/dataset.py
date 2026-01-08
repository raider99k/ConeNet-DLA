
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from .encoding import ConeNetEncoder

class ConeNetDataset(Dataset):
    """
    Standard Dataset for ConeNet. 
    Expects labels in YOLO format (normalized cx, cy, w, h) but converts them
    to absolute coordinates for the encoder.
    """
    def __init__(self, img_dir, label_dir, input_size=(640, 1920), strides=[16, 32], num_classes=1, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.input_h, self.input_w = input_size
        self.encoder = ConeNetEncoder(input_size=input_size, strides=strides, num_classes=num_classes)
        self.transform = transform
        
        self.img_names = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.label_dir, os.path.splitext(img_name)[0] + '.txt')
        
        # Load Image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        orig_h, orig_w = image.shape[:2]
        
        # Resize to input size
        image = cv2.resize(image, (self.input_w, self.input_h))
        
        # Load Labels
        boxes = []
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    cls, cx, cy, w, h = map(float, line.split())
                    # Convert normalized to absolute (input size)
                    abs_cx = cx * self.input_w
                    abs_cy = cy * self.input_h
                    abs_w = w * self.input_w
                    abs_h = h * self.input_h
                    
                    boxes.append([abs_cx, abs_cy, abs_w, abs_h])
                    labels.append(int(cls))
        
        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)
        
        # Encode targets
        targets = self.encoder.encode(boxes, labels)
        
        # Normalize and convert image to tensor
        # RESEARCH.md Section 6.2: "inputIOFormats=int8:chw" implies we might want to handle it.
        # But for training we usually use float32 [0, 1]
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1) # (3, H, W)
        
        # Add 4th channel (padding) if required by the model (Handled in model.forward too, but better here?)
        # RESEARCH.md Section 3.2: "Input: 1920x640 pixel (RGB). Crop verticale... 3 (4 pad)"
        # Actually Section 6.2 says the engine accepts int8:chw.
        # For training, we keep 3 channels and the model pads it to 4 if needed.
        
        return image, targets

class DummyConeNetDataset(Dataset):
    """
    Useful for testing the training loop without a real dataset.
    """
    def __init__(self, num_samples=10, input_size=(640, 1920), strides=[16, 32], num_classes=1):
        self.num_samples = num_samples
        self.input_h, self.input_w = input_size
        self.encoder = ConeNetEncoder(input_size=input_size, strides=strides, num_classes=num_classes)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Dummy Image
        image = torch.randn(3, self.input_h, self.input_w)
        
        # Random Box
        cx = np.random.uniform(0, self.input_w)
        cy = np.random.uniform(0, self.input_h)
        w = np.random.uniform(20, 200)
        h = np.random.uniform(20, 200)
        
        boxes = np.array([[cx, cy, w, h]], dtype=np.float32)
        labels = np.array([0], dtype=np.int32)
        
        targets = self.encoder.encode(boxes, labels)
        
        return image, targets
