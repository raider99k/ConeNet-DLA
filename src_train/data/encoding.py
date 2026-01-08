
import numpy as np
import torch
import math

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    
    return heatmap

def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    
    return min(r1, r2, r3)

class ConeNetEncoder:
    def __init__(self, input_size=(640, 1920), strides=[16, 32], num_classes=1):
        self.input_h, self.input_w = input_size
        self.strides = strides
        self.num_classes = num_classes
        
        # P3 shape: H/16, W/16 (40, 120)
        # P4 shape: H/32, W/32 (20, 60)

    def encode(self, boxes, labels):
        """
        boxes: np.array of shape (N, 4) in [x_min, y_min, x_max, y_max] or similar format?
               Actually commonly [cx, cy, w, h] or [x1, y1, x2, y2]. 
               Let's assume [center_x, center_y, w, h] absolute coordinates.
        labels: np.array (N), class indices.
        """
        
        targets = []
        
        for stride in self.strides:
            output_h = self.input_h // stride
            output_w = self.input_w // stride
            
            # Init targets
            hm = np.zeros((self.num_classes, output_h, output_w), dtype=np.float32)
            wh = np.zeros((2, output_h, output_w), dtype=np.float32)
            off = np.zeros((2, output_h, output_w), dtype=np.float32)
            # Mask to indicate where objects are
            mask = np.zeros((1, output_h, output_w), dtype=np.float32)
            
            num_objs = len(boxes)
            
            for k in range(num_objs):
                box = boxes[k]
                cls_id = int(labels[k])
                
                cx, cy, w, h = box
                
                # Assign to scale
                # Usually we assign to P3 or P4 based on size.
                # Research.md doesn't explicitly specify scale assignment rule, 
                # but "P3 (Coni Piccoli) -> Stage 3" and "P4 (Coni Grandi) -> Stage 4".
                # Let's define a heuristic threshold.
                # Common threshold: area or max dim.
                # 32 pixel stride means it sees larger context.
                # Small cones at distance -> P3. Large cones close up -> P4.
                # Let's say: if max(w,h) > 64 pixels -> P4, else P3.
                # This is heuristic. Need to match stride.
                # If stride=16, we handle specific range.
                # Let's assume input targets list has filtered boxes for this stride?
                # Or we decide here.
                
                is_p4 = (w * h) > (32 * 32) # Heuristic: Area > 1024
                
                if (stride == 32 and not is_p4) or (stride == 16 and is_p4):
                    continue
                
                # Project, floor
                ct = np.array([cx / stride, cy / stride], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                
                # Check bounds
                if not (0 <= ct_int[0] < output_w and 0 <= ct_int[1] < output_h):
                    continue
                
                # Draw Gaussian
                # Box size in output map
                h_out = h / stride
                w_out = w / stride
                radius = gaussian_radius((math.ceil(h_out), math.ceil(w_out)))
                radius = max(0, int(radius))
                
                draw_gaussian(hm[cls_id], ct_int, radius)
                
                # Size (w, h) - Model predicts absolute or relative?
                # Research md: "Channel 3-4 (Size): Larghezza e altezza del box (w, h)."
                # Typically normalized or absolute. 
                # Let's keep it in "output stride pixels" or "input pixels"?
                # RESEARCH explicitly says "Output tensori puri".
                # CenterNet usually predicts size in *input resolution* but sometimes *output resolution*.
                # If we use L1 loss, we want manageable values.
                # Let's predict size in Input Pixels (like official CenterNet) but we can also do Stride pixels.
                # Given QAT/INT8, maybe smaller numbers are better?
                # Let's predict Log(size)? No, standard is raw size.
                # Let's stick to Input Resolution Size (w, h) as per standard CenterNet. 
                # But wait, CIoU works on boxes.
                wh[0, ct_int[1], ct_int[0]] = w 
                wh[1, ct_int[1], ct_int[0]] = h
                
                # Offset (x, y) - "Offset x, y per recuperare l'errore di discretizzazione"
                off[0, ct_int[1], ct_int[0]] = ct[0] - ct_int[0]
                off[1, ct_int[1], ct_int[0]] = ct[1] - ct_int[1]
                
                mask[0, ct_int[1], ct_int[0]] = 1
                
            targets.append({
                'hm': torch.from_numpy(hm),
                'off': torch.from_numpy(off),
                'wh': torch.from_numpy(wh),
                'mask': torch.from_numpy(mask)
            })
            
        return targets # List [P3_target, P4_target] based on strides order
