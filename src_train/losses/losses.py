
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ModifiedFocalLoss(nn.Module):
    """
    Modified Focal Loss from CenterNet.
    Penalty-reduced pixelwise logistic regression with focal loss.
    """
    def __init__(self, alpha=2.0, beta=4.0):
        super(ModifiedFocalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, gt):
        """
        pred: (batch, c, h, w) - heatmap prediction (after sigmoid/relu+clip) 
              Research.md specifically says ReLU in model, but traditionally heatmap needs 0-1.
              We assume the model output is raw logits and we apply sigmoid here?
              Actually Research.md says: "Channel 0 (Heatmap): Probabilita' di presenza..."
              A probability implies [0,1].
              If the head output is plain Conv, we need to clamp or sigmoid.
              Common practice in CenterNet: Sigmoid is part of the loss or head 'post-process'.
              Given "ReLU" constraint in backbone, head usually ends with Conv1x1.
              We'll apply sigmoid here for stability if not already applied.
        gt: (batch, c, h, w) - heatmap ground truth with gaussian blobs
        """
        # Clamp for numerical stability
        pred = torch.clamp(pred, min=1e-4, max=1-1e-4)

        pos_inds = gt.eq(1)
        neg_inds = gt.lt(1)

        neg_weights = torch.pow(1 - gt, self.beta)

        loss = 0

        pos_loss = torch.log(pred) * torch.pow(1 - pred, self.alpha) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, self.alpha) * neg_weights * neg_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = -neg_loss
        else:
            loss = -(pos_loss + neg_loss) / num_pos
            
        return loss

class RegL1Loss(nn.Module):
    """
    L1 loss for quantization error offsets.
    Only calculated at positive locations.
    """
    def __init__(self):
        super(RegL1Loss, self).__init__()
    
    def forward(self, pred, gt, mask):
        """
        pred: (batch, 2, h, w) - predicted offsets
        gt: (batch, 2, h, w) - ground truth offsets
        mask: (batch, 1, h, w) - mask of positive locations (1 at cone centers)
        """
        # Expand mask to match channel dim
        mask = mask.expand_as(pred)
        loss = F.l1_loss(pred * mask, gt * mask, reduction='sum')
        num_pos = mask.float().sum() + 1e-4 # + epsilon
        loss = loss / (num_pos + 1e-4)
        return loss

class CIoULoss(nn.Module):
    """
    Complete IoU Loss for size regression.
    """
    def __init__(self):
        super(CIoULoss, self).__init__()

    def forward(self, pred_wh, gt_wh, mask):
        """
        pred_wh: (batch, 2, h, w) - predicted width, height
        gt_wh: (batch, 2, h, w) - ground truth width, height
        mask: (batch, 1, h, w)
        """
        mask = mask.expand_as(pred_wh)
        pred_wh = pred_wh * mask
        gt_wh = gt_wh * mask
        
        # We process only positive samples to save computation/avoid zeros
        # Flattening
        pred_w = pred_wh[:, 0, :, :][mask[:, 0, :, :] > 0]
        pred_h = pred_wh[:, 1, :, :][mask[:, 0, :, :] > 0]
        gt_w = gt_wh[:, 0, :, :][mask[:, 0, :, :] > 0]
        gt_h = gt_wh[:, 1, :, :][mask[:, 0, :, :] > 0]
        
        if len(pred_w) == 0:
            return torch.tensor(0.0, device=pred_wh.device)

        # Intersection
        inter_w = torch.min(pred_w, gt_w)
        inter_h = torch.min(pred_h, gt_h)
        inter = inter_w * inter_h
        
        # Union
        union = (pred_w * pred_h) + (gt_w * gt_h) - inter
        iou = inter / (union + 1e-6)
        
        # DIoU term (Distance)
        # CenterNet regresses size (w, h) at the specific center point.
        # But for CIoU we conceptually treating them as boxes centered at (0,0) relative to each other?
        # Actually CIoU needs center distance. 
        # In CenterNet 'Size' head, we are regressing w/h. The center is implicitly "matched" by logic.
        # However, slight offset errors exist.
        # Regular CenterNet uses L1 for Size. 
        # RESEARCH.md explicitly asks for CIoU.
        # CIoU = IoU - (rho^2 / c^2) - alpha * v
        # Since we use 'offset' head for center regression, we can assume centers are aligned?
        # OR we should combine offset_pred + grid vs offset_gt + grid.
        # Let's simplify: The Loss is applied on the Size prediction.
        # If we assume centers are perfectly aligned (since we compute loss only on GT center), 
        # distance term (rho) is 0? No, that would make it IoU - 0 - alpha*v.
        # BUT, the network predicts size independently of offset.
        # Usually CIoU is for Bounding Box regression (x,y,w,h).
        # Should we construct the box from (center_gt, size_pred) and (center_gt, size_gt)?
        # If centers are same, rho=0. Then DIoU -> IoU.
        # CIoU Spec Section 4: CIoU = IoU - aspect_diff * 0.1
        # Our previous manual implementation was slightly different. 
        # Using the simplified Spec version for consistency.
        loss = 1.0 - (iou - aspect_diff * 0.1)
        return loss.mean()

class ConeNetLoss(nn.Module):
    def __init__(self, lambda_hm=1.0, lambda_off=1.0, lambda_wh=0.1):
        super(ConeNetLoss, self).__init__()
        self.heatmap_loss = ModifiedFocalLoss()
        self.offset_loss = RegL1Loss()
        self.wh_loss = CIoULoss()
        
        self.lambda_hm = lambda_hm
        self.lambda_off = lambda_off
        self.lambda_wh = lambda_wh

    def forward(self, outputs, batch_targets):
        """
        outputs: list of [p3_out, p4_out]
                 Each out: (N, 5, H, W) -> 0:hm, 1-2:off, 3-4:size
        batch_targets: list of targets for [p3, p4]
        """
        total_loss = 0
        loss_stats = {'hm': 0, 'off': 0, 'wh': 0}

        for i, output in enumerate(outputs):
            target = batch_targets[i] # {'hm': ..., 'ind': ..., 'off': ..., 'wh': ..., 'mask': ...}
            
            # Split output
            pred_hm = torch.sigmoid(output[:, 0:1, :, :]) # Heatmap
            pred_off = output[:, 1:3, :, :]               # Offset
            pred_wh = output[:, 3:5, :, :]                # Size
            
            gt_hm = target['hm']
            mask = target['mask'] # (N, 1, H, W) - 1 at cone centers
            gt_off = target['off']
            gt_wh = target['wh']
            
            # Heatmap Loss
            h_loss = self.heatmap_loss(pred_hm, gt_hm)
            
            # Offset Loss
            o_loss = self.offset_loss(pred_off, gt_off, mask)
            
            # Size Loss
            w_loss = self.wh_loss(pred_wh, gt_wh, mask)
            
            total_loss += self.lambda_hm * h_loss + self.lambda_off * o_loss + self.lambda_wh * w_loss
            
            loss_stats['hm'] += h_loss.item()
            loss_stats['off'] += o_loss.item()
            loss_stats['wh'] += w_loss.item()
            
        return total_loss, loss_stats
