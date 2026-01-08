
import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from src_train.modeling.conenet import ConeNet
from src_train.losses.losses import ConeNetLoss
from src_train.data.dataset import DummyConeNetDataset, ConeNetDataset
from src_train.qat.qat_utils import initialize_qat, calibrate_model, HAS_PYTORCH_QUANT
from src_train.sparsity.sparsity_utils import apply_24_sparsity, HAS_APEX_SPARSITY

def collate_fn(batch):
    """
    Custom collate to handle list of targets from ConeNetDataset.
    batch: list of (image, targets)
           targets: list of scale_dicts [P3_dict, P4_dict]
    """
    imgs = torch.stack([item[0] for item in batch])
    
    # targets is a list of lists. We want a list of batch_dicts (one per scale).
    num_scales = len(batch[0][1])
    batch_targets = []
    
    for s in range(num_scales):
        scale_batch = {}
        for key in batch[0][1][s].keys():
            scale_batch[key] = torch.stack([item[1][s][key] for item in batch])
        batch_targets.append(scale_batch)
        
    return imgs, batch_targets

def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch):
    model.train()
    total_loss = 0
    stats_agg = {'hm': 0, 'off': 0, 'wh': 0}
    
    for i, (imgs, targets) in enumerate(dataloader):
        imgs = imgs.to(device)
        targets = [ {k: v.to(device) for k, v in t.items()} for t in targets]
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(imgs)
            loss, stats = criterion(outputs, targets)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        for k in stats_agg:
            stats_agg[k] += stats[k]
            
        if i % 10 == 0:
            print(f"Epoch {epoch} [{i}/{len(dataloader)}] Loss: {loss.item():.4f} (HM: {stats['hm']:.4f}, OFF: {stats['off']:.4f}, WH: {stats['wh']:.4f})")
            
    avg_loss = total_loss / len(dataloader)
    return avg_loss

def main(args):
    if args.qat:
        if HAS_PYTORCH_QUANT:
            initialize_qat()
        else:
            print("ERROR: pytorch-quantization not available. Cannot run QAT.")
            return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Model
    model = ConeNet(deploy=False).to(device)
    
    # 2. Dataset
    if args.dummy:
        print("Using Dummy Dataset")
        dataset = DummyConeNetDataset(num_samples=args.batch_size * 5)
    else:
        dataset = ConeNetDataset(args.img_dir, args.label_dir)
        
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=collate_fn)
    
    # 3. Loss, Optimizer, Scheduler
    criterion = ConeNetLoss().to(device)
    
    # 3.1 QAT Calibration
    if args.qat and args.calibrate:
        print("Starting QAT Calibration...")
        # Use a separate loader for calibration to avoid shuffling issues if needed
        calib_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
        calibrate_model(model, calib_loader, device, num_batches=args.calib_batches)
        
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # 3.2 Structured Sparsity (2:4)
    asp = None
    if args.sparsity:
        if HAS_APEX_SPARSITY:
            # We apply only to Stage 3 and 4 as per RESEARCH.md
            whitelist = ['stage3', 'stage4']
            asp = apply_24_sparsity(model, optimizer, whitelist=whitelist)
            # Crucial: Initialize optimizer for pruning awareness
            asp.init_optimizer_for_pruning(optimizer)
        else:
            print("ERROR: apex.contrib.sparsity not available. Cannot apply sparsity.")
            return
    
    # OneCycleLR is great for fast convergence
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, 
                                            steps_per_epoch=len(dataloader), 
                                            epochs=args.epochs)
    
    scaler = GradScaler()
    
    # 4. Training Loop
    best_loss = float('inf')
    os.makedirs(args.work_dir, exist_ok=True)
    
    for epoch in range(args.epochs):
        avg_loss = train_one_epoch(model, dataloader, criterion, optimizer, scaler, device, epoch)
        scheduler.step()
        
        print(f"Epoch {epoch} Average Loss: {avg_loss:.4f}")
        
        # Save Checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(args.work_dir, 'best_model.pth'))
            print("Saved Best Model")
            
        if epoch % 5 == 0:
            torch.save(model.state_dict(), os.path.join(args.work_dir, f'epoch_{epoch}.pth'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, default='data/images')
    parser.add_argument('--label_dir', type=str, default='data/labels')
    parser.add_argument('--work_dir', type=str, default='work_dirs/conenet_v1')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--dummy', action='store_true', help='Use dummy data for testing')
    parser.add_argument('--qat', action='store_true', help='Enable Quantization Aware Training')
    parser.add_argument('--calibrate', action='store_true', help='Run calibration before starting QAT')
    parser.add_argument('--calib_batches', type=int, default=8, help='Number of batches for calibration')
    parser.add_argument('--sparsity', action='store_true', help='Enable 2:4 Structured Sparsity (Stage 3 & 4)')
    
    args = parser.parse_args()
    main(args)
