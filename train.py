import os
import argparse
from dataclasses import asdict
from typing import Dict

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from datasets import DataConfig, make_dataloaders
from models import make_model
from utils.seed import set_seed

def freeze_backbone_params(model: nn.Module):
    # common patterns: convnext/vit/tinycnn/frcnn backbone
    for name, p in model.named_parameters():
        p.requires_grad = True

    # try freezing all but classifier/head
    for name, p in model.named_parameters():
        lname = name.lower()
        if ("classifier" in lname) or ("head" in lname) or ("heads" in lname):
            p.requires_grad = True
        else:
            p.requires_grad = False

def train_one_epoch(model, loader, optimizer, device, use_amp, scaler, use_grad_clip, grad_clip):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    running = 0.0
    correct = 0
    total = 0

    for x, y in tqdm(loader, desc="train", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with autocast():
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            if use_grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            if use_grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        running += loss.item() * x.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += x.size(0)

    return running / max(total, 1), correct / max(total, 1)

@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    running = 0.0
    correct = 0
    total = 0
    for x, y in tqdm(loader, desc="val", leave=False):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        running += loss.item() * x.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return running / max(total, 1), correct / max(total, 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", type=str, default="convnext_tiny")
    ap.add_argument("--dataset", type=str, default="cifar10")
    ap.add_argument("--pretrained", type=int, default=1)

    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--img_size", type=int, default=224)

    # toggles
    ap.add_argument("--use_amp", type=int, default=1)
    ap.add_argument("--use_scheduler", type=int, default=1)
    ap.add_argument("--use_grad_clip", type=int, default=0)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--use_weight_decay", type=int, default=1)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--freeze_backbone", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--ckpt_dir", type=str, default="checkpoints")
    args = ap.parse_args()

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else print("cpu")

    dcfg = DataConfig(
        name=args.dataset,
        batch_size=args.batch_size,
        img_size=args.img_size,
    )
    train_loader, val_loader, num_classes, preprocess_info = make_dataloaders(dcfg)

    model = make_model(args.arch, num_classes=num_classes, pretrained=bool(args.pretrained))
    if args.freeze_backbone:
        freeze_backbone_params(model)

    model.to(device)

    wd = args.weight_decay if args.use_weight_decay else 0.0
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=wd)

    scheduler = None
    if args.use_scheduler:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    use_amp = bool(args.use_amp) and (device == "cuda")
    scaler = GradScaler(enabled=use_amp)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    best_acc = -1.0
    best_path = None

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, device, use_amp,
            scaler, bool(args.use_grad_clip), args.grad_clip
        )
        va_loss, va_acc = eval_epoch(model, val_loader, device)

        if scheduler is not None:
            scheduler.step()

        print(f"ep {epoch:02d} | train {tr_loss:.4f}/{tr_acc:.4f} | val {va_loss:.4f}/{va_acc:.4f}")

        if va_acc > best_acc:
            best_acc = va_acc
            best_path = os.path.join(args.ckpt_dir, f"best_{args.arch}_{args.dataset}.pt")
            torch.save({
                "model_state": model.state_dict(),
                "arch": args.arch,
                "dataset": args.dataset,
                "num_classes": num_classes,
                "pretrained": bool(args.pretrained),
                "img_size": args.img_size,
                "preprocess": preprocess_info,
                "args": vars(args),
            }, best_path)

    print(f"Best val acc: {best_acc:.4f}")
    print(f"Saved: {best_path}")

if __name__ == "__main__":
    main()
