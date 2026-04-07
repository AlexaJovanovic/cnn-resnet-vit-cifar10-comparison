import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim
import pandas as pd
import time

def train_model(
        model: nn.Module,
        tr_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        sheduler: torch.optim.lr_scheduler,
        criterion: nn.Module,
        num_epochs: int,
        device,
        batch_transform = None):    
    # move model to target device
    model.to(device)
    
    tr_losses = []
    tr_accs = []
    val_losses = []
    val_accs = []

    start_time = time.time()
    for epoch in range(num_epochs):
        # Training loop
        model.train()
        training_loss = 0
        tr_n_correct = 0
        tr_n_total = 0

        for imgs, labels in tr_loader:
            # set previous gradients to zero
            optimizer.zero_grad()
            
            imgs = imgs.to(device)
            labels = labels.to(device)

            if batch_transform is not None:
                imgs, labels = batch_transform(imgs, labels)
            # forward
            pred = model(imgs)
            loss = criterion(pred, labels)

            # backward
            loss.backward()
            optimizer.step()

            # info
            training_loss += loss.item()
            _, pred_labels = torch.max(pred, 1)
            
            if labels.ndim == 2:  # soft labels because of batch transform
                hard_labels = labels.argmax(dim=1)
            else:  # normal labels
                hard_labels = labels
            
            tr_n_correct += (pred_labels == hard_labels).sum().item()
            tr_n_total += labels.shape[0]

        # Validation loop
        model.eval()
        val_loss = 0
        val_n_correct = 0
        val_n_total = 0

        with torch.no_grad():
            for imgs, labels in val_loader:    
                imgs = imgs.to(device)
                labels = labels.to(device)

                # forward
                pred = model(imgs)
                loss = criterion(pred, labels)

                # info
                val_loss += loss.item()
                _, pred_labels = torch.max(pred, 1)
                val_n_correct += (pred_labels == labels).sum().item()
                val_n_total += labels.shape[0]

        # Logging for the user
        training_loss = training_loss / len(tr_loader)
        tr_acc = tr_n_correct/tr_n_total * 100
        val_loss = val_loss / len(val_loader)
        val_acc = val_n_correct/val_n_total * 100
        
        # --- Logging for the user ---
        print(f"\n[Epoch {epoch+1}/{num_epochs}]")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"  Training:   Loss {training_loss:.4f} | Acc {tr_acc:.2f}%")
        print(f"  Validation: Loss {val_loss:.4f} | Acc {val_acc:.2f}%")
        print("-" * 40)

        tr_losses.append(training_loss), tr_accs.append(tr_acc), val_losses.append(val_loss), val_accs.append(val_acc)
        
        # Move scheduler one step forward
        sheduler.step()
    
    total_time = time.time() - start_time
    print("Model training finished!")
    
    metrics_df = pd.DataFrame({
        "train_loss": tr_losses,
        "train_acc": tr_accs,
        "val_loss": val_losses,
        "val_acc": val_accs
    })
    metrics_df["total_time"] = total_time

    return metrics_df

