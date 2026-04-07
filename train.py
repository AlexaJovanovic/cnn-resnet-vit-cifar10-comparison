import torch
from data import get_loaders
from train_model import train_model
from models.cnn import ConvNet
from models.residual_net import ResidualNet, ResNetBottleneck
from models.transformer import VisionTransformer
from wakepy import keep

def get_model(name):
    if name == "cnn":
        return ConvNet(c_in=3, height=32, width=32, n_classes=10)
    elif name == "resnet":
        return ResidualNet()
    elif name == "resnet_bn":
        return ResNetBottleneck()
    elif name == "vit":
        return VisionTransformer(
            patch_size=4,
            d_emb=288,
            n_blocks=6,
            n_heads=4,
            n_classes=10,
            p_dropout=0.2
        )
    else:
        raise ValueError("Unknown model")

def run_training(model_name, epochs, lr, batch_size, weight_decay):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = get_model(model_name).to(device)
    train_loader, val_loader = get_loaders(batch_size)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs+1
    )

    criterion = torch.nn.CrossEntropyLoss()

    metrics_df = train_model(
        model=model,
        tr_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        sheduler=scheduler,
        criterion=criterion,
        num_epochs=epochs,
        device=device
    )

    # Save metrics data
    metrics_df.to_csv(f"training_metrics/{model_name}.csv", index=False)

    # Save model parameters
    torch.save(model.state_dict(), f"trained_models/{model_name}.pth")


if __name__ == "__main__":

    model_names = ["cnn", "resnet", "resnet_bn", "vit"]

    with keep.running():
        for m_name in model_names:
            run_training(
                model_name=m_name,
                epochs=50,
                lr=6e-4,
                batch_size=128,
                weight_decay=0.1
            )