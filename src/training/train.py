import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets.mitbih_loader import MITBIHDataset
from models.hybrid_cnn_lstm import HybridCNNLSTM
from training.trainer import Trainer
from training.loss import AttentionRegularizedLoss

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    train_dataset = MITBIHDataset(data_dir="data/mitbih/train", preload=True)
    val_dataset = MITBIHDataset(data_dir="data/mitbih/val", preload=True)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

    # Model
    model = HybridCNNLSTM(input_channels=1, num_classes=5)

    # Loss
    criterion = AttentionRegularizedLoss(alpha=1e-4)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # Trainer
    trainer = Trainer(model=model,
                      train_loader=train_loader,
                      val_loader=val_loader,
                      criterion=criterion,
                      optimizer=optimizer,
                      device=device,
                      num_epochs=50,
                      log_interval=10)

    # Train
    trainer.fit()

if __name__ == "__main__":
    main()
