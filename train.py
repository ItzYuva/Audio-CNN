import os
import sys
import pandas as pd
import modal
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import torchaudio
import torch
import torch.nn as nn
import torchaudio.transforms as T
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from model import AudioCNN


app = modal.App("Audio-CNN")

image = (modal.Image.debian_slim()
         .pip_install_from_requirements("requirements.txt")
         .apt_install(["wget", "ffmpeg", "unzip", "libsndfile1"])  # ✅ fixed typo in package name
         .run_commands([
             "cd /tmp && wget https://github.com/karolpiczak/ESC-50/archive/master.zip -O esc50.zip",
             "cd /tmp && unzip esc50.zip",
             "mkdir -p /opt/esc50-data",
             "cp -r /tmp/ESC-50-master/* /opt/esc50-data/",
             "rm -rf /tmp/esc50.zip /tmp/ESC-50-master"
         ])
         .add_local_python_source("model"))

volume = modal.Volume.from_name("esc50-data", create_if_missing=True)
model_volume = modal.Volume.from_name("esc-model", create_if_missing=True)


class ESC50Dataset(Dataset):
    def __init__(self, data_dir, metadata_file, split="train", transform=None):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform

        # ✅ FIX 1: Previously you tried self.metadata before defining it.
        # Read CSV into a temp DataFrame first.
        df = pd.read_csv(metadata_file)

        if split == "train":
            self.metadata = df[df['fold'] != 5].copy()
        else:
            self.metadata = df[df['fold'] == 5].copy()

        self.classes = sorted(self.metadata['category'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.metadata.loc[:, 'label'] = self.metadata['category'].map(self.class_to_idx)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        audio_path = self.data_dir / "audio" / row['filename']

        waveform, sample_rate = torchaudio.load(audio_path)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if self.transform:
            spectogram = self.transform(waveform)
        else:
            spectogram = waveform

        return spectogram, row['label']

def mixup_data(x, y):
    lam = np.random.beta(0.2, 0.2)
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    # (0.7 * audio) + (0.3 * audio)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return criterion(pred, y_a) * lam + criterion(pred, y_b) * (1 - lam)


@app.function(image=image, gpu="A10G", volumes={"/data": volume, "/models": model_volume}, timeout=60*60*3)
def train():
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f'/models/tensorboard_logs/run_{timestamp}'
    writer = SummaryWriter(log_dir)

    # FIX 2: Use /data for caching, not /opt
    dataset_path = "/data/ESC-50-master"

    # Download only if dataset not already cached
    if not os.path.exists(dataset_path):
        print("Dataset not found in volume. Downloading...")
        os.system("wget https://github.com/karolpiczak/ESC-50/archive/master.zip -O /tmp/esc50.zip")
        os.system("unzip /tmp/esc50.zip -d /tmp")
        os.system("cp -r /tmp/ESC-50-master /data/")  # ✅ copy entire folder
        os.system("rm -rf /tmp/esc50.zip /tmp/ESC-50-master")
        print("Dataset downloaded and cached in volume.")
    else:
        print("Dataset already cached. Skipping download.")

    # FIX 3: Use cached dataset path
    esc50_dir = Path(dataset_path)

    train_transform = nn.Sequential(
        T.MelSpectrogram(
            sample_rate=44100,
            n_fft=1024,
            hop_length=512,
            n_mels=128,
            f_min=0,
            f_max=11025
        ),
        T.AmplitudeToDB(),
        T.FrequencyMasking(freq_mask_param=30),
        T.TimeMasking(time_mask_param=80)
    )

    val_transform = nn.Sequential(
        T.MelSpectrogram(
            sample_rate=44100,
            n_fft=1024,
            hop_length=512,
            n_mels=128,
            f_min=0,
            f_max=11025
        ),
        T.AmplitudeToDB()
    )

    train_dataset = ESC50Dataset(
        data_dir=esc50_dir,
        metadata_file=esc50_dir / "meta" / "esc50.csv",
        split="train",
        transform=train_transform
    )

    val_dataset = ESC50Dataset(
        data_dir=esc50_dir,
        metadata_file=esc50_dir / "meta" / "esc50.csv",
        split="test",
        transform=val_transform
    )

    print("Training samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioCNN(num_classes=len(train_dataset.classes))
    model.to(device)

    num_epochs = 100
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)

    scheduler = OneCycleLR(
        optimizer,
        max_lr=0.002,
        steps_per_epoch=len(train_dataloader),
        epochs=num_epochs,
        pct_start=0.1
    )

    best_accuracy = 0.0

    print("Start training")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for data, target in progress_bar:
            data, target = data.to(device), target.to(device)

            if np.random.random() > 0.7:
                data, target_a, target_b, lam = mixup_data(data, target)
                output = model(data)
                loss = mixup_criterion(criterion, output, target_a, target_b, lam)
            else:
                output = model(data)
                loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})

        avg_epoch_loss = epoch_loss/len(train_dataloader)
        writer.add_scalar('Loss/train', avg_epoch_loss, epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)

        # Validation after each epoch
        model.eval()

        correct = 0
        total = 0
        val_loss = 0

        with torch.no_grad():
            for data, target in test_dataloader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                loss = criterion(outputs, target)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = 100 * correct/total
        avg_val_loss = val_loss/len(test_dataloader)
        writer.add_scalar('Loss/validation', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/validation', accuracy, epoch)

        print(f'Epoch {epoch + 1}, Loss: {avg_epoch_loss:.4f}, Val loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%')

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'accuracy': accuracy,
                'classes': train_dataset.classes
            }, '/models/best_model.pth')

            print(f"New best model saved with accuracy: {best_accuracy:.2f}%")

    writer.close()
    print(f"Training completed! Best accuracy: {best_accuracy:.2f}%")


@app.local_entrypoint()
def main():
    train.remote()

