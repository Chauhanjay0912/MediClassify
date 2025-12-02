import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
from sklearn.model_selection import train_test_split

from config import *
from dataset import SkinLesionDataset
from model import get_model
from utils import train_model, evaluate_model

# Data transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def main():
    print(f"Using device: {DEVICE}")
    
    # Load data
    df_full = pd.read_csv(METADATA_PATH)
    train_df, val_df = train_test_split(df_full, test_size=0.2, random_state=42, stratify=df_full['diagnostic'])
    
    # Create datasets
    train_dataset = SkinLesionDataset(train_df, IMAGE_DIR, transform=data_transforms['train'])
    val_dataset = SkinLesionDataset(val_df, IMAGE_DIR, transform=data_transforms['val'], label_encoder=train_dataset.le)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Train baseline model
    print("\n--- Training Baseline Model ---")
    baseline_model = get_model(num_classes=len(train_dataset.classes), device=DEVICE)
    
    # Class weights for imbalanced data
    if USE_CLASS_WEIGHTS:
        class_counts = train_df['diagnostic'].value_counts().sort_index()
        class_weights = torch.FloatTensor([1.0 / count for count in class_counts]).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, baseline_model.parameters()), 
                            lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3) if USE_SCHEDULER else None
    
    baseline_model, history = train_model(baseline_model, criterion, optimizer, NUM_EPOCHS_BASELINE, 
                                          train_loader, val_loader, DEVICE, scheduler, PATIENCE if USE_EARLY_STOPPING else 999)
    
    # Evaluate
    print("\n--- Evaluating Baseline Model ---")
    accuracy = evaluate_model(baseline_model, val_loader, train_dataset.classes, DEVICE)
    print(f"Baseline Model Accuracy: {accuracy:.4f}")
    
    # Save model
    torch.save(baseline_model.state_dict(), 'baseline_model.pth')
    print("Model saved to baseline_model.pth")

if __name__ == "__main__":
    main()
