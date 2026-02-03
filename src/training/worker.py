import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import os
from ray import train, tune
from src.nas.search_space import SuperNet

def get_data_loaders(batch_size=64, data_dir="./data"):
    """
    Setup standard CIFAR-10 dataloaders.
    In a real enterprise setup, this would connect to S3 or a Feature Store.
    """
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Download only once in a real cluster usage, ensuring thread safety
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)

    # For demo speed, use a subset if needed, but we'll use full here
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader

def train_nas_candidate(config):
    """
    The training loop managed by Ray Tune.
    """
    net = SuperNet(config)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=config.get("lr", 1e-3))
    
    # Checkpointing setup
    checkpoint = train.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            model_state = torch.load(os.path.join(checkpoint_dir, "model.pth"))
            net.load_state_dict(model_state)
            
    train_loader, test_loader = get_data_loaders(batch_size=config.get("batch_size", 64))
    
    for epoch in range(10):  # Maximum epochs per trial
        net.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        # Validation
        net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        accuracy = correct / total
        
        # Report metrics to Ray Tune (Triggers Early Stopping if configured)
        # Checkpointing logic
        with train.Checkpoint.from_directory(os.getcwd()) as checkpoint:
             torch.save(net.state_dict(), "model.pth")
             
        train.report({"loss": running_loss / len(train_loader), "accuracy": accuracy})
