import torch
from torch import nn
import torchvision
from torchvision import transforms, datasets
#from sklearn.model_selection import train_test_split


class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.fc1 = nn.Linear(1600, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        # your forward pass here
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = torch.relu(x)
        output = self.fc2(x)
        return output
        
model = DigitCNN()

if __name__ == "__main__":
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root = './data', train = True, transform=transform)

    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size

    # split
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # reproducible split
    )
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=64, shuffle=False)

    num_epochs = 10
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

    #best_train_loss = float('inf') 
    train_loss_total = 0.0
    val_loss_total = 0.0
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        train_loss_total = 0.0
        val_loss_total = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            train_loss = criterion(outputs, labels)
            train_loss.backward()
            optimizer.step()
            train_loss_total += train_loss.item()
        
        avg_train_loss = train_loss_total / len(train_loader) 
        #print(f"Epoch {epoch+1}, Loss: {train_loss.item():.4f}")
        
        
        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                # forward pass only, no backward
                #optimizer.zero_grad()
                outputs = model(images)
                val_loss_total += criterion(outputs, labels).item()
                #optimizer.step()
        avg_val_loss = val_loss_total / len(val_loader)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        avg_val_loss = 0.0
        avg_train_loss = 0.0 
    

