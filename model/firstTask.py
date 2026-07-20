'''
Simplest possible neural network in PyTorch — a single linear layer that takes 784 inputs (28×28 flattened) and outputs 10 classes.
No convolutions yet.
'''
import torch
import torch.nn as nn
from torchvision import datasets, transforms


class firstTask(nn.Module):
    def __init__(self):
        super().__init__()
        # define your layer here
        self.hidden = nn.Linear(784, 256)
        self.output = nn.Linear(256, 10)
    
    def forward(self, x):
        #define the forward pass here
        x = torch.relu(self.hidden(x))
        x = self.output(x)
        return x
    
    
#loading the database
model = firstTask()

transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

num_epochs = 10
#train_dataset = torch.utils.data.TensorDataset("/Users/agrimsharma/Desktop/Projects/VNN/archive")
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)


criterion = nn.CrossEntropyLoss() # just measure loss
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001) # just update weights
    
for epoch in range(num_epochs):
    for images, labels in train_loader:
        # 1. flatten images from (batch, 1, 28, 28) to (batch, 784)
        #.view() that can be used to reshape a tensor without changing its data. 
        # -1 means "infer this dimension based on the other dimensions and the total number of elements in the tensor" 
        # kind of letting pytorch figuring it out themselves and making the last three dimensions (1, 28, 28)/(Channels, Rows, Cols) into 1
        images = images.view(images.shape[0], -1)
        #cross entropy loss only knows how to measure how wrong the predictions are with no knowledge of the weights
        # optimizer() if what holds the actual references to thw weights and how to update them using gradients
        #so .zero_grad() and step() are methods of the optimizer, not the loss function
        #criterion = nn.CrossEntropyLoss() # just measure loss
        #optimizer = torch.optim.adam(model.parameters(), lr = 0.001) # just update weights
        # 2. zero gradients
        optimizer.zero_grad()
        # 3. forward pass
        outputs = model(images)
        # 4. compute loss
        loss = criterion(outputs, labels)
        # 5. backward pass
        loss.backward() #gradients updated here
        # 6. update weights
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")


# Example input
x = torch.randn(1, 784)
y = model(x)
print("Output with torch.nn:", y)