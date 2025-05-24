import torch
import torch.nn as nn
import torch.optim as optim

# Input and expected output (1 sample, 4 features)
x = torch.tensor([[1,2,3,4]], dtype=torch.float32)
y_true = torch.tensor([[1,2,3,4]], dtype=torch.float32)

# Define the model
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 4)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# Instantiate the model
model = SimpleNet()

# Custom weights and biases from user
W = torch.tensor([
    [1.22192, -0.516964, 0.869636, 0.721333],
    [1.58856, 1.61822, -0.187125, -1.18831],
    [-0.06342, -0.687744, -0.385892, 0.849739],
    [-1.05039, 0.297398, 0.957779, 1.32076]
], dtype=torch.float32)

b = torch.tensor([1.22192, -0.516964, 0.869636, 0.721333], dtype=torch.float32)

# Apply to both layers
with torch.no_grad():
    model.fc1.weight.copy_(W)
    model.fc1.bias.copy_(b)
    model.fc2.weight.copy_(W)
    model.fc2.bias.copy_(b)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# ----- Forward, Backward, and Update -----
y_pred = model(x)
loss = criterion(y_pred, y_true)
print(f"Initial Loss: {loss.item():.4f}")

optimizer.zero_grad()
loss.backward()
optimizer.step()

# ----- Print updated weights and biases -----
print("\nUpdated Weights and Biases:")

print("W1:")
print(model.fc1.weight.data)
print("b1:")
print(model.fc1.bias.data)

print("W2:")
print(model.fc2.weight.data)
print("b2:")
print(model.fc2.bias.data)
