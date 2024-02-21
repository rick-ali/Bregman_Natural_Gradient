import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define a simple MLP model
class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.layers(x)

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define the model, loss function, and optimizer
input_size = 10
hidden_size = 20
output_size = 1
model = SimpleMLP(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Generate random training data
num_samples = 1000
input_data = torch.randn(num_samples, input_size)
target_data = torch.randn(num_samples, output_size)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(input_data)
    
    # Compute the loss
    loss = criterion(outputs, target_data)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Print the loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# After training, you can use the trained model for predictions
# Example prediction for a new input
new_input = torch.randn(1, input_size)
prediction = model(new_input)
print("Prediction:", prediction.item())
