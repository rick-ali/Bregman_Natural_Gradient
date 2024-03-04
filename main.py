import click
import torch 
import numpy as np
from torch import nn
from models import SimpleMLP
from metrics import SquaredMetric
from ngd import NGD
from torch.optim import SGD

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

@click.command()
@click.option('--bop', default='max', help='Binary operation.')
@click.option('--num_samples', default=1000, help='Input Size.')
@click.option('--model_type', default='simple', help='Neural network.')
@click.option('--loss_type', default='mse', help='Loss type.')
@click.option('--num_epochs', default=100, help='Number of epochs.')
@click.option('--pullback', default=True, help='Use metric pullback.')
@click.option('--lr', default=0.01, help='Learning rate.')
def train(bop, num_samples, model_type, loss_type, num_epochs, pullback, lr):
    input_size = 2
    X = torch.randn(num_samples, input_size)
    if bop == 'max':
        y, _ = torch.max(X,dim=1)
    if model_type == 'simple':
        model = SimpleMLP(input_size=input_size, 
                          output_size=1,
                          hidden_size=100)
    if loss_type == 'mse':
        criterion = nn.MSELoss()
        metric = SquaredMetric()
    if pullback:
        optimizer = NGD(model.parameters(), lr=lr)
    else:
        optimizer = SGD(model.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        pred_y = model(X)
        if pullback:
            hessian = torch.mean(metric(pred_y, y)).view(1, 1)
            f_x = torch.sum(pred_y) / len(pred_y)
            optimizer.zero_grad()
            f_x.backward(retain_graph=True)
            df_dw = [param.data 
                    for param in model.parameters()]

        loss = criterion(pred_y, y)
        optimizer.zero_grad()
        loss.backward()
        if pullback:
            optimizer.step(metric=hessian, df_dw=df_dw)
        else:
            optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

if __name__ == '__main__':
    train()

    