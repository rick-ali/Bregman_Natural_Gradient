import torch 
import numpy as np
from torch import nn
from models import SimpleMLP
from metrics import SquaredMetric
from ngd import NGD
from torch.optim import SGD

def train(accelerator, config, logger=None):
    input_size = 2
    X = torch.randn(config.num_samples, input_size)
    if config.bop == 'max':
        y, _ = torch.max(X,dim=1)
    if config.model_type == 'simple':
        model = SimpleMLP(input_size=input_size, 
                          output_size=1,
                          hidden_size=10)
    if config.loss_type == 'mse':
        criterion = nn.MSELoss()
        metric = SquaredMetric()
    if config.pullback:
        optimizer = NGD(model.parameters(), lr=config.lr)
    else:
        optimizer = SGD(model.parameters(), lr=config.lr)
    
    device = accelerator.device
    X = X.to(device)
    y = y.to(device)
    model.to(device)
    model, optimizer = accelerator.prepare(model, optimizer)

    for epoch in range(config.num_epochs):
        pred_y = model(X).view(-1)
        if config.pullback:
            hessian = metric(pred_y, y)
            hessian_sqrt = hessian ** 0.5
            f_x = torch.sum(pred_y * hessian_sqrt) / len(pred_y) ** 0.5
            optimizer.zero_grad()
            # f_x.backward(retain_graph=True)
            accelerator.backward(f_x, retain_graph=True)
            G = []
            for param in model.parameters():
                dp = param.data.view(-1, 1)
                G.append(dp @ dp.T)

        loss = criterion(pred_y, y)
        if logger is not None:
            logger.log_hyperparams({'bop': config.bop,
                                        'num_samples': config.num_samples,
                                        'model_type': config.model_type,
                                        'loss_type': config.loss_type,
                                        'num_epochs': config.num_epochs,
                                        'pullback': config.pullback,
                                        'lr': config.lr})
            logger.log_metrics({'loss': loss, 
                                'epoch' : epoch})
        optimizer.zero_grad()
        # loss.backward()
        accelerator.backward(loss)
        if config.pullback:
            optimizer.defaults['metric'] = G
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{config.num_epochs}], Loss: {loss.item():.4f}')
    
    if logger is not None:
        logger.save()

if __name__ == '__main__':
    train()

    