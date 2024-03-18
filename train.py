import torch
from torch import nn
from models import SimpleMLP, BinaryMLP
from metrics import SquaredMetric, BCEMetric, KLMetric
from ngd import MetricGD
from torch.optim import SGD, Adam


def train(config, device, logger=None):
    X = torch.randn(config.num_samples, config.input_dim)
    if config.bop == "max":
        y, _ = torch.max(X, dim=1)
    if config.bop == "add":
        y = X[:, 0] + X[:, 1]
    if config.bop == "sub":
        y = X[:, 0] - X[:, 1]
    if config.bop == "unit":
        y = X[:, 0] ** 2 + X[:, 1] ** 2
        y = y < 1
        y = y.float()

    if config.model_type == "simple":
        model = SimpleMLP(
            input_size=config.input_dim,
            output_size=config.output_dim,
            hidden_size=config.hidden_dim,
        )
    if config.model_type == "binary":
        model = BinaryMLP(
            input_size=config.input_dim,
            output_size=config.output_dim,
            hidden_size=config.hidden_dim,
        )
    if config.loss_type == "mse":
        criterion = nn.MSELoss()
        metric = SquaredMetric()
    if config.loss_type == "bce":
        criterion = nn.BCELoss()
        metric = BCEMetric()
    if config.loss_type == "kl":
        criterion = nn.KLDivLoss()
        metric = KLMetric()
    if config.optimizer == 'ngd':
        optimizer = MetricGD(model.parameters(), lr=config.lr)
    elif config.optimizer == 'bgd': 
        optimizer = MetricGD(model.parameters(), lr=config.lr)
    elif config.optimizer == 'sgd':
        optimizer = SGD(model.parameters(), lr=config.lr)
    elif config.optimizer == 'adam':
        optimizer = Adam(model.parameters(), lr=config.lr)

    X = X.to(device)
    y = y.to(device)
    model.to(device)

    if config.optimizer == 'ngd':
        assert config.loss_type == 'kl'

    for epoch in range(config.num_epochs):
        pred_y = model(X).view(-1)
        if config.optimizer == 'ngd':
            G = []
            for i, param in enumerate(model.parameters()):
                n = torch.numel(param)
                G.append(torch.zeros(n, n, device=device))

            log_pred_y = torch.log(pred_y)
            log_pred_1my = torch.log(1 - pred_y)
            for x in range(len(X)):
                optimizer.zero_grad()
                log_pred_y[x].backward(retain_graph=True)
                for i, param in enumerate(model.parameters()):
                    dp = param.grad.view(-1, 1)
                    G[i] += pred_y[x] * dp @ dp.T
                optimizer.zero_grad()
                log_pred_1my[x].backward(retain_graph=True)
                for i, param in enumerate(model.parameters()):
                    dp = param.grad.view(-1, 1)
                    G[i] += (1 - pred_y[x]) * dp @ dp.T
            for i, param in enumerate(model.parameters()):
                    G[i] /= len(X) 
        
        elif config.optimizer == 'bgd':
            hessian = metric(pred_y, y) 
            G = []
            for i, param in enumerate(model.parameters()):
                n = torch.numel(param)
                G.append(torch.zeros(n, n, device=device))

            for x in range(len(X)):
                optimizer.zero_grad()
                pred_y[x].backward(retain_graph=True)
                for i, param in enumerate(model.parameters()):
                    dp = param.grad.view(-1, 1)
                    if config.loss_type == 'kl':
                        dp = torch.hstack([dp, -dp])
                    G[i] += (dp @ hessian[x] @ dp.T)
                    # G[i] += (1/pred_y[x] * dp @ dp.T)
       
            for i, param in enumerate(model.parameters()):
                    G[i] /= len(X)       

        if config.loss_type == 'kl':
            log_pred = torch.log(pred_y)
            loss = criterion(log_pred, y)
        else:
            loss = criterion(pred_y, y)

        if logger is not None:
            logger.log_hyperparams(
                {
                    "bop": config.bop,
                    "num_samples": config.num_samples,
                    "model_type": config.model_type,
                    "loss_type": config.loss_type,
                    "num_epochs": config.num_epochs,
                    "optimizer": config.optimizer,
                    "lr": config.lr,
                }
            )
            logger.log_metrics({"loss": loss, "epoch": epoch})
        optimizer.zero_grad()
        loss.backward()
        if config.optimizer == 'ngd' or config.optimizer == 'bgd':
            optimizer.defaults["metric"] = G
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{config.num_epochs}], Loss: {loss.item():.4f}")

    if logger is not None:
        logger.save()

