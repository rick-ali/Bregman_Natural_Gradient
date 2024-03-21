import torch
from torch import nn
from models import SimpleMLP, BinaryMLP
from metrics import SquaredMetric, BCEMetric, KLMetric, KLMetric_p2
from ngd import MetricGD
from torch.optim import SGD, Adam
from torchmetrics.regression import MeanSquaredError
from torchmetrics.classification import BinaryAccuracy


def train(config, device, logger=None, log_G=False, G_logger=None):
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
        if config.optimizer == 'p2':
            metric = KLMetric_p2()
        else:
            metric = KLMetric()
    if config.optimizer == 'ngd':
        optimizer = MetricGD(model.parameters(), lr=config.lr)
    elif config.optimizer == 'bgd': 
        optimizer = MetricGD(model.parameters(), lr=config.lr)
    elif config.optimizer == 'p2': 
        optimizer = MetricGD(model.parameters(), lr=config.lr)
    elif config.optimizer == 'sgd':
        optimizer = SGD(model.parameters(), lr=config.lr)
    elif config.optimizer == 'adam':
        optimizer = Adam(model.parameters(), lr=config.lr)

    X = X.to(device)
    y = y.to(device)
    model.to(device)

    if log_G:
        assert G_logger is not None

    for epoch in range(config.num_epochs):
        pred_y = model(X).view(-1)
        if config.optimizer == 'ngd':
            G = []
            for i, param in enumerate(model.parameters()):
                n = torch.numel(param)
                G.append(torch.zeros(n, n, device=device))

            if log_G:
                num_param = sum(p.numel() for p in model.parameters())
                F = torch.zeros(num_param, num_param)

            log_pred_y = torch.log(pred_y)
            log_pred_1my = torch.log(1 - pred_y)
            for x in range(len(X)):
                optimizer.zero_grad()
                log_pred_y[x].backward(retain_graph=True)

                if log_G:
                    J = torch.cat([torch.cat([param.grad.flatten()]) for param in model.parameters()]).view(-1, 1)
                    F += pred_y[x] * J @ J.T
                for i, param in enumerate(model.parameters()):
                    dp = param.grad.view(-1, 1)
                    G[i] += pred_y[x] * dp @ dp.T
                optimizer.zero_grad()
                log_pred_1my[x].backward(retain_graph=True)
                for i, param in enumerate(model.parameters()):
                    dp = param.grad.view(-1, 1)
                    G[i] += (1 - pred_y[x]) * dp @ dp.T
                if log_G:
                    J = torch.cat([torch.cat([param.grad.flatten()]) for param in model.parameters()]).view(-1, 1)
                    F += pred_y[x] * J @ J.T
            for i, param in enumerate(model.parameters()):
                G[i] /= len(X) 
            if log_G:
                F /= len(X)
                G_logger.log_metrics({"metric": F.tolist()})
        elif config.optimizer == 'bgd' or config.optimizer == "p2":
            hessian = metric(pred_y, y) 
            G = []
            for i, param in enumerate(model.parameters()):
                n = torch.numel(param)
                G.append(torch.zeros(n, n, device=device))

            if log_G:
                num_param = sum(p.numel() for p in model.parameters())
                F = torch.zeros(num_param, num_param)
            for x in range(len(X)):
                optimizer.zero_grad()
                pred_y[x].backward(retain_graph=True)
                if log_G:
                    J = torch.cat([torch.cat([param.grad.flatten()]) for param in model.parameters()]).view(-1, 1)
                    if config.loss_type == 'kl':
                        J = torch.hstack([J, -J])
                    F += J @ hessian[x] @ J.T
                for i, param in enumerate(model.parameters()):
                    dp = param.grad.view(-1, 1)
                    if config.loss_type == 'kl':
                        dp = torch.hstack([dp, -dp])
                    G[i] += (dp @ hessian[x] @ dp.T)
       
            for i, param in enumerate(model.parameters()):
                G[i] /= len(X)       
            if log_G:
                F /= len(X)
                G_logger.log_metrics({"metric": F.tolist()})

        if config.loss_type == 'kl':
            preds = torch.hstack([pred_y, 1 - pred_y])
            y_ = torch.hstack([y, 1 - y])
            log_pred = torch.log(preds)
            loss = criterion(log_pred, y_)
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
        if config.optimizer == 'ngd' or config.optimizer == 'bgd' or config.optimizer == "p2":
            optimizer.defaults["metric"] = G
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{config.num_epochs}], Loss: {loss.item():.4f}")

    if logger is not None:
        logger.save()
    return model, X

def test(config, device, model, X=None, logger=None):
    if X is None:
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
    
    X = X.to(device)
    y = y.to(device)
    model.to(device)
    model.eval()

    if logger is not None:
        with torch.no_grad():
            pred_y = model(X).view(-1)
            if config.loss_type == 'mse':
                mse = MeanSquaredError()
                logger.log_metrics({'loss' : mse(pred_y, y)})
            if config.loss_type == 'kl':
                acc = BinaryAccuracy()
                logger.log_metrics({'accuracy': acc(pred_y, y)})
            logger.save()

         