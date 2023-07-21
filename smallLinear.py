import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn, Tensor

from utils3 import log1mexp
import math


class BCEWithLogSigmoidLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, log_sigmoid: Tensor, target: Tensor) -> Tensor:
        return torch.neg(
            torch.mul(log_sigmoid, target)
            + torch.mul(1 - target, log1mexp(log_sigmoid))
        )


class FC_BE(nn.Module):
    def __init__(self, L=512, alpha=0.2):
        super(FC_BE, self).__init__()
        self.L = L
#         self.objective = torch.nn.BCEWithLogitsLoss()
        self.objective = BCEWithLogSigmoidLoss()
        self.estimator = nn.Parameter(torch.Tensor([alpha]))

        self.fc = nn.Sequential(
        #    nn.Linear(512 * 49, 512 * 7),
        #    nn.ReLU(),
        #    nn.Linear(512 * 7, 512),
        #    nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        torch.nn.init.xavier_uniform_(self.fc[0].weight)
        self.fc[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc[3].weight)
        self.fc[3].bias.data.zero_()

        self.classifier = nn.Sequential(
            nn.Linear(128, 1),
            nn.LogSigmoid()
        )

        torch.nn.init.xavier_uniform_(self.classifier[0].weight)
        self.classifier[0].bias.data.zero_()

    def forward(self, x):
        # Trash first dimension
        x = x.squeeze()
        if len(x.shape) > 2:
#            x = x.mean(-1).mean(-1)
             x = x.reshape(x.shape[0], -1)
        H = self.fc(x)

        # classification
        y_prob = self.classifier(H)

        y_hat = torch.ge(y_prob, math.log(0.5)).float()

        return y_prob, y_hat

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()

        error = 1 - ((X > math.log(0.5)) == Y).float()
        return error

    def calculate_objective(self, X, Y):
        Y = Y.float()
#         print(Y.shape)
#         print(X.shape, Y.shape)
        y_prob, _ = self.forward(X)
        
        preds = y_prob.flatten().to(y_prob.device)
        preds = torch.sort(preds)[0]
        length = preds.shape[0]
        preds = preds.squeeze()

        k = torch.arange(length, dtype=X.dtype, device=X.device)
        n = torch.tensor(length - 1, dtype=X.dtype, device=X.device)
        log_n_choose_k = (
            torch.lgamma(n + 1) - torch.lgamma((n - k) + 1) - torch.lgamma(k + 1)
        )

        log_estimator = nn.functional.logsigmoid(F.relu(self.estimator))
        log_alpha_k = torch.mul(
            log_estimator,
            torch.arange(length - 1, -1, -1, dtype=X.dtype, device=X.device),
        )
        log_neg_alpha_k = torch.mul(
            log1mexp(log_estimator),
            torch.arange(length, dtype=X.dtype, device=X.device),
        )

        logprob_sorted = preds
        logS_estimator = torch.logsumexp(
            log_n_choose_k + log_alpha_k + log_neg_alpha_k + logprob_sorted,
            dim=0,
            keepdim=True,
        )
#         print(logS_estimator.shape, logS_estimator, Y)
        log_likelihood = self.objective(logS_estimator, Y)
        
        return log_likelihood, logS_estimator[0].detach().cpu().item()
