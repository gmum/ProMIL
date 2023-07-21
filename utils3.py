import numpy as np
from torch.autograd import Variable

from ColonLoader import ColonCancerBagsCross
import time
from collections.abc import Mapping
from math import log
from typing import Callable

import numpy as np
import torch
from matplotlib.figure import Figure
from sklearn.metrics import (
    balanced_accuracy_score,
    average_precision_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

def train(args, train_loader, model, optimizer):
    # set loss to 0
    train_loss = 0.
    train_error = 0.

    # set models in training mode
    model.train(True)

    # start training
    for batch_idx, (data, label) in enumerate(train_loader):
        label = label[0]
        if args.cuda:
            data, label = data.cuda(), label.cuda()
        data, label = Variable(data), Variable(label)

        # reset gradients
        optimizer.zero_grad()
        # calculate loss and metrics
        loss, gamma, gamma_kernel = model.calculate_objective(data, label)
        train_loss += loss.data[0]
        train_error += model.calculate_classification_error(data, label)[0]
        # backward pass
        loss.backward()
        # optimization
        optimizer.step()

    # calculate final loss
    train_loss /= len(train_loader)
    train_error /= len(train_loader)

    return model, train_loss, train_error, gamma, gamma_kernel


def evaluate(args, model, train_loader, data_loader, mode):
    # set model to evaluation mode
    model.eval()

    if mode == 'validation':
        # set loss to 0
        evaluate_loss = 0.
        evaluate_error = 0.
        # CALCULATE classification error and log-likelihood for VALIDATION SET
        for batch_idx, (data, label) in enumerate(data_loader):
            label = label[0]
            if args.cuda:
                data, label = data.cuda(), label.cuda()
            data, label = Variable(data), Variable(label)
            # reset gradients# calculate loss and metrics
            evaluate_loss_p, pred, _ = model.calculate_objective(data, label)
            evaluate_loss += evaluate_loss_p.item()
            evaluate_error += model.calculate_classification_error(pred, label)

        # calculate final loss
        evaluate_loss /= len(data_loader)
        evaluate_error /= len(data_loader)

    if mode == 'test':
        # set loss to 0
        train_error = 0.
        train_loss = 0.
        evaluate_error = 0.
        evaluate_loss = 0.
        # CALCULATE classification error and log-likelihood for TEST SET
        t_ll_s = time.time()
        for batch_idx, (data, label) in enumerate(data_loader):
            label = label[0]
            if args.cuda:
                data, label = data.cuda(), label.cuda()
            data, label = Variable(data), Variable(label)
            print(data.shape)
            evaluate_loss_p, pred, _ = model.calculate_objective(data, label)
            evaluate_loss += evaluate_loss_p.item()
            evaluate_error += model.calculate_classification_error(pred, label)
        t_ll_e = time.time()
        evaluate_error /= len(data_loader)
        evaluate_loss /= len(data_loader)
        print(f'\tTEST classification error value (time): {evaluate_error} ({t_ll_e - t_ll_s}s)')
        print(f'\tTEST log-likelihood value (time): {evaluate_loss} ({t_ll_e - t_ll_s}s)\n')

        # CALCULATE classification error and log-likelihood for TRAINING SET
        t_ll_s = time.time()
        for batch_idx, (data, label) in enumerate(train_loader):
            label = label[0]
            if args.cuda:
                data, label = data.cuda(), label.cuda()
            data, label = Variable(data), Variable(label)
            train_loss_p, pred, _ = model.calculate_objective(data, label)
            train_loss += train_loss_p.item()
            train_error += model.calculate_classification_error(pred, label)
        t_ll_e = time.time()
        train_error /= len(train_loader)
        train_loss /= len(train_loader)
        print('\tTRAIN classification error value (time): {train_error} ({t_ll_e - t_ll_s}s)')
        print('\tTRAIN log-likelihood value (time): {train_loss} ({t_ll_e - t_ll_s}s)\n')

    if mode == 'test':
        return evaluate_loss, evaluate_error, train_loss, train_error
    else:
        return evaluate_loss, evaluate_error


def kfold_indices_warwick(N, k, seed=777):
    r = np.random.RandomState(seed)
    all_indices = np.arange(N, dtype=int)
    r.shuffle(all_indices)
    idx = [int(i) for i in np.floor(np.linspace(0, N, k + 1))]
    train_folds = []
    valid_folds = []
    for fold in range(k):
        valid_indices = all_indices[idx[fold]:idx[fold + 1]]
        valid_folds.append(valid_indices)
        train_fold = np.setdiff1d(all_indices, valid_indices)
        r.shuffle(train_fold)
        train_folds.append(train_fold)
    return train_folds, valid_folds


def load_warwick(train_fold, val_fold, test_fold, loc_info):
    print('\t-> Loading the following dataset')
    train_set, val_set, test_set = load_warwick_cross(train_fold, val_fold, test_fold, loc_info)
    return train_set, val_set, test_set


def load_warwick_cross(train_fold, val_fold, test_fold, loc_info):

    train_set = ColonCancerBagsCross('/shared/sets/datasets/vision/ColonCancer//',
                                     train_val_idxs=train_fold,
                                     test_idxs=test_fold,
                                     train=True,
                                     shuffle_bag=True,
                                     data_augmentation=True,
                                     loc_info=loc_info)

    val_set = ColonCancerBagsCross('/shared/sets/datasets/vision/ColonCancer//',
                                   train_val_idxs=val_fold,
                                   test_idxs=test_fold,
                                   train=True,
                                   shuffle_bag=True,
                                   data_augmentation=True,
                                   loc_info=loc_info)

    test_set = ColonCancerBagsCross('/shared/sets/datasets/vision/ColonCancer//',
                                    train_val_idxs=train_fold,
                                    test_idxs=test_fold,
                                    train=False,
                                    shuffle_bag=False,
                                    data_augmentation=False,
                                    loc_info=loc_info)

    return train_set, val_set, test_set

neg_log2 = -log(2)

def log1mexp(negative_data):  # log(1 - exp(-x)), x > 0
    negative_data = negative_data + torch.tensor([1e-4], device=negative_data.device)
    assert torch.lt(negative_data, 0).all(), negative_data
    return torch.where(
        torch.lt(negative_data, neg_log2),
        torch.log1p(torch.exp(negative_data).neg()),
        torch.log(torch.expm1(negative_data).neg()),
    )


def mixup_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    batch_size = x.shape[0]
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def nested_dict(opts):
    res = {}
    for keys, val in opts.items():
        tmp = res
        levels = keys.split(".")
        for level in levels[:-1]:
            tmp = tmp.setdefault(level, {})
        tmp[levels[-1]] = val
    return res


def update(orig, new):
    for key, val in new.items():
        if isinstance(val, Mapping):
            orig[key] = update(orig.get(key, type(val)()), val)
        else:
            orig[key] = val
    return orig


class Scores:
    def __init__(
        self,
        output_transform: Callable = lambda x: x,
    ):
        self.names = [
            "balACC",
            "ACC",
            "AP",
            "F1",
        ]
        self._num_examples = 0
        self.array_predict = None
        self.array_target = None
        self._output_transform = output_transform
        self.reset()

    def reset(self) -> None:
        self._num_examples = 0
        self.array_predict = None
        self.array_target = None

    @torch.no_grad()
    def update(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> None:
        y_pred = self._output_transform(y_pred)
        y_true = y_true.detach().unsqueeze(0)
        y_pred = torch.tensor([y_pred])

        if len(y_true.shape) != 1:
            raise SystemError(
                f"Variable 'y_true' has a shape greater than d1 [{y_true.shape}]"
            )
        if len(y_pred.shape) != 1:
            raise SystemError(
                f"Variable 'y_pred' has a shape greater than d1 [{y_pred.shape}]"
            )

        self._num_examples += y_pred.shape[0]
        if self.array_target is None:
            self.array_target = y_true
            self.array_predict = y_pred
        else:
            self.array_predict = torch.cat([self.array_predict, y_pred])
            self.array_target = torch.cat([self.array_target, y_true])

    def compute(self, threshold: float = 0.5, conf_mat: bool = False):
        if self._num_examples == 0:
            raise SystemError(
                "Confusion matrix must have at least one example before it can be computed."
            )

        y_true = self.array_target.cpu().numpy()
        y_score = self.array_predict.cpu().numpy()
        y_pred = (y_score > threshold).astype(float)
        res = {
            "balACC": balanced_accuracy_score(y_true, y_pred),
            "ACC": accuracy_score(y_true, y_pred),
            "AP": average_precision_score(y_true, y_score),
            "F1": f1_score(y_true, y_pred),
        }

        if conf_mat:
            cm = confusion_matrix(y_true, y_pred)

            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=None)
            disp.plot()
            return res, disp.figure_
        return res
