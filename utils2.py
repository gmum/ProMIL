import numpy as np
from torch.autograd import Variable
from torch.utils.data import Subset

from ColonLoader import ColonCancerBagsCross
from utils3 import Scores
import time
import torch
import math

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
        loss, pred = model.calculate_objective(data, label)
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
    sc = Scores(output_transform=math.exp)
    model.eval()

    if mode == 'validation':
        # set loss to 0
        evaluate_loss = 0.
        evaluate_error = 0.
        # CALCULATE classification error and log-likelihood for VALIDATION SET
        for batch_idx, (data, label) in enumerate(data_loader):
            label = label[0]
            if len(label.shape) != 0:
                if len(label.shape) > 1 or label.shape[0] > 1:
                    label = label.squeeze()
                    label = label[0]
            if args.cuda:
                data, label = data.cuda(), label.cuda()
            data, label = Variable(data), Variable(label)
            # reset gradients# calculate loss and metrics
            evaluate_loss_p, pred = model.calculate_objective(data, label)
            evaluate_loss += evaluate_loss_p.item()
            evaluate_error += model.calculate_classification_error(pred, label)
            sc.update(label, pred)
        metrics_val = sc.compute()
        sc.reset()

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
            if len(label.shape) != 0:
                if len(label.shape) > 1 or label.shape[0] > 1:
                    label = label.squeeze()
                    label = label[0]
            if args.cuda:
                data, label = data.cuda(), label.cuda()
            data, label = Variable(data), Variable(label)
            evaluate_loss_p, pred = model.calculate_objective(data, label)
            evaluate_loss += evaluate_loss_p.item()
            evaluate_error += model.calculate_classification_error(pred, label)
            sc.update(label, pred)
        t_ll_e = time.time()
        evaluate_error /= len(data_loader)
        evaluate_loss /= len(data_loader)
        metrics_val = sc.compute()
        sc.reset()
        print(f'\tTEST classification error value (time): {evaluate_error} ({t_ll_e - t_ll_s}s)')
        print(f'\tTEST log-likelihood value (time): {evaluate_loss} ({t_ll_e - t_ll_s}s)\n')

        # CALCULATE classification error and log-likelihood for TRAINING SET
        t_ll_s = time.time()
        for batch_idx, (data, label) in enumerate(train_loader):
            label = label[0]
            if len(label.shape) != 0:
                if len(label.shape) > 1 or label.shape[0] > 1:
                    label = label.squeeze()
                    label = label[0]
            if args.cuda:
                data, label = data.cuda(), label.cuda()
            data, label = Variable(data), Variable(label)
            train_loss_p, pred = model.calculate_objective(data, label)
            train_loss += train_loss_p.item()
            train_error += model.calculate_classification_error(pred, label)
            sc.update(label[0], pred)
        t_ll_e = time.time()
        train_error /= len(train_loader)
        train_loss /= len(train_loader)
        metrics_train = sc.compute()
        sc.reset()
        print('\tTRAIN classification error value (time): {train_error} ({t_ll_e - t_ll_s}s)')
        print('\tTRAIN log-likelihood value (time): {train_loss} ({t_ll_e - t_ll_s}s)\n')

    if mode == 'test':
        return evaluate_loss, evaluate_error, train_loss, train_error, metrics_val, metrics_train
    else:
        return evaluate_loss, evaluate_error, metrics_val


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


def get_nsclc_dataset(fold, seed):
    assert fold < 10
    from tcga_nsclc.dataset import NSCLCPreprocessedBagsCross
    ds = NSCLCPreprocessedBagsCross(path="/shared/sets/datasets/vision/TCGA-NSCLC/patches/", train=True, shuffle_bag=True,
                                           data_augmentation=True,)
    
    perm = torch.randperm(len(ds))
    p = int(len(ds) // 10)
    valid_idx = perm[(9 - fold)*p:(9 - fold + 1)* p]
    ds_valid = Subset(ds, valid_idx)
    test_idx = perm[fold*p:(fold + 1)* p]
    ds_test = Subset(ds, test_idx)
    train_idx = [pe for pe in perm if pe not in valid_idx]
    train_idx = [pe for pe in train_idx if pe not in test_idx]
    ds_train = Subset(ds, train_idx)
    return ds_train, ds_valid, ds_test
    
    
