"""Performs experiments for a given number of epoches, after each epoch the validation error and loss is computed to
 monitor the training."""

from __future__ import print_function

import time

import numpy as np

import torch
import torch.utils.data as data_utils
from torch.autograd import Variable

from utils2 import evaluate
from utils2 import train


def experiment(args, kwargs, current_fold, train_set, val_set, test_set, model, optimizer, scheduler, dir, sw):
    best_error = 1.
    best_loss = 1000.
    best_error_train = 1.
    e = 1
    train_loss_history = []
    train_error_history = []
    val_loss_history = []
    val_error_history = []
    time_history = []

    path_name_current_fold = dir + args.model_name + '_fold' + str(current_fold)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=True, **kwargs)

    for epoch in range(1, args.epochs + 1):
        time_start = time.time()
        scheduler.step()

        model, train_loss, train_error = train(args, train_loader, model, optimizer)
        val_loss, val_error, val_metrics = evaluate(args, model, train_loader, val_loader, mode='validation')

        time_end = time.time()
        time_elapsed = time_end - time_start

        sw.add_scalar('train_loss', train_loss, epoch)
        sw.add_scalar('val_loss', val_loss, epoch)
        sw.add_scalar('train_error', train_error, epoch)
        sw.add_scalar('val_error', val_error, epoch)
        sw.add_scalar('val_ACC', val_metrics['ACC'], epoch)
        sw.add_scalar('val_balACC', val_metrics['balACC'], epoch)
        sw.add_scalar('val_AP', val_metrics['AP'], epoch)
        sw.add_scalar('val_F1', val_metrics['F1'], epoch)

        # appending history
        train_loss_history.append(train_loss)
        train_error_history.append(train_error)
        val_loss_history.append(val_loss)
        val_error_history.append(val_error)
        time_history.append(time_elapsed)

        # printing results
        print(f'\tResults Epoch: {epoch}/{args.epochs} in Test-Train fold: {current_fold}/{args.kfold_test}, Time elapsed: {time_elapsed}s\n'
              f'\t* Train loss: {train_loss}   , error: {train_error}\n'
              f'\to Val.  loss: {val_loss}   , error: {val_error}\n'
              f'\t--> Early stopping: {e}/{args.early_stopping_epochs} (BEST: {best_error})\n\n'.format( 
        ))

        # early-stopping
        if val_error < best_error:
            e = 0
            best_error = val_error
            best_loss = val_loss
            torch.save(model, path_name_current_fold + '.models')
            print('>>--models saved--<<')
            print(path_name_current_fold + '.models')
        elif val_error == best_error:
            if val_loss < best_loss and train_error < best_error_train:
                e = 0
                best_error = val_error
                best_loss = val_loss
                best_error_train = train_error
                torch.save(model, path_name_current_fold + '.models')
                print('>>--models saved--<<')
                print(path_name_current_fold + '.models')
            else:
                e += 1
                if e > args.early_stopping_epochs:
                    break
        else:
            e += 1
            if e > args.early_stopping_epochs:
                break

    # SAVING
    torch.save(args, path_name_current_fold + '.config')

    # FINAL EVALUATION
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, **kwargs)

    evaluate_model = torch.load(path_name_current_fold + '.models')
    print('>>--models loaded--<<')
    print(path_name_current_fold + '.models')
    objective_test, error_test, objective_train, error_train, test_metrics, train_metrics = evaluate(args,
                                                                                                    evaluate_model,
                                                                                                    train_loader,
                                                                                                    test_loader,
                                                                                                    mode='test')

    sw.add_scalar('error_test', error_test, 0)
    
    sw.add_scalar('test_ACC', test_metrics['ACC'], epoch)
    sw.add_scalar('test_balACC', test_metrics['balACC'], epoch)
    sw.add_scalar('test_AP', test_metrics['AP'], epoch)
    sw.add_scalar('test_F1', test_metrics['F1'], epoch)
    
    sw.add_scalar('train_ACC', train_metrics['ACC'], epoch)
    sw.add_scalar('train_balACC', train_metrics['balACC'], epoch)
    sw.add_scalar('train_AP', train_metrics['AP'], epoch)
    sw.add_scalar('train_F1', train_metrics['F1'], epoch)

    print(f'\tFINAL EVALUATION ON TEST SET OF TEST-TRAIN FOLD: {current_fold}/{args.kfold_test}\n'
          f'\tLogL (TEST): {objective_test}\n'
          f'\tLogL (TRAIN): {objective_train}\n'
          f'\tERROR (TEST): {error_test}\n'
          f'\tERROR (TRAIN): {error_train}\n'
    )

    with open('experiment_log_' + args.operator + '.txt', 'a') as f:
        print(f'\tFINAL EVALUATION ON TEST SET OF TEST-TRAIN FOLD: {current_fold}/{args.kfold_test}\n'
          f'\tLogL (TEST): {objective_test}\n'
          f'\tLogL (TRAIN): {objective_train}\n'
          f'\tERROR (TEST): {error_test}\n'
          f'\tERROR (TRAIN): {error_train}\n', file=f)

    # SAVING
    torch.save(train_loss_history, path_name_current_fold + '.train_loss')
    torch.save(train_error_history, path_name_current_fold + '.train_error')
    torch.save(val_loss_history, path_name_current_fold + '.val_loss')
    torch.save(val_error_history, path_name_current_fold + '.val_error')
    torch.save(objective_test, path_name_current_fold + '.objective_test')
    torch.save(objective_train, path_name_current_fold + '.objective_train')
    torch.save(error_test, path_name_current_fold + '.error_test')
    torch.save(error_train, path_name_current_fold + '.error_train')

    return error_train, error_test


def train(args, train_loader, model, optimizer):
    # set loss to 0
    train_loss = 0.
    train_error = 0.

    # set models in training mode
    model.train(True)

    # start training
    for batch_idx, (data, label) in enumerate(train_loader):
        label = label[0]
#        perm = torch.randperm(data.size(1))
#        idx = perm[:200]
        samples = data  #[:, idx]
        if args.cuda:
            data, label = data.cuda(), label.cuda()
        data, label = Variable(data), Variable(label)

        # reset gradients
        optimizer.zero_grad()
        # calculate loss and metrics
        loss, pred = model.calculate_objective(data, label)
        train_loss += loss
        train_error += model.calculate_classification_error(pred, label)
        # backward pass
        loss.backward()
        # optimization
        optimizer.step()

    # calculate final loss
    train_loss /= len(train_loader)
    train_error /= len(train_loader)

    return model, train_loss, train_error
