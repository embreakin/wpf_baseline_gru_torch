# -*-Encoding: utf-8 -*-
################################################################################
#
# Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Description: Training and Validation
Authors: Lu,Xinjiang (luxinjiang@baidu.com)
Date:    2022/03/10
"""
import os
import time
import numpy as np
from typing import Callable
import torch
#from paddle.io import DataLoader
#import torch.utils.data as data
from common import EarlyStopping
from common import adjust_learning_rate
from common import Experiment
from common import traverse_wind_farm
from prepare import prep_env
from torch.optim.lr_scheduler import ReduceLROnPlateau


def val(experiment, data_loader, criterion):
    # type: (Experiment, DataLoader, Callable) -> np.array
    """
    Desc:
        Validation function
    Args:
        experiment:
        data_loader:
        criterion:
    Returns:
        The validation loss
    """
    validation_loss = []
    for i, (batch_x, batch_y) in enumerate(data_loader):
        sample, true = experiment.process_one_batch(batch_x, batch_y)
        loss = criterion(sample, true)
        validation_loss.append(loss.item())
    validation_loss = np.average(validation_loss)
    return validation_loss


def train_and_val(experiment, model_folder, is_debug=True):
    # type: (Experiment, str, bool) -> None
    """
    Desc:
        Training and validation
    Args:
        experiment:
        model_folder: folder name of the model
        is_debug:
    Returns:
        None
    """
    args = experiment.get_args()
    model = experiment.get_model()
    train_data, train_loader = experiment.get_data(flag='train')
    # print(f"train_data.shape = {train_data.size}")
    val_data, val_loader = experiment.get_data(flag='val')
    # print(f"val_data.shape = {val_data.shape}")
    
    path_to_model = os.path.join(args["checkpoints"], model_folder)
    if not os.path.exists(path_to_model):
        os.makedirs(path_to_model)

    time_now = time.time()
    early_stopping = EarlyStopping(patience=args["patience"], verbose=True)
    model_optim = experiment.get_optimizer()
    criterion = Experiment.get_criterion()
    max_grad_norm=50

    #Define the learning rate scheduler
    if args["lr_adjust"] == "rlrop":
        scheduler = ReduceLROnPlateau(model_optim, mode='min', patience=5, factor=0.5, verbose=True)

    epoch_start_time = time.time()
    for epoch in range(args["train_epochs"]):
        iter_count = 0
        train_loss = []
        model.train()
        for i, (batch_x, batch_y) in enumerate(train_loader):
            iter_count += 1
            # print(f"iter_count = {iter_count}")
            # print(f"batch_x.shape = {batch_x.size}")
            # print(f"batch_y.shape = {batch_y.size}")

            sample, truth = experiment.process_one_batch(batch_x, batch_y)
            loss = criterion(sample, truth)
            train_loss.append(loss.item())
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            # model_optim.minimize(loss)
            model_optim.step()
        print(f"iter_count = {iter_count}")
        val_loss = val(experiment, val_loader, criterion)

        if args["lr_adjust"] == "rlrop":
            scheduler.step(val_loss)

        if is_debug:
            train_loss = np.average(train_loss)
            epoch_end_time = time.time()
            print("Epoch: {}, \nTrain Loss: {}, \nValidation Loss: {}".format(epoch, train_loss, val_loss))
            print("Elapsed time for epoch-{}: {}".format(epoch, epoch_end_time - epoch_start_time))
            epoch_start_time = epoch_end_time

        # Early Stopping if needed
        early_stopping(val_loss, model, path_to_model, args["turbine_id"])
        if early_stopping.early_stop:
            print("Early stopped! ")
            break
        # adjust_learning_rate(model_optim, epoch + 1, args)


if __name__ == "__main__":
    settings = prep_env()
    #
    # Set up the initial environment
    # Current settings for the model
    cur_setup = '{}_t{}_i{}_o{}_ls{}_train{}_val{}'.format(
        settings["filename"], settings["task"], settings["input_len"], settings["output_len"], settings["lstm_layer"],
        settings["train_size"], settings["val_size"]
    )
    traverse_wind_farm(train_and_val, settings, cur_setup)
