import os
import os.path
import time
import torch
import logging

from utils.factory import get_model
from utils.toolkit import print_args, format_elapsed_time
from utils.toolkit import setup_logging, set_device, set_random
from dataloaders.data_manager import DataManager


def start(args):
    data_manager, model = initialize(args)
    train(data_manager, model, args)


def initialize(args):
    # log
    args['logfilename'] = os.path.join(args['logdir'], 'seed{}'.format(args['seed']))
    setup_logging(args['logfilename'], args['save_ckp'])

    # random seed and device
    set_random(args)
    set_device(args)
    print_args(args)

    # datamanager
    data_manager = DataManager(args['dataset'], args['shuffle'], args['seed'], args['init_cls'], 
                               args['increment'], args)
    # model
    model = get_model(args['method'], args)

    return data_manager, model


def train(data_manager, model, args):

    curve_accy, curve_accy_with_task, curve_accy_task = {'top1': []}, {'top1': []}, {'top1': []}

    # Train and Eval sequentially for N tasks
    for task in range(data_manager.task_num):
        logging.info('='*80)
        model.before_task(data_manager)

        # learning on the new task (train)
        time_start = time.time()
        model.incremental_train(data_manager)
        time_end = time.time()
        logging.info('Training time: {}'.format(format_elapsed_time(time_start, time_end)))

        # evaluate the model (eval)
        time_start = time.time()
        accy, accy_with_task, accy_task = model.incremental_test(data_manager)
        time_end = time.time()
        logging.info('Evaluation time: {}'.format(format_elapsed_time(time_start, time_end)))

        model.after_task()

        # logging
        logging.info('Accuracy: {}'.format(accy['grouped']))
        curve_accy['top1'].append(accy['top1'])
        curve_accy_with_task['top1'].append(accy_with_task['top1'])
        curve_accy_task['top1'].append(accy_task)
        logging.info('(curve) top1 Acc: {}'.format(curve_accy['top1']))  # Average Accuracy (A_t)
        logging.info('(curve) top1 Acc with task: {}'.format(curve_accy_with_task['top1']))  # Average Accuracy with task id
        logging.info('(curve) top1 Acc task: {}'.format(curve_accy_task['top1']))
        logging.info('='*80)

        # save model
        torch.save(model.network.state_dict(), os.path.join(args['logfilename'], "task_{}.pth".format(int(task)))) if args['save_ckp'] else None
