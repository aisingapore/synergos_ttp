#!/usr/bin/env python

####################
# Required Modules #
####################

# Generic

# Libs
import torch as th
from torch import nn

##################
# Configurations #
##################

def fate_lr_decay(self, initial_lr, lr_decay, epochs):
    """ FATE's learning rate decay equation 
    
    Args:
        initial_lr  (float): Initial learning rate specified
        lr_decay    (float): Scaling factor for Learning rate 
        epochs        (int): No. of epochs that have passed
    Returns:
        Scaled learning rate (float)
    """
    lr = initial_lr / (1 + (lr_decay * epochs))
    return lr

MODULE_OF_OPTIMIZERS = "torch.optim"
MODULE_OF_CRITERIONS = "torch.nn"
    
###########################################
# Parameter Abstraction class - Arguments #
###########################################

class Arguments:
    """ 
    PySyft, at its heart, is an extension of PyTorch, which already supports a
    plathora of functions for various deep-learning operations. Hence, it would
    be unwise to re-implement what already exists. However, across all functions
    there exists too many arguments for different functions. This class provides
    a means to localise all required parameters for functions that might be used
    during the federated training.
    
    # Model Arguments (reference purposes only)
    input_size, output_size, is_condensed

    # Optimizer Arguments (only for selected optimizer(s))
    torch.optim.SGD(params, lr=<required parameter>, momentum=0, 
                     dampening=0, weight_decay=0, nesterov=False)
                     
    # Criterion Arguments
    torch.nn.BCELoss(weight=None, size_average=None, reduce=None, reduction='mean')
    SurrogateCriterion(mu, l1_lambda, l2_lambda)

    # LR Decay Arguments (selected schedulers only)
    torch.optim.lr_scheduler.LambdaLR   (optimizer, lr_lambda, last_epoch=-1)
                            .CyclicLR   (optimizer, base_lr, max_lr, 
                                         step_size_up=2000, step_size_down=None,
                                         mode='triangular', gamma=1.0,
                                         scale_fn=None, scale_mode='cycle', 
                                         cycle_momentum=True, base_momentum=0.8, 
                                         max_momentum=0.9, last_epoch=-1)

    # Early Stopping Arguments
    EarlyStopping (patience, delta)

    # Arguments for functions are retrieved via `func.__code__.co_varnames`    
    """
    def __init__(self, input_size, output_size, batch_size=None, rounds=10, 
                 epochs=100, lr=0.001, lr_decay=0.1, weight_decay=0, seed=42,
                 is_condensed=True, is_snn=False, precision_fractional=5, 
                 use_CLR=True, mu=0.1, reduction='mean', l1_lambda=0, l2_lambda=0, 
                 optimizer=th.optim.SGD, criterion=nn.BCELoss, dampening=0, 
                 lr_lambda=None, base_lr=0.001, max_lr=0.1, step_size_up=2000, 
                 step_size_down=None, mode='triangular', gamma=1.0, scale_fn=None, 
                 scale_mode='cycle', cycle_momentum=True, base_momentum=0.8, 
                 max_momentum=0.9, last_epoch=-1, patience=10, delta=0.0, 
                 cumulative_delta=False):

        self.__FUNCTIONAL_MAPPING = {
            'sgd': th.optim.SGD,
            'bce': nn.BCELoss
        }

        # General Parameters
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size     # Default: None (i.e. bulk analysis)
        self.rounds = rounds
        self.epochs = epochs
        self.seed = seed
        self.is_condensed = is_condensed
        self.is_snn = is_snn
        self.precision_fractional = precision_fractional

        # Optimizer Parameters
        self.lr = lr
        self.weight_decay = 0 if l2_lambda else weight_decay
        self.momentum = base_momentum
        self.dampening = dampening
        self.nesterov = False
        
        # Criterion Parameters
        self.mu = mu
        self.reduction = reduction
        
        # Regularisation Parameters
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda

        # LR Decay Parameters
        self.optimizer = optimizer
        self.criterion = criterion
        self.use_CLR = use_CLR
        self.lr_decay = lr_decay
        self.lr_lambda = lambda epochs: fate_lr_decay(
            self.lr, 
            self.lr_decay, 
            epochs
        ) if not lr_lambda else lr_lambda
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size_up = step_size_up
        self.step_size_down = step_size_down
        self.mode = mode
        self.gamma = gamma
        self.scale_fn = scale_fn
        self.scale_mode = scale_mode
        self.cycle_momentum = cycle_momentum
        self.base_momentum = base_momentum
        self.max_momentum = max_momentum
        self.last_epoch = last_epoch

        # Early Stopping parameters
        self.patience = patience
        self.delta = delta

    ###########
    # Getters #
    ###########

    @property
    def model_params(self):
        return {
            'input_size': self.input_size,
            'output_size': self.output_size,
            'is_condensed': self.is_condensed
        }


    @property
    def optimizer_params(self):
        return {
            'lr': self.lr,
            'weight_decay': self.weight_decay,
            'momentum': self.momentum,
            'dampening': self.dampening,
            'nesterov': False
        }
    
    
    @property
    def criterion_params(self):
        return {
            'mu': self.mu,
            'l1_lambda': self.l1_lambda,
            'l2_lambda': self.l2_lambda,
            'reduction': self.reduction
        }


    @property
    def lr_decay_params(self):
        if self.use_CLR:
            return {
                'base_lr': self.base_lr,
                'max_lr': self.max_lr, 
                'step_size_up': self.step_size_up,
                'step_size_down': self.step_size_down,
                'mode': self.mode,
                'gamma': self.gamma,
                'scale_fn': self.scale_fn,
                'scale_mode': self.scale_mode,
                'cycle_momentum': self.cycle_momentum,
                'base_momentum': self.base_momentum,
                'max_momentum': self.max_momentum, 
                'last_epoch': self.last_epoch
            }
        else:
            return {
                'lr_lambda': self.lr_lambda,
                'last_epoch': self.last_epoch
            }


    @property
    def early_stopping_params(self):
        return {
            'patience': self.patience,
            'delta': self.delta
        }
        
#########
# Tests #
#########

if __name__ == "__main__":
    args = Arguments(input_size=20, output_size=1)
    print(args.optimizer_params)
    print(args.lr_decay_params)
    print(args.early_stopping_params)
