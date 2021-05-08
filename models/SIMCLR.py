from argparse import ArgumentParser
from enum import Enum

import numpy as np
from pl_bolts.models.self_supervised import SimCLR
from pl_bolts.models.self_supervised.simclr.simclr_module import Projection
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import ImageFolder

# Internal Imports
from dali_utils.dali_transforms import SimCLRTransform
from models.CustomDataSetter import SIMCLRData


class SIMCLR( SimCLR ):

    def __init__(self, encoder, DATA_PATH, VAL_PATH, hidden_dim, image_size, seed, cpus, transform=SimCLRTransform,
                 **simclr_hparams):

        data_temp = ImageFolder( DATA_PATH )

        # derived values (not passed in) need to be added to model hparams
        simclr_hparams['num_samples'] = len( data_temp )
        simclr_hparams['dataset'] = None
        simclr_hparams['max_epochs'] = simclr_hparams['epochs']

        self.DATA_PATH = DATA_PATH
        self.VAL_PATH = VAL_PATH
        self.hidden_dim = hidden_dim
        self.transform = transform
        self.image_size = image_size
        self.cpus = cpus
        self.seed = seed

        super().__init__( **simclr_hparams )
        self.encoder = encoder

        self.projection = Projection( input_dim=self.encoder.embedding_size, hidden_dim=self.hidden_dim )

        self.save_hyperparameters()

    # override pytorch SIMCLR with our own encoder so we will overwrite the function plbolts calls to init the encoder
    def init_model(self):
        return None

    # def setup(self, stage='inference'):
    #     Options = Enum( 'Loader', 'fit test inference' )
    #     if stage == Options.fit.name:
    #         train = self.transform( self.DATA_PATH, batch_size=self.batch_size, input_height=self.image_size, copies=3,
    #                                 stage='train', num_threads=self.cpus, device_id=self.local_rank, seed=self.seed )
    #         val = self.transform( self.VAL_PATH, batch_size=self.batch_size, input_height=self.image_size, copies=3,
    #                               stage='validation', num_threads=self.cpus, device_id=self.local_rank, seed=self.seed )
    #         self.train_loader = SimCLRWrapper( transform=train )
    #         self.val_loader = SimCLRWrapper( transform=val )
    #         breakpoint()
    #     elif stage == Options.inference.name:
    #         self.test_dataloader = SimCLRWrapper(
    #             transform=self.transform( self.DATA_PATH, batch_size=self.batch_size, input_height=self.image_size,
    #                                       copies=1, stage='inference', num_threads=2 * self.cpus,
    #                                       device_id=self.local_rank, seed=self.seed ) )
    #         self.inference_dataloader = self.test_dataloader
    #         breakpoint()

    def setup(self, stage='inference'):
        Options = Enum( 'Loader', 'fit test inference' )
        if stage == Options.fit.name:
            train_data = SIMCLRData( input_height=self.image_size, DATA_PATH=self.DATA_PATH, copies=3, stage='train' )
            val_data = SIMCLRData( input_height=self.image_size, DATA_PATH=self.DATA_PATH, copies=3,
                                   stage='validation' )

            valid_size = 0.2

            # Dividing the indices for train and cross validation
            indices = list( range( len( train_data ) ) )
            np.random.shuffle( indices )
            split = int( np.floor( valid_size * len( train_data ) ) )

            train_idx, valid_idx = indices[split:], indices[:split]

            self.train_loader = DataLoader( train_data, batch_size=self.batch_size,
                                            sampler=SubsetRandomSampler( train_idx ) )
            self.val_loader = DataLoader( val_data, batch_size=self.batch_size,
                                          sampler=SubsetRandomSampler( valid_idx ) )
            breakpoint()
        elif stage == Options.inference.name:
            data = SIMCLRData( input_height=self.image_size, DATA_PATH=self.DATA_PATH, copies=3, stage='inference' )
            self.test_dataloader = DataLoader( data, batch_size=self.batch_size, shuffle=False )
            self.inference_dataloader = self.test_dataloader
            breakpoint()

    def train_dataloader(self):
        return iter( self.train_loader )

    def val_dataloader(self):
        return iter( self.val_loader )

    # give user permission to add extra arguments for SIMCLR model particularly
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser( parents=[parent_parser], add_help=False )

        # things we need to pass into pytorch lightning simclr model

        parser.add_argument( "--num_workers", default=8, type=int, help="num of workers per GPU" )
        parser.add_argument( "--optimizer", default="adam", type=str, help="choose between adam/sgd" )
        parser.add_argument( "--lars_wrapper", action='store_true', help="apple lars wrapper over optimizer used" )
        parser.add_argument( '--exclude_bn_bias', action='store_true', help="exclude bn/bias from weight decay" )
        parser.add_argument( "--warmup_epochs", default=1, type=int, help="number of warmup epochs" )

        parser.add_argument( "--temperature", default=0.1, type=float, help="temperature parameter in training loss" )
        parser.add_argument( "--weight_decay", default=1e-6, type=float, help="weight decay" )

        parser.add_argument( "--start_lr", default=0, type=float, help="initial warmup learning rate" )
        parser.add_argument( "--final_lr", type=float, default=1e-6, help="final learning rate" )

        return parser
