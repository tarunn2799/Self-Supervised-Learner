import random

import torch
from sklearn import preprocessing
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import RandomResizedCrop

from models.RandAugment import RandAugment


class SIMCLRData( Dataset ):
    def __init__(self, DATA_PATH, input_height, copies, stage):
        super().__init__()
        self.image_ids = [f'im{i}' for i in range( copies )]
        if stage != 'inference':
            self.image_ids.append( 'label' )

        self.m = random.randint( 1, 2 )
        self.n = random.randint( 1, 4 )
        self.num_samples = len( ImageFolder( DATA_PATH ) )

        self.copies = copies
        self.input_height = input_height
        self.stage = stage
        self.input = ImageFolder( DATA_PATH )
        self.label_transform = dict()

        le = preprocessing.LabelEncoder().fit( self.input.targets )

        for key in self.input.targets:  # For mapping labels to integer values
            self.label_transform[key] = le.transform( key )

        self.randaug = RandAugment( self.n, self.m )
        self.crop = RandomResizedCrop( input_height )

    def __len__(self):
        return self.num_samples

    def val_transform(self, image):  ## must rewrite these below functions into pytorch transforms functions
        image = self.crop()
        # self.swapaxes = ops.Transpose( perm=[2, 0, 1], device="gpu" )
        #     No need to transpose cos pytorch is automatically like that
        # image = self.swapaxes( image )
        return image

    def __getitem__(self, index):
        returnable = dict()
        sample, label = self.input[index]
        if self.stage == 'train':
            self.transform = self.randaug
        else:
            self.transform = self.val_transform

        for i, key in enumerate( self.image_ids ):
            if key == 'label':
                returnable['label'] = torch.tensor( self.label_transform[label] )
            else:
                returnable[key] = torch.tensor( self.transform( sample ) )

        return returnable
