import random

import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import RandomResizedCrop

from models.RandAugment import RandAugment


class SIMCLRData( Dataset ):
    def __init__(self, DATA_PATH, input_height, copies, stage):
        super().__init__()
        self.image_ids = [f'im{i}' for i in range( copies )]

        self.m = random.randint( 1, 2 )
        self.n = random.randint( 0, 30 )
        self.num_samples = len( ImageFolder( DATA_PATH ) )

        self.copies = copies
        self.input_height = input_height
        self.stage = stage
        self.input = ImageFolder( DATA_PATH )

        self.randaug = RandAugment( self.n, self.m, input_height )
        self.crop = RandomResizedCrop( input_height )

    def __len__(self):
        return self.num_samples

    def val_transform(self, image):
        image = self.crop( image )
        return image

    def __getitem__(self, index):
        sample, label = self.input[index]
        if self.stage == 'train':
            self.transform = self.randaug
        else:
            self.transform = self.val_transform

        if self.stage != 'inference':
            return tuple( [transforms.ToTensor()( self.transform( sample ) ) for _ in self.image_ids] ), label
        else:
            return tuple( [transforms.ToTensor()( self.transform( sample ) ) for _ in self.image_ids] )
