import random

import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from models.RandAugment import RandAugment


class SIMCLRData( Dataset ):
    def __init__(self, DATA_PATH, input_height, copies, stage):
        super().__init__()

        self.m = random.randint( 1, 2 )
        self.n = random.randint( 1, 4 )
        self.num_samples = len( ImageFolder( DATA_PATH ) )

        self.copies = copies
        self.input_height = input_height
        self.stage = stage
        self.input = ImageFolder( DATA_PATH )
        self.randaug = RandAugment( self.n, self.m )

    def __len__(self):
        return self.num_samples

    def val_transform(self, image):  ## must rewrite these below functions into pytorch transforms functions
        image = self.crop(
            image )  # ops.RandomResizedCrop( size=self.input_height, minibatch_size=batch_size, device="gpu",
        # dtype=types.FLOAT )
        # self.swapaxes = ops.Transpose( perm=[2, 0, 1], device="gpu" )

        image = self.swapaxes( image )
        return image

    def __getitem__(self, index):
        returnable = dict()
        sample, label = self.input[index]
        if self.stage == 'train':
            self.transform = self.randaug
        else:
            self.transform = self.val_transform

        if self.stage == 'train':
            returnable['label'] = torch.tensor( label )

        returnable['image'] = self.transform( sample )

        return returnable
#  this is what is normally done in a pytorch dataloader, the below code is from the dali code, idk what to do about it
#         batch = ()
#         for i in range(self.copies):
#             batch += (self.transform(jpegs), )
#             breakpoint()
#         if self.stage is not 'inference':
#             label = label.gpu()
#             label = self.to_int64(label)
#             batch += (label, )
#         return batch
