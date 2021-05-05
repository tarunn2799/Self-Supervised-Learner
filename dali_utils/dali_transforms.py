import random

import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
from torchvision.datasets import ImageFolder

from models.RandAugment import RandAugment

rander = RandAugment( 5, 5 )


class SimCLRTransform( Pipeline ):
    def __init__(self, DATA_PATH, input_height, batch_size, copies, stage, num_threads, device_id, seed=1729):
        super( SimCLRTransform, self ).__init__( batch_size, num_threads, device_id, seed=seed )

        # this lets our pytorch compat function find the length of our dataset
        self.num_samples = len( ImageFolder( DATA_PATH ) )

        self.m = random.randint( 1, 2 )
        self.n = random.randint( 1, 4 )
        self.copies = copies
        self.input_height = input_height
        self.stage = stage

        self.input = ops.FileReader( file_root=DATA_PATH, random_shuffle=True, seed=seed )
        self.to_int64 = ops.Cast( dtype=types.INT64, device="gpu" )
        self.to_int32_cpu = ops.Cast( dtype=types.INT32, device="cpu" )

        self.coin = ops.random.CoinFlip( probability=0.5 )
        self.uniform = ops.random.Uniform( range=[0.5, 1.5] )
        self.blur_amt = ops.random.Uniform( values=[float( i ) for i in range( 1, int( 0.1 * self.input_height ), 2 )] )

        self.decode = ops.ImageDecoder( device='mixed', output_type=types.RGB )
        self.crop = ops.RandomResizedCrop( size=self.input_height, minibatch_size=batch_size, device="gpu",
                                           dtype=types.FLOAT )
        self.flip = ops.Flip( vertical=self.coin(), horizontal=self.coin(), device="gpu" )
        self.colorjit_gray = ops.ColorTwist( brightness=self.uniform(), contrast=self.uniform(), hue=self.uniform(),
                                             saturation=self.uniform(), device="gpu" )
        self.blur = ops.GaussianBlur( window_size=self.to_int32_cpu( self.blur_amt() ), device="gpu",
                                      dtype=types.FLOAT )

        self.swapaxes = ops.Transpose( perm=[2, 0, 1], device="gpu" )
        self.brightness_contrast=ops.brightness_contrast(brightness=self.uniform(),
                                                         contrast=self.uniform(),contrast_center=self.uniform())
        self.hue=ops.hue(hue=self.uniform())
        self.hsv=ops.hsv(hue=self.uniform(),saturation=self.uniform(),value=self.uniform())
        self.water=ops.water(ampl_x=self.uniform,ampl_y=self.uniform,fill_value=self.uniform,
                             freq_X=self.uniform(),freq_y=self.uniform())
        self.transforms_shear=ops.transforms.shear(angles=[self.uniform(),self.uniform()])
        self.augment_list = [
            (ops.RandomResizedCrop( size=self.input_height, minibatch_size=batch_size, device="gpu",
                                    dtype=types.FLOAT ),),
            (ops.Flip( vertical=self.coin(), horizontal=self.coin(), device="gpu" ),),
            (ops.ColorTwist( brightness=self.uniform(), contrast=self.uniform(), hue=self.uniform(),
                             saturation=self.uniform(), device="gpu" ),),
            (ops.GaussianBlur( window_size=self.to_int32_cpu( self.blur_amt() ), device="gpu", dtype=types.FLOAT )),
            (ops.Transpose( perm=[2, 0, 1], device="gpu" )),
            (ops.brightness_contrast(brightness=self.uniform(),
                                                         contrast=self.uniform(),contrast_center=self.uniform())),
            (ops.water(ampl_x=self.uniform,ampl_y=self.uniform,fill_value=self.uniform,
                             freq_X=self.uniform(),freq_y=self.uniform())),
            (ops.transforms.shear(angles=[self.uniform(),self.uniform()]))


        ]

        # Todo: Increase the list of dali ops
        #       See how to apply magnitude to each accordingly
        #       Run and pray to god

    def rand_aug(self):

        ops = random.choices( self.augment_list, k=self.n )
        for op, minval, maxval in ops:
            val = (float( self.m ) / 30) * float( maxval - minval ) + minval
            img = op( img, val )

        return img

    def train_transform(self, image):
        # breakpoint()
        image = self.crop( image )
        image = self.flip( image )
        image = self.colorjit_gray( image )
        image = self.blur( image )
        image = self.swapaxes( image )
        return image

    def train_rand_transform(self, image):
        breakpoint()
        image = rander( image )
        breakpoint()
        return image

    def val_transform(self, image):
        image = self.crop( image )
        image = self.swapaxes( image )
        return image

    def define_graph(self):
        breakpoint()
        jpegs, label = self.input()
        jpegs = self.decode( jpegs )
        breakpoint()
        if self.stage == 'train':
            self.transform = self.train_transform  # Todo: concern
        else:
            self.transform = self.val_transform

        batch = ()
        for i in range(self.copies):
            batch += (self.transform(jpegs), )
            breakpoint()
        if self.stage is not 'inference':
            label = label.gpu()
            label = self.to_int64(label)
            batch += (label, )
        return batch
