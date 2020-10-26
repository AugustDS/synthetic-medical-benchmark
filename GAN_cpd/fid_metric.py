# Copyright 2018, Tero Karras, NVIDIA CORPORATION
#
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
#
# Adapted from the original implementation by Tero Karras.
# Source https://github.com/tkarras/progressive_growing_of_gans

import os
import time
import re
import bisect
from collections import OrderedDict
import numpy as np
import tensorflow as tf
import scipy.ndimage
import scipy.misc
import metrics 
import config
import misc
import tfutil


#----------------------------------------------------------------------------
# Evaluate FID metrics on the Run.

def compute_fid(Gs, minibatch_size, dataset_obj, iter_number,lod=0,num_images=10000, printing=True):

    # Initialize metrics.
    from metrics.frechet_inception_distance import API as class_def 
    image_shape = [3] + dataset_obj.shape[1:]
    obj = class_def(num_images=num_images, image_shape=image_shape, image_dtype=np.uint8, minibatch_size=minibatch_size)
    mode = 'warmup'
    obj.begin(mode)
    for idx in range(10):
        obj.feed(mode, np.random.randint(0, 256, size=[minibatch_size]+image_shape, dtype=np.uint8))
    obj.end(mode)

    # Print table header
    if printing:
        print(flush=True)
        print('%-10s%-12s' % ('KIMG', 'Time_eval'), end='',flush=True)
        print('%-12s' % ('FID'), end='',flush=True)
        print(flush=True)
        print('%-10s%-12s%-12s' % ('---', '---','---'), end='',flush=True)
        print(flush=True)

    # Feed in reals.
    print('%-10s' % "Reals", end='',flush=True)
    time_begin = time.time()
    labels = np.zeros([num_images, dataset_obj.label_size], dtype=np.float32)
    obj.begin(mode)
    for begin in range(0, num_images, minibatch_size):
        end = min(begin + minibatch_size, num_images)
        images, labels[begin:end] = dataset_obj.get_minibatch_np(end - begin, lod=lod)
        if images.shape[1] == 1:
            images = np.tile(images, [1, 3, 1, 1]) # grayscale => RGB
            obj.feed(mode, images)         
    results = obj.end(mode)
    if printing:
        print('%-12s' % misc.format_time(time.time() - time_begin), end='',flush=True)
        print(results[0], end='',flush=True)
        print(flush=True)

    # Evaluate each network snapshot.
    if printing:
        print('%-10d' % iter_number, end='',flush=True)
    mode ='fakes'
    obj.begin(mode) 
    time_begin = time.time()
    for begin in range(0, num_images, minibatch_size):
        end = min(begin + minibatch_size, num_images)
        latents = misc.random_latents(end - begin, Gs)
        images = Gs.run(latents, labels[begin:end], num_gpus=config.num_gpus, out_mul=127.5, out_add=127.5, out_dtype=np.uint8)
        if images.shape[1] == 1:
            images = np.tile(images, [1, 3, 1, 1]) # grayscale => RGB
        obj.feed(mode, images)
    results = obj.end(mode)
    if printing: 
        print('%-12s' % misc.format_time(time.time() - time_begin), end='',flush=True)
        print(results[0], end='',flush=True)
        print(flush=True)
    return results[0]

#----------------------------------------------------------------------------
