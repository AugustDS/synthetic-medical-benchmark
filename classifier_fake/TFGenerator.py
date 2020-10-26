# Copyright (c) 2018 Bruce Chou
#
# Licensed under the MIT License;
# you may not use this file except in compliance with the License.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Adapted from the original implementation by Bruce Chou.
# Source: https://github.com/brucechou1983/CheXNet-Keras

import numpy as np
import os
import pandas as pd
import keras.backend as K
import tensorflow as tf 
import glob 
from tensorflow.image import ResizeMethod
from keras.utils import Sequence


def parse_tfrecord_np(record):
    ex = tf.train.Example()
    ex.ParseFromString(record)
    shape = ex.features.feature['shape'].int64_list.value
    data = ex.features.feature['data'].bytes_list.value[0]
    return np.fromstring(data, np.uint8).reshape(shape)
def parse_tfrecord_tf(record):
    features = tf.parse_single_example(record, features={
        'shape': tf.FixedLenFeature([3], tf.int64),
        'data': tf.FixedLenFeature([], tf.string)})
    data = tf.decode_raw(features['data'], tf.uint8)
    return tf.reshape(data, features['shape'])

class TFWrapper(Sequence):
    def __init__(self, tfrecord_dir,
                    record_file_endings = None, 
                    batch_size=16,
                    model_target_size=(224, 224),
                    steps=None,
                    augment=True,
                    shuffle=True,
                    prefetch=True,
                    repeat=True):

        self.tfrecord_dir = tfrecord_dir
        self.batch_size = batch_size
        self.model_target_size = model_target_size
        self.augment = augment
        self.shuffle = shuffle
        self.prefetch = prefetch
        self.repeat = repeat
        self.dtype = 'uint8'

        if record_file_endings is None:
            search = '*.tfrecords'
        else:
            search = record_file_endings

        assert os.path.isdir(self.tfrecord_dir)
        tfr_file = sorted(glob.glob(os.path.join(self.tfrecord_dir, search)))[0]
        tfr_opt = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.NONE)
        for record in tf.python_io.tf_record_iterator(tfr_file, tfr_opt):
            tfr_shape = parse_tfrecord_np(record).shape
            break
        bytes_per_item = np.prod(tfr_shape) * np.dtype(self.dtype).itemsize
        label_file = sorted(glob.glob(os.path.join(self.tfrecord_dir, '*.labels')))[0]
        np_labels = np.load(label_file)
        self.label_size = np_labels.shape[1]

        if steps is None:
            self.steps = int(np.floor(np_labels.shape[0]/float(self.batch_size)))
        else:
            self.steps = steps
            
        tf_labels_dataset = tf.data.Dataset.from_tensor_slices(np_labels)
        dset = tf.data.TFRecordDataset(tfr_file, compression_type='', buffer_size=256<<20)
        dset = dset.map(parse_tfrecord_tf)
        dset = tf.data.Dataset.zip((dset, tf_labels_dataset))
        dset = dset.map(self.preprocess_image)
        if self.prefetch:
            dset = dset.prefetch(((2048 << 20) - 1) // bytes_per_item + 1)
        if self.shuffle:
            dset = dset.shuffle(10000)
        if self.repeat:
            dset = dset.repeat()
        dset = dset.batch(batch_size)
        self.dataset = dset 
        self.iterator = tf.data.Iterator.from_structure(self.dataset.output_types, self.dataset.output_shapes)
        self.init_dset = self.iterator.make_initializer(self.dataset)

    def __bool__(self):
        return True

    def __len__(self):
        return self.steps 

    def __getitem__(self, idx):
        batch    = self.iterator.get_next()
        return K.get_session().run(batch)

    def initialise(self):
        K.get_session().run(self.init_dset)
        
    def get_y_true(self):
        if self.shuffle:
            raise ValueError("""
            You're trying run get_y_true() when generator option 'shuffle_on_epoch_end' is True.
            """)
        batch  = self.iterator.get_next()
        return np.asarray([K.get_session().run(batch[1]) for i in range(0,self.steps)]).reshape(-1,self.label_size)

    def preprocess_image(self,x,y):
        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.float32)
        x = tf.transpose(x,[1,2,0]) # NCHW-> NHWC
        x = tf.tile(x,[1,1,3])
        x = x / 255. 
        x = tf.image.resize_images(x, self.model_target_size) 
        imagenet_mean = np.array([0.485, 0.456, 0.406])
        imagenet_std = np.array([0.229, 0.224, 0.225])
        x = (x - imagenet_mean)/imagenet_std
        if self.augment is True:
            mask = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)
            x = tf.cond(mask<0.5,lambda:x,lambda:tf.reverse(x, axis=[1]))
        return x,y

    def on_epoch_end(self):
        pass
        #self.initialise()
