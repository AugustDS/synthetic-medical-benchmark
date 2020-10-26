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

import json
import shutil
import os
from sklearn import model_selection
import pickle
from callback import MultipleClassAUROC 
from configparser import ConfigParser
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from utility import get_sample_counts, get_class_names
from weights import get_class_weights
from TFGenerator import TFWrapper
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.applications.densenet import DenseNet121
import importlib
from keras.layers import Input
from keras.layers.core import Dense
from keras.models import Model
import h5py
import numpy as np
import sys
import argparse

def train(model_dir, results_dir, random_seed, resolution):
    np.random.seed(random_seed)
    tf.set_random_seed(np.random.randint(1 << 31))
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    session_conf.gpu_options.allow_growth = True
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    set_session(sess)

    # parser config
    config_file = model_dir+ "/config.ini"
    print("Config File Path:", config_file, flush=True)
    assert os.path.isfile(config_file)
    cp = ConfigParser()
    cp.read(config_file)

    # default config
    base_model_name = cp["DEFAULT"].get("base_model_name")

    # train config
    path_model_base_weights = cp["TRAIN"].get("path_model_base_weights")
    use_trained_model_weights = cp["TRAIN"].getboolean("use_trained_model_weights")
    use_best_weights = cp["TRAIN"].getboolean("use_best_weights")
    output_weights_name = cp["TRAIN"].get("output_weights_name")
    epochs = cp["TRAIN"].getint("epochs")
    batch_size = cp["TRAIN"].getint("batch_size")
    initial_learning_rate = cp["TRAIN"].getfloat("initial_learning_rate")
    image_dimension = cp["TRAIN"].getint("image_dimension")
    patience_reduce_lr = cp["TRAIN"].getint("patience_reduce_lr")
    min_lr = cp["TRAIN"].getfloat("min_lr")
    positive_weights_multiply = cp["TRAIN"].getfloat("positive_weights_multiply")
    patience = cp["TRAIN"].getint("patience")
    samples_per_epoch = cp["TRAIN"].getint("samples_per_epoch")
    reduce_lr = cp["TRAIN"].getfloat("reduce_lr")

    print("** DenseNet input resolution:", image_dimension, flush=True)
    print("** GAN image resolution:", resolution, flush=True)
    print("** Patience epochs", patience, flush=True)
    print("** Samples per epoch:", samples_per_epoch, flush=True)

    log2_record = int(np.log2(resolution))
    record_file_ending = "*"+ np.str(log2_record)+".tfrecords"
    print("** Resolution ", resolution, " corresponds to ", record_file_ending, " TFRecord file.", flush=True)

    output_dir  = os.path.join(results_dir, "classification_results_res_"+ np.str(2**log2_record)+"/train")
    print("Output Directory:", output_dir, flush=True)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # if previously trained weights is used, never re-split
    if use_trained_model_weights:
        print("** use trained model weights **", flush=True)
        training_stats_file = os.path.join(output_dir, ".training_stats.json")
        if os.path.isfile(training_stats_file):
            # TODO: add loading previous learning rate?
            training_stats = json.load(open(training_stats_file))
        else:
            training_stats = {}
    else:
        # start over
        training_stats = {}

    show_model_summary = cp["TRAIN"].getboolean("show_model_summary")
    running_flag_file = os.path.join(output_dir, ".training.lock")
    if os.path.isfile(running_flag_file):
        raise RuntimeError("A process is running in this directory!!!")
    else:
        open(running_flag_file, "a").close()

    try:
        print("backup config file to", output_dir,flush=True)
        shutil.copy(config_file, os.path.join(output_dir, os.path.split(config_file)[1]))

        tfrecord_dir_tr = os.path.join(results_dir, "train")
        tfrecord_dir_vl = os.path.join(results_dir, "valid")

        shutil.copy(tfrecord_dir_tr+"/train.csv", output_dir)
        shutil.copy(tfrecord_dir_vl+"/valid.csv", output_dir)

        # Get class names 
        class_names = get_class_names(output_dir,"train")

        # get train sample counts
        train_counts, train_pos_counts = get_sample_counts(output_dir, "train", class_names)
        valid_counts, _                = get_sample_counts(output_dir, "valid", class_names)
        
        print("Total Training Data:", train_counts, flush=True)
        print("Total Validation Data:", valid_counts , flush=True)
        train_steps = int(min(samples_per_epoch,train_counts)/batch_size)
        print("** train_steps:", train_steps, flush=True)
        validation_steps = int(np.floor(valid_counts/ batch_size))
        print("** validation_steps:", validation_steps, flush=True)

        # compute class weights
        print("** compute class weights from training data **", flush=True)
        class_weights = get_class_weights(
            train_counts,
            train_pos_counts,
            multiply=positive_weights_multiply,
        )
        print("** class_weights **", flush=True)
        print(class_weights)

        print("** load model **", flush=True)
        if use_trained_model_weights:
            if use_best_weights:
                model_weights_file = os.path.join(output_dir, "best_" + output_weights_name)
            else:
                model_weights_file = os.path.join(output_dir, output_weights_name)
        else:
            model_weights_file = None

        # Use downloaded weights 
        if os.path.isfile(path_model_base_weights):
            base_weights = path_model_base_weights
            print("** Base weights will be loaded.", flush=True)
        else:
            base_weights = None
            print("** No Base weights.", flush=True)

        # Get Model
        # ------------------------------------
        input_shape=(image_dimension, image_dimension, 3)
        img_input = Input(shape=input_shape)

        base_model = DenseNet121(
            include_top = False, 
            weights = base_weights,
            input_tensor = img_input,
            input_shape = input_shape,
            pooling = "avg")

        x = base_model.output
        predictions = Dense(len(class_names), activation="sigmoid", name="predictions")(x)
        model = Model(inputs=img_input, outputs = predictions)

        if use_trained_model_weights and model_weights_file != None:
            print("** load model weights_path:", model_weights_file, flush=True)
            model.load_weights(model_weights_file)
        # ------------------------------------
        
        if show_model_summary:
            print(model.summary())

        print("** create image generators", flush=True)
        train_seq = TFWrapper(
            tfrecord_dir=tfrecord_dir_tr,
            record_file_endings = record_file_ending,
            batch_size = batch_size,
            model_target_size = (image_dimension, image_dimension),
            steps = train_steps,
            augment=True,
            shuffle=True,
            prefetch=True,
            repeat=True)

        valid_seq = TFWrapper(
            tfrecord_dir=tfrecord_dir_vl,
            record_file_endings = record_file_ending,
            batch_size = batch_size,
            model_target_size = (image_dimension, image_dimension),
            steps = None,
            augment=False,
            shuffle=False,
            prefetch=True,
            repeat=True)

        # Initialise train and valid iterats
        print("** Initialise train and valid iterators", flush=True)
        train_seq.initialise()
        valid_seq.initialise()

        output_weights_path = os.path.join(output_dir, output_weights_name)
        print("** set output weights path to:", output_weights_path,flush=True)

        print("** SINGLE_gpu_model is used!", flush=True)
        model_train = model
        checkpoint = ModelCheckpoint(
             output_weights_path,
             save_weights_only=True,
             save_best_only=False,
             verbose=1,
        )

        print("** compile model with class weights **",flush=True)
        optimizer = Adam(lr=initial_learning_rate)
        model_train.compile(optimizer=optimizer, loss="binary_crossentropy")
        
        auroc = MultipleClassAUROC(
            sequence=valid_seq,
            class_names=class_names,
            weights_path=output_weights_path,
            stats=training_stats,
            early_stop_p=patience, 
            learn_rate_p=patience_reduce_lr, 
            learn_rate_f=reduce_lr, 
            min_lr=min_lr,
            workers=0
        )

        callbacks = [
            checkpoint,
            TensorBoard(log_dir=os.path.join(output_dir, "logs"), batch_size=batch_size),
            auroc]

        print("** start training **",flush=True)
        history = model_train.fit_generator(
            generator=train_seq,
            steps_per_epoch=train_steps,
            epochs=epochs,
            validation_data=valid_seq,
            validation_steps=validation_steps,
            callbacks=callbacks,
            class_weight=class_weights,
            workers=0,
            shuffle=False,
        )

        # dump history
        print("** dump history **",flush=True)
        with open(os.path.join(output_dir, "history.pkl"), "wb") as f:
            pickle.dump({
                "history": history.history,
                "auroc": auroc.aurocs,
            }, f)
        print("** done! **",flush=True)

    finally:
        os.remove(running_flag_file)


def execute_cmdline(argv):
    prog = argv[0]
    parser = argparse.ArgumentParser()     
    subparsers = parser.add_subparsers(dest='command')
    subparsers.required = True
    def add_command(cmd, desc, example=None):
        epilog = 'Example: %s %s' % (prog, example) if example is not None else None
        return subparsers.add_parser(cmd, description=desc, help=desc, epilog=epilog)

    p = add_command('train',                            'Train Classifier.')
    p.add_argument('model_dir',                    help='Model/Config Dir' )
    p.add_argument('results_dir',                  help='Results Directory')
    p.add_argument('random_seed',              type=int, help='Random Seed')
    p.add_argument('resolution',               type=int,  help='Resolution')

    args = parser.parse_args(argv[1:] if len(argv) > 1 else ['-h'])
    func = globals()[args.command]
    del args.command
    func(**vars(args))

#---------------------------------------------------------------------------

if __name__ == "__main__":
    execute_cmdline(sys.argv)
