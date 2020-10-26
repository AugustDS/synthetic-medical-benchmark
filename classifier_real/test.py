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
from configparser import ConfigParser
from sklearn.metrics import roc_auc_score, roc_curve
from TFGenerator import TFWrapper
from utility import get_sample_counts, get_class_names
from keras.applications.densenet import DenseNet121
import importlib
from keras.layers import Input
from keras.layers.core import Dense
from keras.models import Model
import h5py
import pandas as pd
import sys
import argparse
import shutil 
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

def test(model_dir, results_dir, random_seed, resolution):
    np.random.seed(random_seed)
    tf.set_random_seed(np.random.randint(1 << 31))
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    set_session(sess)

    # parser config
    config_file = model_dir+ "/config.ini"
    print("Config File Path:", config_file,flush=True)
    assert os.path.isfile(config_file)
    cp = ConfigParser()
    cp.read(config_file)

    # default config
    image_dimension = cp["TRAIN"].getint("image_dimension")
    batch_size = cp["TEST"].getint("batch_size")
    use_best_weights = cp["TEST"].getboolean("use_best_weights")

    print("** DenseNet input resolution:", image_dimension, flush=True)
    print("** GAN image resolution:", resolution, flush=True)

    log2_record = int(np.log2(resolution))
    record_file_ending = "*"+ np.str(log2_record)+ ".tfrecords"
    print("** Resolution ", resolution, " corresponds to ", record_file_ending, " TFRecord file.", flush=True)

    output_dir  = os.path.join(results_dir, "classification_results_res_"+np.str(2**log2_record)+"/test")
    print("Output Directory:", output_dir,flush=True)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    if use_best_weights:
        print("** Using BEST weights",flush=True)
        model_weights_path = os.path.join(results_dir, "classification_results_res_"+np.str(2**log2_record)+"/train/best_weights.h5")
    else:
        print("** Using LAST weights",flush=True)
        model_weights_path = os.path.join(results_dir, "classification_results_res_"+np.str(2**log2_record)+"/train/weights.h5")


    # get test sample count
    shutil.copy(results_dir+"/test/test.csv", output_dir)
    tfrecord_dir_te = os.path.join(results_dir, "test")
    class_names = get_class_names(output_dir,"test")

    test_counts, _ = get_sample_counts(output_dir, "test", class_names)

    # get indicies (all of csv file for validation)
    print("** test counts:", test_counts, flush=True)

    # compute steps
    test_steps = int(np.floor(test_counts / batch_size))
    print("** test_steps:", test_steps, flush=True)

    # Get Model
    # ------------------------------------
    input_shape=(image_dimension, image_dimension, 3)
    img_input = Input(shape=input_shape)

    base_model = DenseNet121(
        include_top = False, 
        weights = None,
        input_tensor = img_input,
        input_shape = input_shape,
        pooling = "avg")

    x = base_model.output
    predictions = Dense(len(class_names), activation="sigmoid", name="predictions")(x)
    model = Model(inputs=img_input, outputs = predictions)

    print(" ** load model from:", model_weights_path, flush=True)
    model.load_weights(model_weights_path)
    # ------------------------------------


    print("** load test generator **", flush=True)
    test_seq = TFWrapper(
            tfrecord_dir=tfrecord_dir_te,
            record_file_endings = record_file_ending,
            batch_size = batch_size,
            model_target_size = (image_dimension, image_dimension),
            steps = None,
            augment=False,
            shuffle=False,
            prefetch=True,
            repeat=False)

    print("** make prediction **", flush=True)
    test_seq.initialise() #MAKE SURE REINIT
    y_hat = model.predict_generator(test_seq, workers=0)
    test_seq.initialise() #MAKE SURE REINIT
    y = test_seq.get_y_true()
    test_log_path = os.path.join(output_dir, "test.log")
    print("** write log to", test_log_path, flush=True)
    aurocs = []
    tpr_fpr_thr = []
    with open(test_log_path, "w") as f:
        for i in range(len(class_names)):
            tpr, fpr, thr = roc_curve(y[:, i], y_hat[:, i])
            roc_rates = np.concatenate((fpr.reshape(-1,1),tpr.reshape(-1,1),thr.reshape(-1,1)),axis=1)
            tpr_fpr_thr.append(roc_rates)
            try:
                score = roc_auc_score(y[:, i], y_hat[:, i])
                if score < 0.5:
                    score = 1.-score
                aurocs.append(score)
            except ValueError:
                score = 0
            f.write(np.str(class_names[i]) + " : " + np.str(score) + "\n")
        mean_auroc = np.mean(aurocs)
        f.write("-------------------------\n")
        f.write("mean auroc: " +np.str(mean_auroc) + "\n")
        print("mean auroc:", mean_auroc, flush=True)

    roc_char = np.asarray(tpr_fpr_thr)
    np.save(output_dir+"/roc_char.npy", roc_char)
    print("Saved ROC data (TPR, FPR, THR) to:", output_dir+"/roc_char.npy", flush=True)

def execute_cmdline(argv):
    prog = argv[0]
    parser = argparse.ArgumentParser()     
    subparsers = parser.add_subparsers(dest='command')
    subparsers.required = True
    def add_command(cmd, desc, example=None):
        epilog = 'Example: %s %s' % (prog, example) if example is not None else None
        return subparsers.add_parser(cmd, description=desc, help=desc, epilog=epilog)

    p = add_command('test',                            'Test Classifier.')
    p.add_argument('model_dir',                    help='Model Directory')
    p.add_argument('results_dir',                help='Results Directory')
    p.add_argument('random_seed',            type=int, help='Random Seed')
    p.add_argument('resolution',             type=int,  help='Resolution')

    args = parser.parse_args(argv[1:] if len(argv) > 1 else ['-h'])
    func = globals()[args.command]
    del args.command
    func(**vars(args))

#---------------------------------------------------------------------------

if __name__ == "__main__":
    execute_cmdline(sys.argv)
