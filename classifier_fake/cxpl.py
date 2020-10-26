import numpy as np
import os
import pandas as pd
from PIL import Image
from configparser import ConfigParser
from keras.applications.densenet import DenseNet121
import importlib
from keras.layers import Input
from utility import get_class_names
from keras.layers.core import Dense
from keras.models import Model
import h5py
import sys
import argparse
import shutil 
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
from tensorflow.python.keras.losses import categorical_crossentropy
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#----------------------------------------------------------------------------
# Convenience func that normalizes labels.
def normalize_labels(lab):
    labels_sum = np.sum(lab,axis=1).reshape(-1,1)
    lab_new = np.divide(lab,labels_sum)
    return lab_new
#----------------------------------------------------------------------------
# Mask input by numpy multiplication.
def mask_input(x,i,j,BS,C,H,W,ds):
    mask = np.ones([H,W,C])
    mask[i*ds:(i+1)*ds,j*ds:(j+1)*ds,:] = np.zeros([ds,ds,C])
    return np.multiply(x,mask)
#----------------------------------------------------------------------------
# Upscale cxplain attention map
# x = [BS,H,W,C]
def upscale2d(x, factor=2):
    x = np.transpose(x,[0,3,1,2]) #[BS,H,W,C]->[BS,C,H,W]
    assert isinstance(factor, int) and factor >= 1
    if factor == 1: return x
    s = x.shape
    x = np.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
    x = np.repeat(x,factor,axis=3)
    x = np.repeat(x,factor,axis=5)
    x = np.reshape(x,[-1,s[1],s[2] * factor, s[3] * factor])
    x = np.transpose(x,[0,2,3,1])
    return x
#----------------------------------------------------------------------------
# BCE loss in numpy
def binary_crossentropy(output,target,epsilon=1e-07):
    output = np.clip(output, epsilon, 1. - epsilon)
    bce = target * np.log(output+epsilon)
    bce += (1 - target) * np.log(1 - output+epsilon)
    return np.mean(-bce,axis=1)
#----------------------------------------------------------------------------
# Computes delta maps as new input for discrimiantor 
# x = [BS,H,W,C]
# labels = [BS,Y]
def get_delta_map(x, model, labels, 
                  downsample_factor=2, 
                  log_transform=False,
                  normalize=False):
    
    BS, H, W, C = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
    H_new = (H//downsample_factor)
    W_new = (W//downsample_factor)
    num_masks = H_new*W_new
    
    # Tile and replicate 
    x_tiled = np.reshape(x,[1,BS,H,W,C])
    x_rep   = np.repeat(x_tiled,num_masks,axis=0)

    #Get masked tensors and compute delta_errors
    base_loss = binary_crossentropy(output=model.predict(x), target=labels)

    idx = 0
    delta_errors = []
    for i in range(0,H_new):
        for j in range(0,W_new):
            x_mask = mask_input(x_rep[idx],i,j,BS=BS,C=C,H=H,W=W,ds=downsample_factor)
            loss   = binary_crossentropy(output=model.predict(x_mask), target=labels)
            delta  = np.maximum(loss-base_loss,1e-07)
            if log_transform:
                delta = np.log(1.0 + delta)
            delta_errors.append(delta)
            idx += 1
    delta_errors = np.asarray(delta_errors)                     #[num_masks,BS,1]
    delta_errors = np.transpose(delta_errors,[1,0])             #[BS,num_masks]
    delta_map   = np.reshape(delta_errors, [BS,H_new,W_new,1])  #[BS,H_new,W_new,1]
    delta_map   = upscale2d(delta_map,factor=downsample_factor) #[BS,H,W,1]

    if normalize:
        delta_map_sum = np.sum(delta_map,axis=(1,2,3)).reshape(-1,1,1,1)
        delta_map = delta_map / delta_map_sum
    return delta_map

def cxpl(model_dir, results_dir):
    np.random.seed(0)
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

    output_dir = os.path.join(results_dir, "classification_results/test")
    print("Output Directory:", output_dir,flush=True)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)


    # default config
    image_dimension = cp["TRAIN"].getint("image_dimension")
    batch_size = cp["TEST"].getint("batch_size")
    use_best_weights = cp["TEST"].getboolean("use_best_weights")
    batchsize_cxpl = cp["CXPL"].getint("batchsize_cxpl")

    if use_best_weights:
        print("** Using BEST weights",flush=True)
        model_weights_path = os.path.join(results_dir, "classification_results/train/best_weights.h5")
    else:
        print("** Using LAST weights",flush=True)
        model_weights_path = os.path.join(results_dir, "classification_results/train/weights.h5")

    print("** DenseNet Input Resolution:", image_dimension, flush=True)
    class_names = get_class_names(output_dir,"test")

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

    # Load Paths & Labels
    print(" ** load .csv and images.", flush=True)
    paths=[]
    labels=[]
    df_nn = pd.read_csv(output_dir+"/nn_files/nn_path_and_labels.csv")
    for row in df_nn.iterrows():
        labels.append(row[1][1:].astype(np.float32))
        paths.append(row[1][0])

    y_cx = np.asarray(labels)
    all_paths = np.asarray(paths)

    # Load Images
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])
    imgs = []
    for path in paths:
        img = Image.open(output_dir+"/nn_files/"+path)
        img = np.asarray(img.convert("L"))
        img = img / 255.
        img = np.reshape(img,[img.shape[0],img.shape[1],1])
        img = np.repeat(img,3,axis=2)
        img = (img - imagenet_mean)/imagenet_std
        imgs.append(img)
    
    x_cx = np.asarray(imgs)

    print(" ** compute attribution maps.", flush=True)
    # Compute causal contribution:
    n_max = x_cx.shape[0]//batchsize_cxpl
    for i in range(0,n_max):
        attribution_map = get_delta_map(x=x_cx[i*batchsize_cxpl:(i+1)*batchsize_cxpl], model=model, labels=y_cx[i*batchsize_cxpl:(i+1)*batchsize_cxpl], 
                      downsample_factor=4, 
                      log_transform=False,
                      normalize=False)
        if i == 0:
            amap_final = attribution_map
        else:
            amap_final = np.concatenate((amap_final,attribution_map), axis=0)
    x_unnorm = x_cx*imagenet_std+imagenet_mean
    print(" ** save under", output_dir, "/nn_files", flush=True)
    np.save(output_dir+"/nn_files/y_cx_nn.npy", y_cx[:n_max*batchsize_cxpl])
    np.save(output_dir+"/nn_files/x_cx_nn.npy", x_unnorm[:n_max*batchsize_cxpl])
    np.save(output_dir+"/nn_files/attr_nn.npy", amap_final)

    print(" ** done.", flush=True)

def execute_cmdline(argv):
    prog = argv[0]
    parser = argparse.ArgumentParser()     
    subparsers = parser.add_subparsers(dest='command')
    subparsers.required = True
    def add_command(cmd, desc, example=None):
        epilog = 'Example: %s %s' % (prog, example) if example is not None else None
        return subparsers.add_parser(cmd, description=desc, help=desc, epilog=epilog)

    p = add_command('cxpl',                            'Test Classifier.')
    p.add_argument('model_dir',                    help='Model Directory')
    p.add_argument('results_dir',                help='Results Directory')

    args = parser.parse_args(argv[1:] if len(argv) > 1 else ['-h'])
    func = globals()[args.command]
    del args.command
    func(**vars(args))
#---------------------------------------------------------------------------

if __name__ == "__main__":
    execute_cmdline(sys.argv)
