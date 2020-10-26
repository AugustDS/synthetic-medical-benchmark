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

import pickle
import numpy as np
import tensorflow as tf
import PIL.Image
import pandas as pd
import os 
import sys
import argparse

class TFRecordExporter:
    def __init__(self, tfrecord_dir, expected_images, print_progress=True, progress_interval=10):
        self.tfrecord_dir       = tfrecord_dir
        self.tfr_prefix         = os.path.join(self.tfrecord_dir, os.path.basename(self.tfrecord_dir))
        self.expected_images    = expected_images
        self.cur_images         = 0
        self.shape              = None
        self.resolution_log2    = None
        self.tfr_writers        = []
        self.print_progress     = print_progress
        self.progress_interval  = progress_interval
        if self.print_progress:
            print('Creating dataset "%s"' % tfrecord_dir)
        if not os.path.isdir(self.tfrecord_dir):
            os.makedirs(self.tfrecord_dir)
        assert(os.path.isdir(self.tfrecord_dir))
        
    def close(self):
        if self.print_progress:
            print('%-40s\r' % 'Flushing data...', end='', flush=True)
        for tfr_writer in self.tfr_writers:
            tfr_writer.close()
        self.tfr_writers = []
        if self.print_progress:
            print('%-40s\r' % '', end='', flush=True)
            print('Added %d images.' % self.cur_images)

    def choose_shuffled_order(self): # Note: Images and labels must be added in shuffled order.
        order = np.arange(self.expected_images)
        np.random.RandomState(123).shuffle(order)
        return order

    def add_image(self, img):
        if self.print_progress and self.cur_images % self.progress_interval == 0:
            print('%d / %d\r' % (self.cur_images, self.expected_images), end='', flush=True)
        if self.shape is None:
            self.shape = img.shape
            self.resolution_log2 = int(np.log2(self.shape[1]))
            assert self.shape[0] in [1, 3]
            assert self.shape[1] == self.shape[2]
            #assert self.shape[1] == 2**self.resolution_log2
            tfr_opt = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.NONE)
            for lod in range(0,1):
                tfr_file = self.tfr_prefix + '-r%02d.tfrecords' % (self.resolution_log2 - lod)
                self.tfr_writers.append(tf.python_io.TFRecordWriter(tfr_file, tfr_opt))
        assert img.shape == self.shape
        for lod, tfr_writer in enumerate(self.tfr_writers):
            if lod:
                img = img.astype(np.float32)
                img = (img[:, 0::2, 0::2] + img[:, 0::2, 1::2] + img[:, 1::2, 0::2] + img[:, 1::2, 1::2]) * 0.25
            quant = np.rint(img).clip(0, 255).astype(np.uint8)
            ex = tf.train.Example(features=tf.train.Features(feature={
                'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=quant.shape)),
                'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[quant.tostring()]))}))
            tfr_writer.write(ex.SerializeToString())
        self.cur_images += 1

    def add_labels(self, labels):
        if self.print_progress:
            print('%-40s\r' % 'Saving labels...', end='', flush=True)
        assert labels.shape[0] == self.cur_images
        with open(self.tfr_prefix + '-rxx.labels', 'wb') as f:
            np.save(f, labels.astype(np.float32))
            
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()

def test(data_dir, results_dir, random_seed, batch_size = 20): 

	model_name_path = results_dir + "/network-final.pkl"
	print("Loading model: ", model_name_path, flush=True)
	print("Data Base Dir:", data_dir, flush=True)
	tf.InteractiveSession()
	csv_input_train = data_dir + "/train/train.csv"
	csv_input_valid = data_dir + "/valid/valid.csv"
	csv_input_test  = data_dir + "/test/test.csv"
	inf_path_train = results_dir + "/inference/train"
	inf_path_valid = results_dir + "/inference/valid"
	inf_path_test = results_dir + "/inference/test"
	csv_save_train = inf_path_train + "/train.csv" 
	csv_save_valid = inf_path_valid + "/valid.csv"
	csv_save_test = inf_path_test + "/test.csv"

	if not os.path.exists(inf_path_train):
		os.makedirs(inf_path_train)

	if not os.path.exists(inf_path_valid):
		os.makedirs(inf_path_valid)

	if not os.path.exists(inf_path_test):
		os.makedirs(inf_path_test)

	data_frame_tr = pd.read_csv(csv_input_train)
	data_frame_vl = pd.read_csv(csv_input_valid)
	data_frame_te = pd.read_csv(csv_input_test)

	split_headers = list(data_frame_tr.columns[0:]) #Remove "Path" from names

	paths_tr = []
	paths_vl = []
	paths_te = []
	np_labels_tr = []
	np_labels_vl = []
	np_labels_te = []

	for row in data_frame_tr.iterrows():
		np_labels_tr.append(row[1][1:].values)

	for row in data_frame_vl.iterrows():
		np_labels_vl.append(row[1][1:].values)

	for row in data_frame_te.iterrows():
		np_labels_te.append(row[1][1:].values)

	labels_arr_tr   = np.asarray(np_labels_tr)
	num_examples_tr = labels_arr_tr.shape[0]
	num_batches_tr  = np.int(np.ceil(num_examples_tr/batch_size))

	print("Train Label Data Shape:", labels_arr_tr.shape,flush=True)
	print("Number Train Batches:", num_batches_tr,flush=True)

	labels_arr_vl   = np.asarray(np_labels_vl)
	num_examples_vl = labels_arr_vl.shape[0]
	num_batches_vl  = np.int(np.ceil(num_examples_vl/batch_size))

	labels_arr_te   = np.asarray(np_labels_te)
	num_examples_te = labels_arr_te.shape[0]
	num_batches_te  = np.int(np.ceil(num_examples_te/batch_size))

	print("Validation Label Data Shape:", labels_arr_vl.shape,flush=True)
	print("Number Val Batches:", num_batches_vl,flush=True)

	print("Test Label Data Shape:", labels_arr_te.shape,flush=True)
	print("Number Test Batches:", num_batches_te,flush=True)

	# Import official network.
	with open(model_name_path, 'rb') as file:
		all_models = pickle.load(file)
		Gs = all_models[-1]

	# Create CSV File Train
	ids_i_tr   = np.arange(1,labels_arr_tr.shape[0]+1,1)
	ids_tr     = np.array([inf_path_train+"/"+np.str(np.int(ids_i_tr[j])) + ".jpg" for j in range(0,len(ids_i_tr))]).reshape(labels_arr_tr.shape[0],1)
	ids_and_labs_tr = np.concatenate((ids_tr,labels_arr_tr),axis=1)
	df_new_tr = pd.DataFrame(columns=split_headers,data=ids_and_labs_tr)
	df_new_tr.to_csv(csv_save_train, mode='w', header=True,index=False)

	# Create CSV File Val
	ids_i_vl   = np.arange(1,labels_arr_vl.shape[0]+1,1)
	ids_vl     = np.array([inf_path_valid+"/"+np.str(np.int(ids_i_vl[j])) + ".jpg" for j in range(0,len(ids_i_vl))]).reshape(labels_arr_vl.shape[0],1)
	ids_and_labs_vl = np.concatenate((ids_vl,labels_arr_vl),axis=1)
	df_new_vl = pd.DataFrame(columns=split_headers,data=ids_and_labs_vl)
	df_new_vl.to_csv(csv_save_valid, mode='w', header=True,index=False)

	# Create CSV File Test
	ids_i_te   = np.arange(1,labels_arr_te.shape[0]+1,1)
	ids_te     = np.array([inf_path_test+"/"+np.str(np.int(ids_i_te[j])) + ".jpg" for j in range(0,len(ids_i_te))]).reshape(labels_arr_te.shape[0],1)
	ids_and_labs_te = np.concatenate((ids_te,labels_arr_te),axis=1)
	df_new_te = pd.DataFrame(columns=split_headers,data=ids_and_labs_te)
	df_new_te.to_csv(csv_save_test, mode='w', header=True,index=False)

	# Generate Latents
	latents_all = np.random.RandomState(random_seed).randn(num_examples_tr+num_examples_vl+num_examples_te,*Gs.input_shapes[0][1:])
	# Split latents for train, val, test
	latents_tr  = latents_all[0:num_examples_tr,:]
	latents_vl  = latents_all[num_examples_tr:num_examples_tr+num_examples_vl,:]
	latents_te  = latents_all[num_examples_tr+num_examples_vl:,:]
	assert latents_tr.shape[0] == num_examples_tr
	assert latents_vl.shape[0] == num_examples_vl
	assert latents_te.shape[0] == num_examples_te

	# Generate Train Images, save as tf_record files
	print('Generating inference train images..', flush=True)
	for i in range(0,num_batches_tr):
		labels  = labels_arr_tr[i*batch_size:(i+1)*batch_size]
		latents = latents_tr[i*batch_size:(i+1)*batch_size]
		images  = Gs.run(latents, labels)
		images  = np.clip(np.rint((images + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8)
		if i == 0:
			all_images = images 
		else:
			all_images = np.concatenate((all_images,images),axis=0)

	print('Saving train inference TFRecords..', flush=True)
	with TFRecordExporter(inf_path_train, all_images.shape[0], progress_interval=5000) as tfr:
		order = tfr.choose_shuffled_order()
		for idxi in range(order.size):
			tfr.add_image(all_images[order[idxi]])
		tfr.add_labels(labels_arr_tr[order])
	print('TFRecord done for train inference.', flush=True)

	# VALIDATION
	print('Generating inference valid images..', flush=True)
	for i in range(0,num_batches_vl):
		labels  = labels_arr_vl[i*batch_size:(i+1)*batch_size]
		latents = latents_vl[i*batch_size:(i+1)*batch_size]
		images  = Gs.run(latents, labels)
		images = np.clip(np.rint((images + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8)
		if i == 0:
			all_images = images 
		else:
			all_images = np.concatenate((all_images,images),axis=0)

	print('Saving valid inference TFRecords..', flush=True)
	with TFRecordExporter(inf_path_valid, all_images.shape[0], progress_interval=1000) as tfr:
		order = tfr.choose_shuffled_order()
		for idxi in range(order.size):
			tfr.add_image(all_images[order[idxi]])
		tfr.add_labels(labels_arr_vl[order])

	print('Generating inference test images..', flush=True)
	for i in range(0,num_batches_te):
		labels  = labels_arr_te[i*batch_size:(i+1)*batch_size]
		latents = latents_te[i*batch_size:(i+1)*batch_size]
		images  = Gs.run(latents, labels)
		images = np.clip(np.rint((images + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8)
		if i == 0:
			all_images = images 
		else:
			all_images = np.concatenate((all_images,images),axis=0)

	print('Saving test inference TFRecords..', flush=True)
	with TFRecordExporter(inf_path_test, all_images.shape[0], progress_interval=1000) as tfr:
		order = tfr.choose_shuffled_order()
		for idxi in range(order.size):
			tfr.add_image(all_images[order[idxi]])
		tfr.add_labels(labels_arr_te[order])

	print("** Inference Done!", flush=True)

def execute_cmdline(argv):
    prog = argv[0]
    parser = argparse.ArgumentParser()     
    subparsers = parser.add_subparsers(dest='command')
    subparsers.required = True
    def add_command(cmd, desc, example=None):
        epilog = 'Example: %s %s' % (prog, example) if example is not None else None
        return subparsers.add_parser(cmd, description=desc, help=desc, epilog=epilog)

    p = add_command('test',                                      'Testing.')
    p.add_argument('data_dir',                        help='Data load Path')
    p.add_argument('results_dir',                  help='Results Directory')
    p.add_argument('random_seed',              type=int, help='Random Seed')

    args = parser.parse_args(argv[1:] if len(argv) > 1 else ['-h'])
    func = globals()[args.command]
    del args.command
    func(**vars(args))
#----------------------------------------------------------------------------

if __name__ == "__main__":
    execute_cmdline(sys.argv)
