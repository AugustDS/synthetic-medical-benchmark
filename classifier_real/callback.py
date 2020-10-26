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
import keras.backend as kb
import numpy as np
import os
import shutil
import warnings
import pandas as pd
from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score
import keras.backend as K
class MultipleClassAUROC(Callback):
    """
    Monitor mean AUROC and update model
    """
    def __init__(self, sequence, class_names, weights_path, stats=None, early_stop_p=3, learn_rate_p=2, learn_rate_f=0.1, min_lr=1e-8, workers=0):
        super(Callback, self).__init__()
        self.sequence = sequence
        self.workers = workers
        self.class_names = class_names
        self.weights_path = weights_path
        self.best_weights_path = os.path.join(
            os.path.split(weights_path)[0],
            "best_"+os.path.split(weights_path)[1],
        )

        self.best_auroc_log_path = os.path.join(
            os.path.split(weights_path)[0],
            "best_auroc.log",
        )

        self.val_aurocs_tr = os.path.join(
            os.path.split(weights_path)[0],
            "val_aurocs_tr.csv",
        )

        self.stats_output_path = os.path.join(
            os.path.split(weights_path)[0],
            ".training_stats.json"
        )
        # for resuming previous training
        if stats:
            self.stats = stats
        else:
            self.stats = {"best_mean_auroc": 0}

        # Early stopping and Patience 
        self.epoch_since_es = 0
        self.epoch_since_lr = 0
        self.early_stop_p = early_stop_p
        self.learn_rate_p = learn_rate_p
        self.learn_rate_f = learn_rate_f
        self.min_lr = min_lr

        # aurocs log
        self.aurocs = {}
        for c in self.class_names:
            self.aurocs[c] = []

    def on_epoch_end(self, epoch, logs={}):
        """
        Calculate the average AUROC and save the best model weights according
        to this metric.
        """

        print("\n*********************************", flush=True)
        self.stats["lr"] = float(kb.eval(self.model.optimizer.lr))
        print("current learning rate:", self.stats['lr'], flush=True)

        self.sequence.initialise() #MAKE SURE REINIT
        y_hat = self.model.predict_generator(self.sequence, workers=0)

        self.sequence.initialise()
        y = self.sequence.get_y_true()

        print("*** epoch", epoch + 1, ", validation auroc ***", flush=True)
        current_auroc = []

        ####
        for i in range(len(self.class_names)):
            try:
                score = roc_auc_score(y[:, i], y_hat[:, i])
            except ValueError:
                score = 0
            self.aurocs[self.class_names[i]].append(score)
            current_auroc.append(score)
            print(i+1,self.class_names[i],": ", score, flush=True)
        print("*********************************", flush=True)

        # customize your multiple class metrics here
        mean_auroc = np.mean(current_auroc)
        print("mean auroc: ", mean_auroc, flush=True)

        # Save aurocs to excel file
        all_cur_auroc = [epoch]+current_auroc+[mean_auroc]
        all_names     = ["Epoch"]+self.class_names+["Mean_auroc"]
        df = pd.DataFrame(data=np.asarray(all_cur_auroc).reshape(1,len(all_cur_auroc)), index=None, columns=all_names)
        hdr = False  if os.path.isfile(self.val_aurocs_tr) else True
        df.to_csv(self.val_aurocs_tr, mode='a', header=hdr, index=None) 
        ####

        if mean_auroc > self.stats["best_mean_auroc"]:
            print("update best mean validation auroc from ", self.stats['best_mean_auroc'], " to ", mean_auroc, flush=True)

            # 1. copy best model
            shutil.copy(self.weights_path, self.best_weights_path)

            # 2. update log file
            print("update log file:", self.best_auroc_log_path, flush=True)
            with open(self.best_auroc_log_path, "a") as f:
                f.write(np.str(epoch + 1) + " mean val. auroc: " + np.str(mean_auroc) + " lr: " + np.str(self.stats['lr'])+ "\n")

            # 3. write stats output, this is used for resuming the training
            with open(self.stats_output_path, 'w') as f:
                json.dump(self.stats, f)

            # Reset early_stopping epochs since improvements 
            self.epoch_since_es = 0
            self.epoch_since_lr = 0

            print("update model file:", self.weights_path, "->", self.best_weights_path, flush=True)
            self.stats["best_mean_auroc"] = mean_auroc
            print("*********************************", flush=True)

        else:
            self.epoch_since_es += 1
            self.epoch_since_lr += 1
            print("MEAN AUROC has not improved for", self.epoch_since_es,"/",self.early_stop_p, "epochs.", flush=True)
            if self.epoch_since_es >= self.early_stop_p:
                print("Early stopping!", flush=True)
                self.model.stop_training = True
            if self.epoch_since_lr >= self.learn_rate_p:
                self.epoch_since_lr = 0 
                old_lr = float(K.get_value(self.model.optimizer.lr))
                if old_lr > self.min_lr:
                    new_lr = old_lr * self.learn_rate_f
                    new_lr = max(new_lr, self.min_lr)
                    K.set_value(self.model.optimizer.lr, new_lr)
                    print("Learning Rate reduced from", old_lr, "to", new_lr, flush=True)
        return