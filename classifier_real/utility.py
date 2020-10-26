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


def get_sample_counts(output_dir, dataset, class_names):
    """
    Get total and class-wise positive sample count of a dataset

    Arguments:
    output_dir - str, folder of dataset.csv
    dataset - str, train|dev|test
    class_names - list of str, target classes

    Returns:
    total_count - int
    class_positive_counts - dict of int, ex: {"Effusion": 300, "Infiltration": 500 ...}
    """
    df = pd.read_csv(os.path.join(output_dir, dataset + ".csv"))
    labels = df[class_names].as_matrix()
    total_count = labels.shape[0]
    positive_counts = np.sum(labels, axis=0)
    class_positive_counts = dict(zip(class_names, positive_counts))
    return total_count, class_positive_counts

def get_class_names(output_dir, dataset):
    df = pd.read_csv(os.path.join(output_dir, dataset + ".csv"))
    return list(df.columns[1:]) #Remove "Path" from names
