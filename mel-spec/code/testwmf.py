#!/usr/bin/python

""" Run this file from input folder to create labels file in 
input directory """

from __future__ import print_function

import numpy as np
import pickle
import sys
import os
from numpy import array

if __name__ == "__main__":

    # Keep it eqaual to the number of songs in MSD data folder
    num_songs = 100
    # keep it equal to the last output layer of the model
    emb_dim = 10

    #Load Data
    labels = np.random.uniform(-1.0,1.0,(num_songs,emb_dim))
    #print(labels)

    # Serializes the numpy array and writes it to labels file
    with open("labels", 'wb') as f:
        f.write(pickle.dumps(array(labels)))