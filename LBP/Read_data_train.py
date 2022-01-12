import os
import cv2
import numpy as np
from sklearn.utils import shuffle
import joblib
import argparse
from function import *
from function_new import *

pos_dir = 'Path to positive image'
neg_dir = 'Path to negative image'

X, y, pos_count, neg_count = read_images(pos_dir, neg_dir)
X, y = np.array(X), np.array(y)
X, y = shuffle(X, y, random_state=0)

print('Number of positive images:', pos_count)
print('Number of negative images:', neg_count)
