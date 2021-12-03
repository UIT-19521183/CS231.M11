import os
import cv2
import numpy as np
from sklearn.utils import shuffle
import joblib
import argparse
from function import *

parser = argparse.ArgumentParser(description='Parse Training Directory')
parser.add_argument('--path', help='Path to directory contraining training images')

args = parser.parse_args()
train_dir = args.path
pos_dir = os.path.join(train_dir, 'pos')
neg_dir = os.path.join(train_dir, 'neg')

X, y, pos_count, neg_count = read_images(pos_dir, neg_dir)
X, y = np.array(X), np.array(y)
X, y = shuffle(X, y, random_state=0)

print('Number of positive images:', pos_count)
print('Number of negative images:', neg_count)
