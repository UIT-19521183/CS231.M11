import os
import cv2
import numpy as np
from sklearn.utils import shuffle
import joblib
import argparse
from function import *

parser = argparse.ArgumentParser(description='Parse Training Directory')
parser.add_argument('--path', help='Path to directory contraining testing images')

args = parser.parse_args()
test_dir = args.path
pos_dir_test = os.path.join(train_dir, 'pos')
neg_dir_test = os.path.join(train_dir, 'neg')

pos_features, neg_features, pos_count_test, neg_count_test = read_images_test(pos_dir_test, neg_dir_test)
pos_features, neg_features = np.array(pos_features), np.array(neg_features)

print('Number of positive images:', pos_count_test)
print('Number of negative images:', neg_count_test)