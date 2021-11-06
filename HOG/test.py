import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
import argparse
from HOG import *

parser = argparse.ArgumentParser(description='Parse Training Directory')
parser.add_argument('--path', help='Path to directory contraining training images')

# Thư mục test gồm: thư mục pos chứa các ảnh có người và thư mục neg chứa các ảnh không có người
args = parser.parse_args()
test_dir = args.path
pos_dir = os.path.join(test_dir, 'pos')
neg_dir = os.path.join(test_dir, 'neg')

# Cắt ra các ảnh nhỏ tại vị trí pos
def crop_image(img, pos):
    x0, y0, w, h = map(int, pos)
    crop = img[y0:y0+h, x0:x0+w]
    return crop

# Tạo n cửa sổ ngẫu nhiên kích thước 64x128
def random_windows(img, n):
  max_y, max_x, _ = img.shape
  windows = []
  for i in range (n):
    x0 = np.random.randint(0, max_x-64)
    y0 = np.random.randint(0, max_y-128)
    windows.append([x0, y0, 64, 128])
  return windows

# Đọc các ảnh chứa người và không chứa người, sau đó trích xuất đặc trưng HOG để train
def read_images(pos_dir, neg_dir):
  '''
  pos_dir: đường dẫn đến thư mục chứa các ảnh có người đi bộ
  neg_dir: đường dẫn đến thư mục chứa các ảnh không có người đi bộ
          (trong pos_dir và neg_dir có thư mục images chứa các ảnh và thư mục annotations chứa các labels)
  '''

  pos_images_dir = pos_dir + '/images'
  pos_annotations_dir = pos_dir + '/annotations'
  neg_images_dir = neg_dir + '/images'
  neg_annotations_dir = neg_dir + '/annotations'

  pos_features, neg_features = [], []
  pos_count, neg_count = 0, 0

  # Đọc từng ảnh từ thư mục images và file json tương ứng trong thư mục annotations
  for pos_image in os.listdir(pos_images_dir):
    pos_annotation = open(pos_annotations_dir+'/'+pos_image[:-3]+'json')
    objects = json.load(pos_annotation)
    img = cv2.imread(pos_images_dir+'/'+pos_image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Với từng đối tượng được gắn nhãn tron file json
    for obj in objects:
      if obj['lbl'] == 'person': # nếu đối tượng đó là người thì cắt ra và trích xuất đặc trưng HOG
        cropped = crop_image(gray, obj['pos'])
        cropped = cv2.resize(cropped,(64, 128))
        feature = extract_hog_feature_vector(cropped, (8, 8), (2, 2))
        pos_features.append(feature)
        pos_count += 1 

  for neg_image in os.listdir(neg_images_dir):
    img = cv2.imread(neg_images_dir+'/'+neg_image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Với mỗi ảnh không có người, tạo thêm 1 - 5 cửa sổ ngẫu nhiên và cắt ảnh từ cửa sổ
    windows = random_windows(img, np.random.randint(1, 6))
    for win in windows:
      cropped = crop_image(gray, win) # cắt ảnh ra từ các cửa sổ ngẫu nhiên và trích xuất đặc trưng HOG
      feature = extract_hog_feature_vector(cropped, (8, 8), (2, 2))
      neg_features.append(feature)
      neg_count += 1 

  
  return pos_features, neg_features, pos_count, neg_count

# Đọc các ảnh từ pos_dir và neg_dir và trích xuất đặc trưng HOG
pos_features, neg_features, pos_count, neg_count = read_images(pos_dir, neg_dir)
pos_features, neg_features = np.array(pos_features), np.array(neg_features)

print('Number of positive images:', pos_count)
print('Number of negative images:', neg_count)

# Predict các samples
clf = joblib.load('pedestrian_final.pkl')
pos_result = clf.predict(pos_features)
neg_result = clf.predict(neg_features)

# Tính accuracy, precision, recall và f1 score
true_positives = cv2.countNonZero(pos_result)
false_negatives = pos_result.shape[0] - true_positives

false_positives = cv2.countNonZero(neg_result)
true_negatives = neg_result.shape[0] - false_positives

accuracy = float(true_positives + true_negatives)/(true_positives + true_negatives + false_positives + false_negatives)
precision = float(true_positives) / (true_positives + false_positives)
recall = float(true_positives) / (true_positives + false_negatives)
f1 = 2*precision * recall / (precision + recall)

print(f'Accuracy: {accuracy*100} %')
print(f'Presision: {precision*100} %')
print(f'Recall: {recall*100} %')
print(f'F1 score: {f1*100} %')
