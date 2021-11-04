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
parser.add_argument('--path', help='Path to training directory')

# Thư mục train gồm: thư mục pos chứa các ảnh có người và thư mục neg chứa các ảnh không có người
args = parser.parse_args()
train_dir = args.path
pos_dir = os.path.join(train_dir, 'pos')
neg_dir = os.path.join(train_dir, 'neg')

# Cắt ra các ảnh nhỏ chứa người
def crop_image(img, pos):
    x0, y0, w, h = map(int, pos)
    crop = img[y0:y0+h, x0:x0+w, :]
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

# Đọc các ảnh chứa người và extract HOG feature từng ảnh để train
def read_imgs(pos_dir):
  '''
  pos_dir: đường dẫn đến thư mục chứa các ảnh có người đi bộ
                (trong đó có thư mục images chứa các ảnh và thư mục annotations chứa các labels)
  '''
  X, y = [], []
  images_path = pos_dir + '/images'
  annotations_path = pos_dir + '/annotations'

  # Đọc từng ảnh từ thư mục images và file json tương ứng trong thư mục annotations
  for image_file in os.listdir(images_path):
    annotation_file = open(annotations_path+'/'+image_file[:-3]+'json')
    objects = json.load(annotation_file)
    img = cv2.imread(images_path+'/'+image_file)

    # Với từng đối tượng được gắn nhãn tron file json
    for obj in objects:
      if obj['lbl'] == 'person': # nếu đối tượng đó là người thì cắt ra và trích xuất đặc trưng HOG
        cropped = crop_image(img, obj['pos'])
        cropped = cv2.resize(cropped,(64, 128))
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        feature = extract_hog_feature_vector(gray, (8, 8), (2, 2))
        X.append(feature) # Thêm đặc trưng và gắn nhãn là 1 (tức là đặc trưng HOG của người đi bộ)
        y.append(1)

    # Mỗi ảnh đều tạo thêm 10 cửa số ngẫu nhiên kích thước 64 x 128
    windows = random_windows(img, 10)
    for win in windows:
      cropped = crop_image(img, win) # cắt ảnh ra từ các cửa sổ ngẫu nhiên và trích xuất đặc trưng HOG
      gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
      feature = extract_hog_feature_vector(gray, (8, 8), (2, 2))
      X.append(feature)
      y.append(0) # Thêm đặc trưng và gắn nhãn là 0 (do không phải HOG của người đi bộ)
  return X, y

# Đọc các ảnh từ thư mục pos
X, y = read_imgs(pos_dir)
X, y = np.array(X), np.array(y)
X, y = shuffle(X, y, random_state=0)

clf = LinearSVC(C=0.01, max_iter=1000, class_weight='balanced', verbose = 1)
clf.fit(X, y) 
joblib.dump(clf, 'pedestrian_detection.pkl')