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

# Thư mục train gồm: thư mục pos chứa các ảnh có người và thư mục neg chứa các ảnh không có người
# train
# |---pos
# |   |---images
# |   |---annotations
# |
# |---neg
#     |---images
#     |---annotations (không có cũng được)

args = parser.parse_args()
train_dir = args.path
pos_dir = os.path.join(train_dir, 'pos')
neg_dir = os.path.join(train_dir, 'neg')

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

  X, y = [], []
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
        X.append(feature) # Thêm đặc trưng và gắn nhãn là 1 (tức là đặc trưng HOG của người đi bộ)
        y.append(1)
        pos_count += 1 

  for neg_image in os.listdir(neg_images_dir):
    img = cv2.imread(neg_images_dir+'/'+neg_image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Mỗi ảnh không có người đều tạo 10 cửa số ngẫu nhiên kích thước 64 x 128
    windows = random_windows(img, 10)
    for win in windows:
      cropped = crop_image(gray, win) # cắt ảnh ra từ các cửa sổ ngẫu nhiên và trích xuất đặc trưng HOG
      feature = extract_hog_feature_vector(cropped, (8, 8), (2, 2))
      X.append(feature)
      y.append(0) # Thêm đặc trưng và gắn nhãn là 0 (do không phải HOG của người đi bộ)
      neg_count += 1 

  
  return X, y, pos_count, neg_count

def sliding_windows(neq_img, window_size, step_size):
  '''
  neq_img: ảnh không có người
  window_size: kích thước cửa số (width x height). VD: (64 x 128)
  step_size: kích thước của mỗi bước trượt cửa số (step_width x step_height)
             VD: (64 x 128) thì cửa sổ sau trượt đi 64 pixel sang phải hoặc 128 pixel xuống dưới so với cửa sổ trước

  '''
  max_x, max_y = neq_img.shape[1] - window_size[0] + 1, neq_img.shape[0] - window_size[1] + 1
  windows = []
  for x in range(0, max_x, step_size[1]):
    for y in range(0, max_y, step_size[0]):
      windows.append([x, y, window_size[0], window_size[1]])
  return windows

# Đọc các ảnh không chứa người
def read_hard_negative_images(neg_dir, hard_neg_limit, clf):
  neg_images_dir = neg_dir + '/images'
  X, y = [], []

  windows = []; shape = (0, 0)
  hard_neg_count = 0

  for neg_image in os.listdir(neg_images_dir):
    img = cv2.imread(neg_images_dir+'/'+neg_image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Nếu shape của ảnh thay đổi lưu shape mới của ảnh và tạo các cửa số trên ảnh dựa trên shape mới
    if shape != img.shape:
      shape = img.shape
      windows = sliding_windows(img, (64, 128), (64, 128))
    for win in windows:
      cropped = crop_image(gray, win) # cắt ảnh ra từ cửa sổ và trích xuất đặc trưng HOG (ảnh này chắc chắn không có người)
      feature = extract_hog_feature_vector(cropped, (8, 8), (2, 2))
      if clf.predict([feature]) == 1: # nhưng nếu model đã train predict là có người (tức là ảnh này khó)
        X.append(feature)
        y.append(0) # thêm đặc trưng và gắn nhãn là 0 để sau đó train thêm
        hard_neg_count += 1
        if hard_neg_count == hard_neg_limit:
          break

  return X, y, hard_neg_count

# Đọc các ảnh từ pos_dir và neg_dir và trích xuất đặc trưng HOG
X, y, pos_count, neg_count = read_images(pos_dir, neg_dir)
X, y = np.array(X), np.array(y)
X, y = shuffle(X, y, random_state=0)

print('Training started')
print('Number of positive images:', pos_count)
print('Number of negative images:', neg_count)

clf1 = LinearSVC(C=0.01, max_iter=1000, class_weight='balanced', verbose = 1)
clf1.fit(X, y) 
joblib.dump(clf1, 'pedestrian.pkl')

# Trích xuất thêm đặc trưng của các ảnh không chứa người khó, sau đó hợp với các đặc trưng cũ và train lại
hard_neg_limit = 10000
X_hard, y_hard, hard_neg_count = read_hard_negative_images(neg_dir, hard_neg_limit, clf1)
X_hard = np.concatenate((X_hard, X), axis = 0)
y_hard = np.concatenate((y_hard, y), axis = 0)

print('Continue training hard negative images')
print('Number of negative images:', hard_neg_count)

clf2 = LinearSVC(C=0.01, max_iter=1000, class_weight='balanced', verbose = 1)
clf2.fit(X_hard, y_hard)
joblib.dump(clf2, 'pedestrian_final.pkl')

print('Training done')
