import cv2
import json
import numpy as np
from LBP import *


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
      if obj['lbl'] == 'person': 
        cropped = crop_image(gray, obj['pos'])
        hist_lbp = calc_lbp(cropped)
        X.append(hist_lbp) 
        y.append(1)
        pos_count += 1 

  for neg_image in os.listdir(neg_images_dir):
    img = cv2.imread(neg_images_dir+'/'+neg_image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Mỗi ảnh không có người đều tạo 10 cửa số ngẫu nhiên kích thước 64 x 128
    windows = random_windows(img, 10)
    for win in windows:
      cropped = crop_image(gray, win) # cắt ảnh ra từ các cửa sổ ngẫu nhiên và trích xuất đặc trưng HOG
      feature = calc_lbp(cropped)
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
      feature = calc_lbp(cropped)
      feature_new = feature.reshape(1, -1)
      #feature_=np.array([feature])
      #feature = extract_hog_feature_vector(cropped, (8, 8), (2, 2))
      if clf.predict(feature_new) == 1: # nhưng nếu model đã train predict là có người (tức là ảnh này khó)
        X.append(feature)
        y.append(0) # thêm đặc trưng và gắn nhãn là 0 để sau đó train thêm
        hard_neg_count += 1
        if hard_neg_count == hard_neg_limit:
          break

  return X, y, hard_neg_count