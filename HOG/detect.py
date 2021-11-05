import os
import cv2
import numpy as np
from sklearn.externals import joblib
from HOG import *
import argparse

parser = argparse.ArgumentParser(description='Parse Detection Directory')
parser.add_argument('--path', help='Path to directory containing images for detection')
detect_dir = args.path

# Tạo pyramid chứa các ảnh được resize với các tỉ lệ khác nhau
def pyramid(img, scales):
  scaled_imgs = []
  width, height = img.shape[1], img.shape[0]
  for scale in scales:
    new_width = int(img.shape[1] * scale)
    new_height = int(img.shape[0] * scale)
    resized = cv2.resize(img, (new_width, new_height), interpolation = cv2.INTER_AREA)
    scaled_imgs.append((resized, scale))
  return scaled_imgs


clf = joblib.load("pedestrian_final.pkl")
rects = []

window_size = (64, 128) # kích thước cửa số và kích thước của mỗi bước trượt cửa số (w x h)
step_size = (8, 8)

visual = resize_closest(img, (8, 8))

for image in os.listdir(detect_dir):
  img = cv2.imread(os.path.join(detect_dir, image))
  gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  
  scales = np.arange(0.25, 2.01, 0.25) # Tạo pyramid chứa các ảnh resize với tỉ lệ 0.25, 0.5, 0.75, 1 (ảnh gốc), 1.25, 1.5, 1.75, 2 
  scaled_grays = pyramid(gray_img, scales)
  
  for gray, scale in scaled_grays: # Duyệt qua từng ảnh mức xám ở các tỉ lệ khác nhau
    feature, new_shape = extract_hog_feature_vector(gray, (8, 8), (2, 2), resize = False, flatten = False)
    max_x, max_y =  int((new_shape[1] - window_size[0])/8),  int((new_shape[0] - window_size[1])/8)
    step_x, step_y = int(step_size[0]/8), int(step_size[1]/8)
    if step_x == 0:
      step_x = 1
    if step_y == 0:
      step_y = 1
    for x in range(0, max_x + 1, step_x): # Trượt cửa sổ qua toàn bộ ảnh từ trái qua phải, trên xuống dưới
      for y in range(0, max_y + 1, step_y):
        #print(x, y)
        #cropped = gray[i : i + window_size[0], j : j + window_size[1]] # Cắt ảnh từ cửa số và predict
        feature_cropped = feature[y : y + int(window_size[1]/8) - 1, x : x + int(window_size[0]/8) - 1]
        #print(feature_cropped.shape)
        feature_cropped = feature_cropped.flatten()
        
        if feature_cropped.shape != (3780, ):
          cv2.rectangle(gray, (x * 8, y*8), (x * 8 + window_size[0], y * 8 + window_size[1]), (0, 0, 255), 2)

        label = clf.predict([feature_cropped])

        if int(label[0]) == 1: # Nếu kết qủa predict là 1, tức là có người đi bộ trong cửa số đang xét
          confidence = clf.decision_function([feature_cropped])
          #min_x, min_y = int(j/scale), int(i/scale)
          #max_x, max_y = min_x + window_size[1], min_y + window_size[0]
          rects.append([int(x * 8/scale), int(y * 8/scale), 64, 128, confidence])
          cv2.rectangle(visual, (int(x * 8/scale), int(y * 8/scale)), (int(x * 8/scale + window_size[0]*scale), int(y * 8/scale + window_size[1]*scale)), (0, 0, 255), 2) # vẽ bounding box vào ảnh kết quả (chưa dùng non max suppression)
          print([x, y, 64, 128, confidence])
          #cv2.rectangle(gray, (x * 8, y * 8), (x * 8 + window_size[0], y * 8 + window_size[1]), (0, 0, 255), 2)
    #cv2_imshow(visual)
