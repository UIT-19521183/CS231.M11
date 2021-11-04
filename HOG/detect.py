import os
import cv2
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


clf = joblib.load("pedestrian_detection.pkl")
rects = []

for image in os.listdir(detect_dir):
  
  img = cv2.imread(os.path.join(detect_dir, image))
  gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  visual = img.copy()

  window_size = (128, 64) # kích thước cửa số và kích thước của mỗi bước trượt cửa số qua ảnh
  step_size = (16, 16)

  scales = np.arange(0.5, 1.51, 0.5) # Tạo pyramid chứa các ảnh resize với tỉ lệ 0.5, 1 (ảnh gốc), 1.5 
  scaled_grays = pyramid(gray_img, scales)
  for gray, scale in scaled_grays: # Duyệt qua từng ảnh mức xám ở các tỉ lệ khác nhau   
    max_i, max_j = gray.shape[0] - window_size[0] + 1, gray.shape[1] - window_size[1] + 1
    print('max',  max_i, max_j)
    step_height, step_width = int(step_size[0]*scale), int(step_size[1]*scale)
    for i in range(0, max_i, step_height): # Trượt cưa sổ qua toàn bộ ảnh từ trái qua phải, trên xuống dưới
      for j in range(0, max_j, step_width):
        cropped = gray[i : i + window_size[0], j : j + window_size[1]] # Cắt ảnh từ cửa số và predict
        feature = extract_hog_feature_vector(cropped, (8, 8), (2, 2))
        label = clf.predict([feature])

        if int(label[0]) == 1: # Nếu kết qủa predict là 1, tức là có người đi bộ trong cửa số đang xét
          confidence = clf.decision_function([feature])
          min_x, min_y = int(j/scale), int(i/scale)
          max_x, max_y = min_x + window_size[1], min_y + window_size[0]
          rects.append([min_x, min_y, 64, 128, confidence])
          cv2.rectangle(visual, (min_x, min_y), (max_x, max_y), (0, 0, 255), 2) # vẽ bounding box vào ảnh kết quả (chưa dùng non max suppression)

  