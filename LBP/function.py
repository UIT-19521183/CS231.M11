import cv2
import os
import json
import numpy as np
from LBP import *

def crop_image(img, pos):
    x0, y0, w, h = map(int, pos)
    crop = img[y0:y0+h, x0:x0+w]
    return crop

def random_windows(img, n):
  max_y, max_x, _ = img.shape
  windows = []
  for i in range (n):
    x0 = np.random.randint(0, max_x-64)
    y0 = np.random.randint(0, max_y-128)
    windows.append([x0, y0, 64, 128])
  return windows

def read_images(pos_dir, neg_dir):

  pos_images_dir = pos_dir + '/images'
  pos_annotations_dir = pos_dir + '/annotations'
  neg_images_dir = neg_dir + '/images'
  neg_annotations_dir = neg_dir + '/annotations'

  X, y = [], []
  pos_count, neg_count = 0, 0

  for pos_image in os.listdir(pos_images_dir):
    print (pos_image)
    pos_annotation = open(pos_annotations_dir+'/'+pos_image[:-3]+'json')
    objects = json.load(pos_annotation)
    img = cv2.imread(pos_images_dir+'/'+pos_image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    for obj in objects:
      if obj['lbl'] == 'person': 
        cropped = crop_image(gray, obj['pos'])
        if (cropped.shape[1]>0):
          hist_lbp = calc_lbp(cropped)
          feature_new = hist_lbp.reshape(1,-1)
          X.append(feature_new) 
          y.append(1)
          pos_count += 1 
      
    
  for neg_image in os.listdir(neg_images_dir):
    img = cv2.imread(neg_images_dir+'/'+neg_image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    windows = random_windows(img, 10)
    for win in windows:
      cropped = crop_image(gray, win) 
      feature = calc_lbp(cropped)
      feature1 = feature.reshape(1,-1)
      X.append(feature1)
      y.append(0) 
      neg_count += 1 
  
  return X, y, pos_count, neg_count

def sliding_windows(neq_img, window_size, step_size):
  max_x, max_y = neq_img.shape[1] - window_size[0] + 1, neq_img.shape[0] - window_size[1] + 1
  windows = []
  for x in range(0, max_x, step_size[1]):
    for y in range(0, max_y, step_size[0]):
      windows.append([x, y, window_size[0], window_size[1]])
  return windows
