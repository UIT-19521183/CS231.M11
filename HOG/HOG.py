import cv2
import numpy as np

# Tính độ lớn và hướng của gradient cho toàn bộ ảnh img
def get_gradient(img):
  Gx = cv2.Sobel(img, cv2.CV_32F, dx=0, dy=1, ksize=3)
  Gy = cv2.Sobel(img, cv2.CV_32F, dx=1, dy=0, ksize=3)
  Gx, Gy = np.array(Gx), np.array(Gy)
  G, theta = cv2.cartToPolar(Gx, Gy, angleInDegrees=True) # các góc từ 0 - 360 độ
  return G, theta

# chia độ lớn gradient cho 2 bin liền trước và liền sau theo tỉ lệ
def mapping_magnitude(magnitude, orientation, pre_angle, next_angle):
  '''
  magnitude: độ lớn của vector gradient tại một điểm trong ảnh
  orientation: hướng của vector gradient tại một điểm trong ảnh
  pre_angle: hướng của bin liền trước (do orientation nằm giữa 2 bin)
  next_angle: hướng của bin liền sau
  '''
  pre_magnitude = (next_angle - orientation) * magnitude / (next_angle - pre_angle)
  next_magnitude = (orientation - pre_angle) * magnitude / (next_angle - pre_angle)
  return pre_magnitude, next_magnitude

# Tính HOG của mỗi cell, trả về 9-bin histogram của cell đó
def hog_of_cell(cell_size, G_cell, theta_cell):
  '''
  cell_size: kích thước cell. VD: (8, 8)
  G_cell: gradient magnitude của cell
  theta_cell: gradient orientation của cell
  '''
  w, h = cell_size[0], cell_size[1]
  num_of_bins = 9
  bins = [0]*num_of_bins # khởi tạo 9 bins rỗng
  degrees_of_bins = 180//num_of_bins # khoảng cách độ giữa 2 bin kế nhau, ở đây degrees_of_bins = 20

  # Duyệt qua từng điểm ảnh trong cell
  for i in range(w):
    for j in range(h):
      # lấy ra hướng và độ lớn tại điểm đó
      orientation = theta_cell[i][j] % 180 # chia 180 lấy dư để hướng từ 0 - 180 độ (do là unsigned orientation)
      magnitude = G_cell[i][j]

      at_bin = int(orientation/degrees_of_bins)
      if at_bin == 9:
        at_bin = 0
        
      if orientation/degrees_of_bins == at_bin: # Nếu hướng của gradient trúng một trong 9 bin thì cộng thêm độ lớn của gradient vào bin đó
        bins[at_bin] += magnitude 
      else: # Nếu hướng ở giữa 1 trong 2 bin nào đó, chia độ lớn gradient cho 2 bin theo tỉ lệ (hướng gần bin nào hơn thì độ lớn gradient cho bin đó nhiều hơn)
        pre_bin, next_bin = at_bin, at_bin + 1
        pre_angle, next_angle = pre_bin*degrees_of_bins, next_bin*degrees_of_bins
        pre_magnitude, next_magnitude = mapping_magnitude(magnitude, orientation, pre_angle, next_angle)
        if next_bin == 9:
          next_bin = 0
        bins[pre_bin] += pre_magnitude
        bins[next_bin] += next_magnitude
  return bins

# Chuẩn hóa HOG của tất cá các cell
def normalize_block(hog, block_size):
  '''
  hog: HOG của tất cả các cell trong ảnh, mỗi cell là 9-bin histogram (mảng 9 phần tử)
  block_size: kích thước block (theo đơn vị cell) để chuẩn hóa theo từng block. 
              VD: (2 x 2) tức mỗi block gồm 2 cell x 2 cell
  '''
  hog_feature_vector=[]
  hog_row, hog_col, bins = hog.shape
  block_row, block_col = block_size

  for i in range(0, hog_row - block_row + 1, 1):
    hog_feature_row = []
    for j in range(0, hog_col - block_col + 1, 1):
      block = hog[i : i + block_row, j : j + block_col] 
      block = block.flatten() # flatten từng block, mỗi block là 1 vector 36 phần tử (do mỗi cell là mảng 9 phần tử)
      
      normalized_block = block
      normalized = np.linalg.norm(block) # tính độ lớn vector của block (L2 norm)
      if normalized != 0: # nếu độ lớn của vector != 0 thì chia mọi giá trị của block cho độ lớn để chuẩn hóa
        normalized_block /= normalized
      hog_feature_row.append(normalized_block)
    hog_feature_vector.append(hog_feature_row) # Thêm vector 1x36 đã chuẩn hóa vào HOG feature vector

  hog_feature_vector=np.array(hog_feature_vector)
  # mỗi ảnh 64 x 128 có 8 x 16 cell, tức 7 x 15 = 105 block. 
  # mỗi block là vector 36 phần tử => HOG feature vector có 36 x 105 = 3780 phàn tử
  return hog_feature_vector

def resize_closest(gray, cell_size):
  w = int(gray.shape[1]/cell_size[1])*cell_size[1]
  h = int(gray.shape[0]/cell_size[0])*cell_size[0]
  if w < 64:
    gray = cv2.resize(gray, (64, h))
  elif h < 128:
    gray = cv2.resize(gray, (w, 128))
  else:
    gray = cv2.resize(gray, (w, h))
  return gray

def extract_hog_feature_vector(gray, cell_size, block_size, resize = True, flatten = True):
  '''
  gray: ảnh mức xám cần trích xuất đặc trưng HOG
  cell_size: kích thước cell. VD: (8 x 8)
  block_size: kích thước block tính theo đơn vị cell. VD: (2 x 2) tức mỗi block gồm 2 cell x 2 cell
  '''
  if resize:
    if gray.shape[0] != 128 or gray.shape[1] != 64:
      gray = cv2.resize(gray, (64, 128))
  else:
    gray = resize_closest(gray, cell_size)

  row, col = gray.shape
  cell_row, cell_col = cell_size
  #Xử lý trường hợp rol, cow k chia hết cho cell_size
  row=(row//cell_row)*cell_row
  col=(col//cell_col)*cell_col

  G, theta = get_gradient(gray) # Tính độ lớn và hướng của gradient cho ảnh mức xám
  hog = []
  for i in range(0, row, cell_row): # Duyệt qua từng cell và tính hog của từng cell
    hog_row = []
    for j in range(0, col, cell_col):
      G_cell = G[i : i + cell_row, j : j + cell_col]
      theta_cell = theta[i : i + cell_row, j : j + cell_col]
      hog_cell = hog_of_cell(cell_size, G_cell, theta_cell)
      hog_row.append(hog_cell)
    hog.append(hog_row)

  hog = np.array(hog)
  hog_feature_vector = normalize_block(hog, block_size) # chuẩn hóa hog của từng cell ra HOG feature vector kích thước 1 x 3780
  if flatten:
    hog_feature_vector = hog_feature_vector.flatten()
    return hog_feature_vector
  else:
    return hog_feature_vector, gray.shape

# Tham khảo notebook bạn Nguyễn Lâm Thảo Vy: https://colab.research.google.com/drive/1bSEOYGFWYkh476X1YR6rLlLZnx6Jj77O?usp=sharing
