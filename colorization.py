import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from scipy import io
from scipy import sparse
from scipy.sparse import linalg
import colorsys
import os

# 重载图片格式转换函数
def yiq_to_rgb(y, i, q):
    r = y + 0.9468822170900693 * i + 0.6235565819861433 * q
    g = y - 0.27478764629897834 * i - 0.6356910791873801 * q
    b = y - 1.1085450346420322 * i + 1.7090069284064666 * q
    r[r < 0] = 0
    r[r > 1] = 1
    g[g < 0] = 0
    g[g > 1] = 1
    b[b < 0] = 0
    b[b > 1] = 1
    return (r, g, b)

original_img_filename = input('original image file name : ')
marked_img_filename = input('marked image file name : ')

path = os.path.dirname(os.path.realpath(__file__))
original_img = misc.imread(os.path.join(path, original_img_filename))
marked_img = misc.imread(os.path.join(path, marked_img_filename))

# 归一化
original_img = original_img.astype(float)/255.0
marked_img = marked_img.astype(float)/255.0

# 判断初始是否有染色
isColored = abs(original_img - marked_img).sum(2) > 0.01

# YUV三通道(YIQ与YUV只是颜色旋转)
YUV = np.zeros(original_img.shape)
(YUV[:,:,0], _, _) = colorsys.rgb_to_yiq(original_img[:,:,0], original_img[:,:,1], original_img[:,:,2])
(_, YUV[:,:,1], YUV[:,:,2]) = colorsys.rgb_to_yiq(marked_img[:,:,0], marked_img[:,:,1], marked_img[:,:,2])

height = YUV.shape[0]
width = YUV.shape[1]
image_size = height * width
indices_matrix = np.arange(image_size).reshape(height,width,order='F').copy()

wd_size = 1
wd_pixel_num = (2 * wd_size + 1)**2
max_nr = image_size * wd_pixel_num

# W[row_inds[k], col_inds[k]] = vals[k]
# 为中心的像素的index
row_inds = np.zeros(max_nr, dtype=np.int64)
# 中心以及相邻窗口的像素index
col_inds = np.zeros(max_nr, dtype=np.int64)
# 系数：Wrs
vals = np.zeros(max_nr)

pos = 0
pixel_num = 0

for j in range(width):
    for i in range(height):
        
        if (not isColored[i,j]):
            window_index = 0
            # 当前窗口的Y值
            window_vals = np.zeros(wd_pixel_num)
            
            for ii in range(max(0, i - wd_size), min(i + wd_size + 1, height)):
                for jj in range(max(0, j - wd_size), min(j + wd_size + 1, width)):
                    
                    if (ii != i or jj != j):
                        row_inds[pos] = pixel_num
                        col_inds[pos] = indices_matrix[ii,jj]
                        window_vals[window_index] = YUV[ii,jj,0]
                        pos += 1
                        window_index += 1
            
            center = YUV[i,j,0].copy()
            window_vals[window_index] = center

            variance = np.mean((window_vals[0:window_index+1] - np.mean(window_vals[0:window_index+1]))**2)
            sigma = variance * 2
            if (sigma < 0.000001):
                sigma = 0.000001
            
            # 用weight function (2)
            window_vals[0:window_index] = np.exp( -((window_vals[0:window_index] - center)**2) / sigma )
            # 归一化
            window_vals[0:window_index] = window_vals[0:window_index] / np.sum(window_vals[0:window_index])
            vals[pos-window_index:pos] = -window_vals[0:window_index]
        
        row_inds[pos] = pixel_num
        
        col_inds[pos] = indices_matrix[i,j]
        vals[pos] = 1
        pos += 1
        pixel_num += 1

vals = vals[0:pos]
col_inds = col_inds[0:pos]
row_inds = row_inds[0:pos]

# sparse库中的按行对矩阵进行压缩
A = sparse.csr_matrix((vals, (row_inds, col_inds)), (pixel_num, image_size))
b = np.zeros((A.shape[0]))

colorized = np.zeros(YUV.shape)
colorized[:,:,0] = YUV[:,:,0]
# 已经被染色部分
colored_inds = np.nonzero(isColored.reshape(image_size, order='F').copy())

for t in [1, 2]:
    curIm = YUV[:,:,t].reshape(image_size, order='F').copy()
    b[colored_inds] = curIm[colored_inds]
    # linalg.spsolve 解线性方程组：Ax = b
    colorized[:,:,t] = linalg.spsolve(A, b).reshape(height, width, order='F')

# 图片格式转换回RGB
(R, G, B) = yiq_to_rgb(colorized[:,:,0],colorized[:,:,1],colorized[:,:,2])
colorizedRGB = np.zeros(colorized.shape)
colorizedRGB[:,:,0] = R
colorizedRGB[:,:,1] = G
colorizedRGB[:,:,2] = B

plt.imshow(colorizedRGB)
plt.show()

misc.imsave(os.path.join(path, 'colorized.bmp'), colorizedRGB, format='bmp')