from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage import data, io
from skimage.color import label2rgb
import skimage
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import scipy.misc
import glob

# settings for LBP
radius = 3 #3×3的图像块
n_points = 8 * radius #中心图像周围8图像

def LBP(path_file):
    #path_file = glob.glob('dataset/train/F/*.png')  # 获取当前文件夹下个数
   # print(path_file[i])
    image = cv2.imread(path_file)
    #graphname = 'dataset/pictures/train/F/'+str(i+1)+'.png'
    graphname = path_file
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)#转化为灰度图像
    lbp = local_binary_pattern(image, n_points, radius)#进行LBP特征提取
    cv2.imwrite(graphname, lbp)
    #print(graphname)
    cv2.waitKey(0)