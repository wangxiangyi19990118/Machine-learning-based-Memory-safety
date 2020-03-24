import os, string, shutil,re
import sys
import binascii
import cv2
import numpy as np
import scipy.misc
from PIL import Image
from opSimhash import getPicture
import glob


path_file=glob.glob('objdump/T/*') #获取当前文件夹下个数
path_number=len(path_file)
print(path_number)

for i in range(path_number): #针对每一个提取出的text文件，进行simhash图片，并保存到文件夹中
    filename = path_file[i]
    graphname = 'dataset/pictures/1/'+str(i) + '.png'

    getPicture(filename,graphname)