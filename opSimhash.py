import os, string, shutil,re
import sys
import binascii
from simhash import Simhash
import wordninja
import jieba
import cv2
import numpy as np
import scipy.misc
from PIL import Image
import glob
import array
import base64
import pefile
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy
import os
import binascii
import scipy.misc
from PIL import Image
import glob
from picturesLBP import LBP


def draw(x,y,img):#根据横纵坐标，该点亮度调高16
    #print(x,y)
    #print(y)
    if(img[x,y]+16>=255):
        img[x, y] =255
    else:
        img[x,y]+=16

def show(graphname,img):
    #cv2.imshow('img',img)
    cv2.imwrite(graphname, img)
    LBP(graphname)
    #print(graphname)
    cv2.waitKey(0)



def sim(str,img):#结巴分词，dict是自己做的字典，直接调用simhash库
    if(str!=''):
        #print(str)
        #jieba.load_userdict("./dict.txt")
        a =jieba.cut(str,cut_all=True,HMM=False)
        #print(jieba.lcut(str))
        b=hex(Simhash(a).value)
        b=b[2:18]
        #print(b)
        c=''
        for i in range(len(b)):#simhash值切割为2个8进制数，如不够16位数则末位0补齐
            if(b[i]>'7'):
                c+='1'
            else:
                c+='0'
        if(len(b)<16):
                c+='0'
        #print(c)
        if(len(c)==16):
            d=int(c[0:8],2)
            e=int(c[8:16],2)
            #print(d,e)
            draw(d,e,img)


def is_number(s): #判断是否是纯数字
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


def getPicture(filename,graphname):#输入是反汇编后的.text文件操作码，输出为新的图片
    #filename='G:/mm/F/2F.mm'
    img = np.ones((256, 256), dtype=np.uint8)#新图片固定长宽，默认是全黑图像
    str=''
    f = open(filename, 'rb')  # 读入文件
    ln = os.path.getsize(filename)  # 文件长度（byte）
    width = 1024  # 固定图片宽度
    rem = ln % width  # 计算余出的字节
    a = array.array("B")  # uint8 数组
   # print(ln,filename)
    #print(int(ln/width))
    for b in range(10240):
        #print(b)
        f.seek(width*b,0)
        str=f.read(width)
        sim(str,img)
    show(graphname,img)
    f.close()