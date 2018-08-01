# -*- coding: utf-8 -*-

"""
usage: change your own data dir and it works
"""

import numpy as np
import os
from collections import namedtuple
from PIL import Image
import cv2
import matplotlib.pylab as plt
import codecs


Img_dataset_dir = '..\data\images\\raw\img'#train 的未剪切图片路径
Label_dataset_dir = '..\data\images\\ctxt' #对应的txt文件路径
crop_dataset_dir_horiz = '..\data\images\cut\\horiz' #剪切resize后的横图片保存在这里
crop_dataset_dir_vert = '..\data\images\cut\\vert' #竖

Image_list = os.listdir(Img_dataset_dir)
Label_list = os.listdir(Label_dataset_dir)
#获取目录下的文件名列表

#文本文件处理
def get_txt_label(label_path):
    coordinates = []
    labels = []
    #打开label_path，读取为f，按行读取逗号分割
    with open (label_path,encoding='utf-8')  as f:
        for line in f.readlines():
            coordinate = line.split(',')[0:8] #点集
            label = line.split(',')[-1].strip() #匹配的文字信息
            coordinates.append(coordinate) #将各信息存在list中
            labels.append(label)

    return coordinates,labels

#将各点转换成原点坐标系（0,0）开始
def transform(x1,y1,x2,y2,x3,y3,x4,y4):
    height1 = np.sqrt((x1-x4)**2 + (y4-y1)**2)
    height2 = np.sqrt((x3-x2)**2 + (y3-y2)**2)
    h = max(height1,height2)
    #计算长度并求其中的最大值（做矩形）

    width1 = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    width2 = np.sqrt((x3-x4)**2 + (y3-y4)**2)
    w = max(width1,width2)
    #宽

    Pts = np.float32(np.array([[0,0],[w,0],[w,h],[0,h]]))#时针 #逆
    return Pts,w,h



for i in range(len(Image_list)):#不直接for i in Image_list：是为了方便索引txt文件
    img = Image.open(os.path.join(Img_dataset_dir,Image_list[i])).convert('RGB')
    #以rgb方式打开img

    coordinates,labels = get_txt_label(os.path.join(Label_dataset_dir,Label_list[i]))
    #获取点集，和labels
    coordinates = np.array(coordinates) #将list转化为数组
#    print(len(coordinates))
#    print(len(labels))
#    print(Image_list[i])
    for j in range(coordinates.shape[0]):
        coord= namedtuple('coord',['x1','y1','x2','y2','x3','y3','x4','y4'])
        coordinate = coord(coordinates[j][0],coordinates[j][1],coordinates[j][2],coordinates[j][3],coordinates[j][4],coordinates[j][5],coordinates[j][6],coordinates[j][7])
        label =labels[j]
        if label == str('###'):
            pass
#            new.save(os.path.join(crop_no_use_dir,name))
#         if label==str('"###"'):
#             pass
        else:
            X = list(map(float,[coordinate.x1,coordinate.x2,coordinate.x3,coordinate.x4]))
            Y = list(map(float,[coordinate.y1,coordinate.y2,coordinate.y3,coordinate.y4]))

            Xmin = min(X)
            Xmax = max(X)
            Ymin = min(Y)
            Ymax = max(Y)

            Pts1 = np.float32(np.array([[X[0],Y[0]],[X[1],Y[1]],[X[2],Y[2]],[X[3],Y[3]]]))
            Pts2,W,H = transform(X[0],Y[0],X[1],Y[1],X[2],Y[2],X[3],Y[3])
            #原集合pts1和转换后的pts2


            M = cv2.getPerspectiveTransform(Pts1,Pts2) #透视转换(点阵转换)
            img1 = np.array(img) #将img转换成数组
            Dst = cv2.warpPerspective(img1,M,(int(W),int(H))) #切割
            img_new = Image.fromarray(Dst) #新图片

            # plt.imshow(Dst)
            # plt.show() #看图

            name = str(str(i)+'_'+str(j)+'.jpg') #利用循环的索引号编名
            if img_new.size[0]>=1.2*img_new.size[1]:#(0为宽，1位高)横的
                # 对img_new做resize
                p=img_new.size[1]/31 #高除以31
                if p==0:
                    continue
                new_height=int(img_new.size[1]/p) #新高
                new_width=int(img_new.size[0]/p) #新宽
                #成比例放大缩小，高转换成31
                new_0 = img_new.resize((new_width,new_height)) #resize
                try:
                    new_0.save(os.path.join(crop_dataset_dir_horiz, name)) #保存
                    f = codecs.open(os.path.join(crop_dataset_dir_horiz,'label.txt'),"a",encoding='utf-8')
                    #打开crop_dataset_dir_horiz目录下的label_horiz.txt文件
                    f.write(str(crop_dataset_dir_horiz+"\\"+name+' '+label+'\n')) #追加形式写入
                    f1 = codecs.open(os.path.join(crop_dataset_dir_horiz,'label_ciku.txt'),"a",encoding='utf-8')
                    f1.write(label+'\n')
                    #打开label_ciku.txt并追加写入label
                except:
                    continue
            else:#竖的图片
                p = img_new.size[0]/31
                if p==0:
                    continue
                new_height=int(img_new.size[1]/p)
                new_width=int(img_new.size[0]/p)
                new_1=img_new.resize((new_width,new_height))
                try:
                    new_1.save(os.path.join(crop_dataset_dir_vert, name))
                    f = codecs.open(os.path.join(crop_dataset_dir_vert,'label.txt'),"a",encoding='utf-8')
                    f.write(str(crop_dataset_dir_vert+"\\"+name+' '+label+'\n'))
                    f1 = codecs.open(os.path.join(crop_dataset_dir_vert,'label_ciku.txt'),"a",encoding='utf-8')
                    f1.write(label+'\n')
                except:
                    continue
            f.close()
            f1.close()
#    plt.close()