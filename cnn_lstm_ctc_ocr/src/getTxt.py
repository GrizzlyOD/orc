

import os
import path
import glob
from PIL import Image  
from PIL import ImageDraw



# ground truth directory
gt_text_dir = "..\data\images\\raw\\txt"


# revised txt directory
revised_text_dir = "..\data\images\\ctxt"

txtdirs = os.listdir(gt_text_dir)



for txt_name in txtdirs:

    # open the ground truth text file
    bf = open(os.path.join(gt_text_dir, txt_name),encoding='utf-8').read().splitlines()
    #打开gt_text_dir目录下的img_gt_text_name文件（存储的原txt文件），读取的方式

    f_revised = open(os.path.join(revised_text_dir, txt_name),mode = 'w',encoding='utf-8')
    #写方式打开保存文件下txt

    for idx in bf:
        rect = []
        spt = idx.split(',')
        for i in range(8):
            rect.append(float(spt[i])) #将点集添加到list里

        #clockwise adjustment (逆)
        #(x2-x1)*(y3-y2)-(y2-y1)*(x3-x2)
        clockwiseFlag = (rect[2]-rect[0])*(rect[5]-rect[3])-(rect[3]-rect[1])*(rect[4]-rect[2])
        if clockwiseFlag < 0:
            tmp_x2 = rect[2]
            tmp_y2 = rect[3]
            rect[2] = rect[6]
            rect[3] = rect[7]
            rect[6] = tmp_x2
            rect[7] = tmp_y2
            tmp_x2_s = spt[2]
            tmp_y2_s = spt[3]
            spt[2] = spt[6]
            spt[3] = spt[7]
            spt[6] = tmp_x2_s
            spt[7] = tmp_y2_s

        #write txt
        sep= ','
        s1 = sep.join(spt)
        s1= s1 + '\n'
        f_revised.write(s1)
        # draw the polygon with (x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4)
