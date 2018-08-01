from PIL import Image
import numpy as np
import os
import matplotlib.pylab as plt

save_dir = "I:\ICPR_task1_test_20180514\icpr_mtwi_task1\\testimage" #保存图片路径
Img_dataset_dir = "I:\ICPR_task1_test_20180514\icpr_mtwi_task1\\test_line_image" #原图片路径
Image_list = os.listdir(Img_dataset_dir)
# print(Image_list[-1])
for i in Image_list:
    name = i
    img = Image.open(os.path.join(Img_dataset_dir,name)).convert('RGB')
    img1 = np.array(img)
    img_new = Image.fromarray(img1)
    # plt.imshow(img)
    # plt.show()  # 看图
    if img_new.size[0] >=1.2*img_new.size[1]:  # (0为宽，1位高)横的
        # 对img_new做resize
        p = img_new.size[1] / 31  # 高除以31
        if p == 0:
            continue
        new_height = int(img_new.size[1] / p)  # 新高
        new_width = int(img_new.size[0] / p)  # 新宽
        # 成比例放大缩小，高转换成31
        new_0 = img_new.resize((new_width, new_height))  # resize
        # plt.imshow(new_0)
        # plt.show()  # 看图
        print(str(save_dir+"\\"+name+'\n'))
        try:
            new_0.save(os.path.join(save_dir,name))
            f = open(os.path.join(save_dir,'log.txt'),"a")
            # 打开目录下的log.txt文件
            f.write(save_dir+"\\"+name+'\n')
        except:
            continue
    else:  # 竖的图片
        p = img_new.size[0] / 31
        if p == 0:
            continue
        new_height = int(img_new.size[1] / p)
        new_width = int(img_new.size[0] / p)
        new_1 = img_new.resize((new_width, new_height))
        try:
            new_1.save(os.path.join(save_dir, name))  # 保存
            f = open(os.path.join(save_dir, 'log.txt'),"a")
            # 打开目录下的log.txt文件
            f.write(str(save_dir + "\\"+name +'\n'))  # 追加形式写入
        except:
            continue
    f.close()

