
from PIL import Image
import numpy as np
import os
import matplotlib.pylab as plt

save_dir = "I:\ICPR_task1_test_20180514\icpr_mtwi_task1\save"
img_dir = "I:\ICPR_task1_test_20180514\icpr_mtwi_task1\\testimage"
image_name = os.listdir(img_dir)
image_name.sort(key=lambda x: int(x[5:-4]))
for i in  image_name:
    name = i
    img = Image.open(os.path.join(img_dir,name)).convert('RGB')
    img1 = np.array(img)
    img_new = Image.fromarray(img1)
    if img_new.size[0] >=img_new.size[1]:  # (0为宽，1位高)横的
        # 对img_new做resize
        if img_new.size[1] == 31:
            continue
        new_0 = img_new.resize((img_new.size[0],31))  # resize
        print(str(save_dir+"\\"+name+'\n'))
        try:
            new_0.save(os.path.join(save_dir,name))
        except:
            continue
    else:  # 竖的图片
        if img_new.size[0] == 31:
            continue
        new_1 = img_new.resize((31,img_new.size[1]))
        try:
            new_1.save(os.path.join(save_dir, name))  # 保存
        except:
            continue