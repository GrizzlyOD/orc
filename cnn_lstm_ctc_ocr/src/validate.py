# CNN-LSTM-CTC-OCR
# Copyright (C) 2017, 2018 Jerod Weinman
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import os
import sys

import numpy as np

from PIL import Image

import tensorflow as tf
from tensorflow.contrib import learn

import mjsynth
import model
import denseNet

import gc

from threading import Thread
import time

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('model','../data/model',
                          """Directory for model checkpoints""")
#测试checkpoints的路径
tf.app.flags.DEFINE_string('device','/cpu:0',
                           """Device for graph placement""")

tf.logging.set_verbosity(tf.logging.WARN)

# Non-configurable parameters
mode = learn.ModeKeys.INFER # 'Configure' training mode for dropout layers

def _get_image(filename):
    """Load image data for placement in graph"""
    image = Image.open(filename) #打开图片
    image = np.array(image)
    # in mjsynth, all three channels are the same in these grayscale-cum-RGB data
    image = image[:,:,:1] # so just extract first channel, preserving 3D shape

    return image
#加载图片数据


def _preprocess_image(image):

    # Copied from mjsynth.py. Should be abstracted to a more general module.
    
    # Rescale from uint8([0,255]) to float([-0.5,0.5])
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.subtract(image, 0.5)

    # Pad with copy of first row to expand to 32 pixels height
    first_row = tf.slice(image, [0, 0, 0], [1, -1, -1])
    image = tf.concat([first_row, image], 0)

    return image
#图片预处理

def _get_input():
    """Set up and return image and width placeholder tensors"""

    # Raw image as placeholder to be fed one-by-one by dictionary
    # try:
    #     image = tf.placeholder(tf.uint8, shape=[31, None, 1])
    # except:
    image = tf.placeholder(tf.uint8, shape=[31, None, 1])
    width = tf.placeholder(tf.int32, shape=[]) # for ctc_loss

    return image,width
#返回image和宽度

def _get_output(rnn_logits,sequence_length):
    """Create ops for validation
       predictions: Results of CTC beacm search decoding
    """
    with tf.name_scope("test"):
        predictions,_ = tf.nn.ctc_beam_search_decoder(rnn_logits, 
                                                   sequence_length,
                                                   beam_width=128,
                                                   top_paths=1,
                                                   merge_repeated=True)

    return predictions
#获取预测结果


def _get_session_config():
    """Setup session config to soften device placement"""
    config=tf.ConfigProto(
        allow_soft_placement=True, 
        log_device_placement=False)

    return config
#会话的配置

def _get_checkpoint():
    """Get the checkpoint path from the given model output directory"""
    ckpt = tf.train.get_checkpoint_state(FLAGS.model)

    if ckpt and ckpt.model_checkpoint_path:
        ckpt_path=ckpt.model_checkpoint_path
    else:
        raise RuntimeError('No checkpoint file found')

    return ckpt_path
#通过checkpoints获取ckpt文件路径

def _get_init_trained():
    """Return init function to restore trained model from a given checkpoint"""
    saver_reader = tf.train.Saver(
        tf.get_collection(tf.GraphKeys.GLOBAL_STEP) + 
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    )
    
    init_fn = lambda sess,ckpt_path: saver_reader.restore(sess, ckpt_path)
    return init_fn
#返回初始化功能，用来自checkpoint获得的trained模块

def _get_string(labels):
    """Transform an 1D array of labels into the corresponding character string"""
    string = ''.join([mjsynth.out_charset[c] for c in labels])
    return string
#将一维数组转换成string


def main(argv=None):

    with tf.Graph().as_default():
        image,width = _get_input() # Placeholder tensors

        proc_image = _preprocess_image(image)
        proc_image = tf.reshape(proc_image,[1,32,-1,1]) # Make first dim batch

        with tf.device(FLAGS.device):
            features,sequence_length = model.convnet_layers( proc_image, width,mode)
            # features,sequence_length = zf_mod_denseNet2.Dense_net( proc_image, width, mode)
            logits = model.rnn_layers( features, sequence_length,
                                       mjsynth.num_classes() )
            prediction = _get_output( logits,sequence_length)

        session_config = _get_session_config()
        restore_model = _get_init_trained()
        
        init_op = tf.group( tf.global_variables_initializer(),
                            tf.local_variables_initializer()) 

        with tf.Session(config=session_config) as sess:
            
            sess.run(init_op)
            restore_model(sess, _get_checkpoint()) # Get latest checkpoint

            # Iterate over filenames given on lines of standard input
            # for line in sys.stdin: #命令行
            #     # Eliminate any trailing newline from filename
            #     image_data = _get_image(line.rstrip()) #image_data
            #     # Get prediction for single image (isa SparseTensorValue)
            #     [output] = sess.run(prediction,{ image: image_data,
            #                                      width: image_data.shape[1]} )
            #     print(_get_string(output.values))

            # base_dir = "I:\ICPR_task1_test_20180514\icpr_mtwi_task1\\testimage"
            base_dir = "..\\data\\test_image"
            bld = os.listdir(base_dir)
            # base_dir = "I:\ICPR_task1_test_20180514\icpr_mtwi_task1\9999"
            # tem_result = ""

            for i in bld:
                # name = "line_"+str(i)+".jpg"
                name = i
                image_data = _get_image(os.path.join(base_dir,name))
                if image_data.shape[0] > image_data.shape[1]:
                    image_data = image_data.reshape(image_data.shape[1], image_data.shape[0], 1)
                try:
                    [output] = sess.run(prediction, {image: image_data, width: image_data.shape[1]})
                    tem_result = name + " "+_get_string(output.values) + "\n"
                except:
                    f1 = open("..\\data\\val\\error.txt","a",encoding="utf-8")
                    tem_result = name + " 10" + "\n"
                    f1.write(tem_result)
                    f1.close()
                f = open("..\\data\\val\\result.txt", "a", encoding='utf-8')
                f.write(tem_result)
                f.close()
                print(name)

            # image_data = _get_image("I:\cnn_lstm_ctc_ocr_for_ICPR-master\data\images\\train_1000\horiz\\1_13.jpg")
            # [output] = sess.run(prediction, {image: image_data, width: image_data.shape[1]})
            # print(_get_string(output.values))





if __name__ == '__main__':
    tf.app.run()