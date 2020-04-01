# enconding: utf-8
import os
import pickle

import cv2
import numpy as np
import tensorflow as tf

from vgg16 import Vgg16

vgg_mean = [0.485, 0.456, 0.406]
vgg_std = [0.229, 0.224, 0.225]
data = None
weights_path = "./model/vgg16_onnx.npy"
'''use conv3_1 to generate representation'''
FEATURE_LAYERS = ['conv3_1']  # 提取该层feature map
image_shape = (1, 224, 224, 3)


class Config:
    log_dir = "./train_log"  # 日志文件
    log_model_dir = os.path.join(log_dir, 'models')
    exp_name = os.path.basename(log_dir)
    nr_channel = 3
    nr_epoch = 2000
    save_interval = 10
    show_interval = 10
    snapshot_interval = 2


config = Config()
# In[3]:

'''load target image, notice that the pixel value has to be normalized to [0,1]'''
image = cv2.imread('./images/face.jpg')
image = cv2.resize(image, image_shape[1:3])  # 去掉第一个属性
img = image
image = image.reshape(image_shape)
image = (image / 255).astype('float32')

'''training a image from noise that resemble the target'''
# pre_noise = tf.Variable()  # 从[-3,3]范围内产生随机噪声
noise = tf.Variable(tf.nn.sigmoid(tf.random_uniform(image_shape, -3, 3)))


def get_feature_loss(noise, source):
    with tf.name_scope('get_feature_loss'):
        feature_loss = []
        for layer in FEATURE_LAYERS:  # 执行一次 conv3_1
            feature_loss.append(get_l2_loss_for_layer(noise, source, layer))
    return tf.reduce_mean(tf.convert_to_tensor(feature_loss))


def get_l2_loss_for_layer(noise, source, layer):
    alpha = 0.01
    noise_layer = getattr(noise, layer)  # 获得noise对象的layer属性
    source_layer = getattr(source, layer)
    l2_loss = tf.reduce_mean((source_layer - noise_layer) ** 2)
    tv_regular = tf.reduce_sum(tf.image.total_variation(noise_layer))

    return l2_loss + alpha * tv_regular


def output_img(session, x, save=False, out_path=None):
    shape = image_shape
    img = np.clip(session.run(x), 0, 1) * 255
    img = img.astype('uint8')
    if save:
        cv2.imwrite(out_path, (np.reshape(img, shape[1:])))


# '''get representation of the target image, which the noise image will approximate'''
with tf.name_scope('vggNet'):
    image_model = Vgg16()
    image_model.build(image)
    noise_model = Vgg16()  # 获得噪声的图像信息
    noise_model.build(noise)

# with tf.name_scope('resNet'):
#     image_model = Resnet18()
#     image_model.build(image)
#     noise_model = Resnet18()
#     noise_model.build(noise)

'''compute representation difference between noise feature and target feature'''
with tf.name_scope('loss'):
    loss = get_feature_loss(noise_model, image_model)  # 获得噪声和原图像的loss

total_loss = loss
global_steps = tf.Variable(0, trainable=False)
lr = 0.001  # 0.001
with tf.name_scope('update_image'):
    opt = tf.train.AdamOptimizer(lr)  # 优化器
    grads = opt.compute_gradients(total_loss, [noise])  # 计算loss 对noise的倒数
    update_image = opt.apply_gradients(grads)

''' create a session '''
tf.set_random_seed(12345)  # ensure consistent results
global_cnt = 0
epoch_start = 0
losslist = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())  # init all variables
    ## training
    for epoch in range(0, 1000):
        global_cnt += 1
        # print(global_cnt)
        _, loss = sess.run([update_image, total_loss],
                           feed_dict={global_steps: global_cnt})
        losslist.append(loss)
        if global_cnt % config.show_interval == 0:
            print(
                "epoch:{}".format(epoch), 'loss: {:.5f}'.format(loss),
            )

        '''save the trained noise image every 10 epoch, check whether it resembles the target image'''
        if global_cnt % config.save_interval == 0 and global_cnt > 0:
            out_dir = './out'
            out_dir = out_dir + '/{}.png'.format(global_cnt)
            output_img(sess, noise, save=True, out_path=out_dir)
    print('Training is done, exit.')
    path = "tv.obj"
    file = open(path, "wb")
    pickle.dump(losslist, file)
