import tensorflow as tf
import cv2
# import os
import numpy as np
# import matplotlib.pyplot as plt
import time
# import scipy.misc

import random
# from sklearn import preprocessing

def encode_to_tfrecode01(label_file, data_root, new_name='data.tfrecodes', resize=(85,130)):
    writer = tf.python_io.TFRecordWriter(data_root+'\\'+new_name)
    num_example = 0
    with open(label_file, 'r') as f:
        for l in f.readlines():
            l = l.split()
            image = cv2.imread(l[0])
            if resize is not None:
                image = cv2.resize(image, resize)
            height, width, nchannel = image.shape

            label = int(l[1])

            example = tf.train.Example(features=tf.train.Features(feature = {
                'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
                'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
                'nchannel': tf.train.Feature(int64_list=tf.train.Int64List(value=[nchannel])),
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tobytes()])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
            }))
            serialized = example.SerializeToString()
            writer.write(serialized)
            num_example += 1
    print(label_file, "样本数据量:", num_example)
    writer.close()

def encode_to_tfrecode(label_file, data_root, new_name='MedicalTestData_03231001.tfrecodes', resize=(85,130)):
    writer = tf.python_io.TFRecordWriter(data_root+'\\'+new_name)
    num_example = 0
    with open(label_file, 'r') as f:
        for l in f.readlines():
            l = l.split()
            print(l[0])
            image = cv2.imread(l[0])
            #image = cv2.imdecode(np.fromfile(l[0], dtype=np.uint8), -1)
            if resize is not None:
                image = cv2.resize(image, resize)
            height, width, nchannel = image.shape

            label = int(l[1])

            example = tf.train.Example(features=tf.train.Features(feature = {
                'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
                'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
                'nchannel': tf.train.Feature(int64_list=tf.train.Int64List(value=[nchannel])),
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tobytes()])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
            }))
            serialized = example.SerializeToString()
            writer.write(serialized)
            num_example += 1
    print(label_file, "样本数据量:", num_example)
    writer.close()

def decode_from_tfrecode(filename, num_epoch = None):
    filename_queue = tf.train.string_input_producer([filename], num_epochs = num_epoch)
    #因为有的训练数据过于庞大，被分成很多个文件，所以第一个参数就是文件列表名参数
    reader = tf.TFRecordReader()
    _, serialized = reader.read(filename_queue)
    example = tf.parse_single_example(serialized, features={
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'nchannel': tf.FixedLenFeature([], tf.int64),
        'image': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64)
    })
    label = tf.cast(example['label'], tf.int32)
    image = tf.decode_raw(example['image'], tf.uint8)
    image = tf.reshape(image, tf.stack([
        tf.cast(example['height'], tf.int32),
        tf.cast(example['width'], tf.int32),
        tf.cast(example['nchannel'], tf.int32)
    ]))
    return image, label

def funi(i):
    return i
def get_batch(image, label, batch_size):
    # image = tf.image.rgb_to_grayscale(image)
    # reimage = cv2.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))
    resized_image = tf.reshape(image, [IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNEL])

    #生成batch
    #shuffle_batch的参数：capacity用于定义shuttle的范围，如果是对整个训练数据集获取batch，capa应尽量大，保证数据打的足够乱
    image_batch, label_batch = tf.train.shuffle_batch(
        [resized_image, label], batch_size=batch_size, num_threads=16,
        capacity=50000, min_after_dequeue=10000
    )
    return image_batch, tf.reshape(label_batch, [batch_size, 1])


def get_batch_queue(image, label, batch_size):
    # image = tf.image.rgb_to_grayscale(image)
    # reimage = cv2.resize(image, (IMAGE_HEIGHT, IMAGE_WIDTH))
    resized_image = tf.reshape(image, [IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNEL])

    # 生成batch
    image_batch, label_batch = tf.train.batch([resized_image, label], batch_size=batch_size, num_threads=16, capacity=1)
    return image_batch, tf.reshape(label_batch, [batch_size, 1])

def crack_captcha_cnn(w_alpha=0.01, b_alpha=0.1):
    x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL])

    # w_c1_alpha = np.sqrt(2.0/(IMAGE_HEIGHT*IMAGE_WIDTH)) #
    # w_c2_alpha = np.sqrt(2.0/(3*3*32))
    # w_c3_alpha = np.sqrt(2.0/(3*3*64))
    # w_d1_alpha = np.sqrt(2.0/(8*32*64))
    # out_alpha = np.sqrt(2.0/1024)

    # 3 conv layer
    w_c1 = tf.Variable(w_alpha * tf.random_normal([3, 3, 3, 32]))
    b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1 = tf.nn.dropout(conv1, keep_prob)

    w_c2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 64]))
    b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.dropout(conv2, keep_prob)

    w_c3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 64]))
    b_c3 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = tf.nn.dropout(conv3, keep_prob)

    # Fully connected layer
    w_d = tf.Variable(w_alpha * tf.random_normal([17 * 11 * 64, 216]))
    b_d = tf.Variable(b_alpha * tf.random_normal([216]))
    dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    dense = tf.nn.dropout(dense, keep_prob)

    w_out = tf.Variable(w_alpha * tf.random_normal([216, LABELNUM]))
    b_out = tf.Variable(b_alpha * tf.random_normal([LABELNUM]))
    out = tf.add(tf.matmul(dense, w_out), b_out)
    return out

def One_Hot(label):
    shape = label.shape
    num = shape[0]
    label_r = np.zeros([num, LABELNUM])
    for i in range(num):
        label_r[i][label[i]] = 1
    return label_r

def train_crack_captcha_cnn(label_file):
    output = crack_captcha_cnn()
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    predict = tf.reshape(output, [-1, LABELNUM])
    max_idx_p = tf.argmax(predict, 1)############可能要改成0
    max_idx_l = tf.argmax(tf.reshape(Y, [-1, LABELNUM]), 1)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # label_file = "./MedicalTrainData_0315.tfrecodes"
    image, label = decode_from_tfrecode(label_file)
    # image = tf.image.rgb_to_grayscale(image)
    # batch_x = tf.reshape(image, [IMAGE_HEIGHT * IMAGE_WIDTH])
    # batch_y = tf.reshape(label, [-1 , 1])
    batch_x, batch_y = get_batch(image, label, batch_size=512)

    test_label_file = "./MedicalTestData_0703.tfrecodes"
    test_image, test_label = decode_from_tfrecode(test_label_file)
    # test_image = tf.image.rgb_to_grayscale(test_image)
    # test_batch_x = tf.reshape(test_image, [IMAGE_HEIGHT * IMAGE_WIDTH])
    # test_batch_y = tf.reshape(test_label, [-1 , 1])
    test_batch_x, test_batch_y = get_batch(test_image, test_label, batch_size=64)



    saver = tf.train.Saver()
    init = tf.initialize_all_variables()
    # config = tf.ConfigProto(allow_soft_placement=True)
    # # 最多占gpu资源的70%
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    # # 开始不会给tensorflow全部gpu资源 而是按需增加
    # config.gpu_options.allow_growth = True
    # sess = tf.Session(config=config)

    # config = tf.ConfigProto()
    # # config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.9
    # session = tf.Session(config=config)

    with tf.Session() as sess:
        sess.run(init)
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        step = 0
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        tim = str.split(time.asctime(time.localtime(time.time())))
        # print(tim[4] + tim[1] + tim[2] + "-" + tim[3])
        while True:
            # image_np, label_np = sess.run([image, label])
            batch_x_r, batch_y_r = sess.run([batch_x, batch_y])
            batch_y_r = One_Hot(batch_y_r)
            _, loss_ = sess.run([optimizer, loss], feed_dict={X: batch_x_r, Y: batch_y_r, keep_prob: 0.75})
            print(step, loss_)

            # 每100 step计算一次准确率
            if step % 10 == 0:
                # image_np, label_np = sess.run([test_image, test_label])
                batch_x_r, batch_y_r = sess.run([test_batch_x, test_batch_y])
                batch_y_r = One_Hot(batch_y_r)
                acc = sess.run(accuracy, feed_dict={X: batch_x_r, Y: batch_y_r, keep_prob: 1.})
                print('accuracy:', step, acc)

                # 如果准确率大于95%,保存模型,完成训练
            if acc > 0.98 and step%10==0:
                saver.save(sess, "../model_3class/crack_capcha.model"+"-"+tim[4] + tim[1] + tim[2] + "-" + tim[3][0:2]+ tim[3][3:5]+ tim[3][6:8], global_step=step)
            if step==20000:
                saver.save(sess, "../model_3class/crack_capcha.model"+"-"+tim[4] + tim[1] + tim[2] + "-" + tim[3][0:2]+ tim[3][3:5]+ tim[3][6:8], global_step=step)
                break

            step += 1
        coord.request_stop()
        coord.join(threads)
def crack_captcha(captcha_image):
    output = crack_captcha_cnn()

    saver = tf.train.Saver()
    init = tf.initialize_all_variables()
    # config = tf.ConfigProto(allow_soft_placement=True)
    # # 最多占gpu资源的70%
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    # # 开始不会给tensorflow全部gpu资源 而是按需增加
    # config.gpu_options.allow_growth = True
    # sess = tf.Session(config=config)
    config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    session = tf.Session(config=config)
    with tf.Session() as sess:
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        saver.restore(sess, "./model/crack_capcha.model-1420")

        predict = tf.argmax(tf.reshape(output, [-1, LABELNUM]), 1)
        nn = 0
        error2yang = 0
        error2yin = 0
        yang = 0
        yin = 0
        yisi = 0
        for n in range(449):
            batch_x_r, batch_y_r = sess.run([captcha_image, test_batch_y])

            imageshow = batch_x_r.reshape(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL)

            # plt.waitforbuttonpress()
            #  f = plt.figure()
            #  ax = f.add_subplot(111)
            #  ax.text(0.1, 0.9, batch_y_r, ha='center', va='center', transform=ax.transAxes)
            # # plt.imshow(imageshow)
            #  plt.pause(3)
            #  plt.close()

            text_list = sess.run(predict, feed_dict={X: batch_x_r, keep_prob: 1})
            text = text_list[0].tolist()
            # if text > 9:
            #     text = chr(ord('A') + text - 10)
            # else:
            #     text = chr(ord('0') + text)
            if text == 0:
                text = 'Yin'
            elif text == 1:
                text = 'Yang'
            # else:
            #     text = "yisi"

            label = batch_y_r[0].tolist()
            label = label[0]
            # if label > 9:
            #     label = chr(ord('A') + label - 10)
            # else:
            #     label = chr(ord('0') + label)
            if label == 0:
                label = 'Yin'
                yin += 1
            elif label == 1:
                label = 'Yang'
                yang += 1
            else:
                label == "yisi"
                yisi += 1

            print("正确: ", label, "预测: ", text)
            if label == text:
                nn += 1
            elif label == "Yin" and text == "Yang":
                error2yang += 1
            elif label == "Yang" and text == "Yin":
                error2yin += 1

            if isinstance(label, int) == False:
                if label != text:
                    imgname = label + '_' + text + '___' + str(n) + "___.jpg"
                    pathname = "./imgres/" + imgname
                    imageshow = cv2.resize(imageshow, (85, 130))
                    cv2.imwrite(pathname, imageshow)

        print("正确数：",nn,"循环数：",n+1,"准确率：",nn/(n+1-yisi), "阴性准确率：", 1-error2yang/yin ,"阳性准确率：", 1-error2yin/yang )
        print("yin:", yin, "yang:", yang, "yisi:", yisi)
        coord.request_stop()
        coord.join(threads)
        return text, batch_y_r
if __name__ == '__main__':
    train = 1
    if train == 1:

        # 图像大小
        IMAGE_HEIGHT = 130
        IMAGE_WIDTH = 85
        IMAGE_CHANNEL = 3

        LABELNUM = 3

        X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNEL])
        Y = tf.placeholder(tf.float32, [None, LABELNUM])

        keep_prob = tf.placeholder(tf.float32)  # dropout

        # label_file_list = 'D:\\AI4Medical\\project0315\\Train3.0\\test_list0403.txt'
        # data_path = 'D:\\AI4Medical\\project0315\\Train3.0'
        # .............................................................................................................................................
        # encode_to_tfrecode(label_file_list, data_path)

        train_crack_captcha_cnn("./MedicalTrainData_0921.tfrecodes")

    if train == 0:
        # 图像大小
        IMAGE_HEIGHT = 130
        IMAGE_WIDTH = 85
        IMAGE_CHANNEL = 3

        LABELNUM = 3

        test_label_code = "./MedicalTestData_0402.tfrecodes"
        test_image, test_label = decode_from_tfrecode(test_label_code)
        test_batch_x, test_batch_y = get_batch_queue(test_image, test_label, batch_size=1)

        X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNEL])
        Y = tf.placeholder(tf.float32, [None, LABELNUM])
        keep_prob = tf.placeholder(tf.float32)  # dropout

        predict_text, label_test, imageshow = crack_captcha(test_batch_x)


