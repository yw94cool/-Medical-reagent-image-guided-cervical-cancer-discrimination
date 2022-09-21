import tensorflow as tf
import cv2
import os
import numpy as np
# import matplotlib.pyplot as plt
import time
# import scipy.misc

import random
# from sklearn import preprocessing

def encode_to_tfrecode(label_file, path, IMAGE_WIDTH = None, IMAGE_HEIGHT = None):
    # path = os.path.abspath(data_root + new_name + '.tfrecodes')
    writer = tf.python_io.TFRecordWriter(path)
    num_example = 0
    with open(label_file, 'r') as f:
        for l in f.readlines():
            l = l.split()
            print(l[0])
            # if cv2.imread(l[0]):
            image = cv2.imread(l[0])
            # else:
            #     print("opencv读取文件失败！")


            if IMAGE_WIDTH and IMAGE_HEIGHT is not None:
                image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
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
    return num_example

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


def crack_captcha(captcha_image, numberOfImage, modelfile, IMAGE_WIDTH, IMAGE_HEIGHT):
    output = crack_captcha_cnn()

    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    fo = open('./resultTemReport.txt', 'w')
    with tf.Session() as sess:
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        saver.restore(sess, modelfile)

        predict = tf.argmax(tf.reshape(output, [-1, LABELNUM]), 1)
        nn = 0
        error2yang = 0
        error2yin = 0
        yang = 0
        yin = 0
        yisi = 0
        yisi2yang = 0
        yisi2yin = 0
        for n in range(numberOfImage):
            batch_x_r, batch_y_r = sess.run([captcha_image, test_batch_y])

            imageshow = batch_x_r.reshape(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL)

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
            elif text == 2:
                text = "Yisi"

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
                # label = "Yisi"
                yisi += 1

            print("正确: ", label, "预测: ", text)
            if label == text:
                nn += 1
            elif label == "Yin" and text == "Yang":
                error2yang += 1
            elif label == "Yang" and text == "Yin":
                error2yin += 1
            elif label == -99 and text == "Yin":
                yisi2yin += 1
            elif label == -99 and text == "Yang":
                yisi2yang += 1

            if isinstance(label, int) == False:
                if label != text :
                    imgname = label + '_' + text + '___' + str(n) + "___.jpg"
                    pathname = "./imgres/" + imgname
                    imageshow = cv2.resize(imageshow, (IMAGE_WIDTH, IMAGE_HEIGHT))
                    cv2.imwrite(pathname, imageshow)
            if text == 'Yin':
                fo.write('-' + '\n')
            if text == 'Yang':
                fo.write('+' + '\n')
            if text == "Yisi":
                fo.write('?' + '\n')
        if yin == 0 and yang != 0:
            print("正确数：", nn, "循环数：", n + 1, "准确率：", nn / (n + 1 - yisi), "阳性准确率：", 1 - error2yin / yang)
        elif yang == 0 and yin != 0:
            print("正确数：", nn, "循环数：", n + 1, "准确率：", nn / (n + 1 - yisi), "阴性准确率：", 1 - error2yang / yin)
        elif yin != 0 and yang != 0:
            print("正确数：",nn,"循环数：",n+1,"准确率：",nn/(n+1-yisi), "阴性准确率：", 1 - error2yang/yin ,"阳性准确率：", 1 - error2yin/yang )
        elif yin == 0 and yang == 0 and yisi != 0:
            print("疑似数量：", yisi, "预测为阳性：",yisi2yang, "预测为阴性；", yisi2yin)
        print("yin:", yin, "yang:", yang, "yisi:", yisi)
        coord.request_stop()
        coord.join(threads)
    fo.close()

def file_name(file_dir):
    for root, dirs, files in os.walk(file_dir):
        #         print(root) #当前目录路径
        #         print(dirs) #当前路径下所有子目录
        #         print(files) #当前路径下所有非目录子文件
        return files

def generateList(txtdirAndname, imagedir, way2opentxt, _0or1=-99, file=file_name):  # 生成的txt路径、图像路径、打开文件方式、标签
    f = open(txtdirAndname, way2opentxt)
    for n in file(imagedir):
        f.write(imagedir + "\\" + n + " " + str(_0or1) + "\n")
    f.close()

def generateReport(listfile, reportfile, date):
    f0 = open(listfile, 'r')
    f1 = open(reportfile, 'r')
    ff = open("./ReportList"+date+".txt", 'w')
    nowTime = time.strftime(' %Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    l0 = []
    l1 = []
    ff.write(u"      ********************************\n"
             u"      *                              *\n"
             u"      *        图像检测报告          *\n"
            u"      *                              *\n"
             u"      *    IMAGE DETECTION REPORT    *\n"
             u"      *                              *\n"
             u"      *     "+ nowTime+ "     *\n"
                                "      *                              *\n"
                                "      ********************************\n\n")
    ff.write("序号"+ '      '+ "影像名"+ '    '+ "Ai检测结果" + '    '+ "TCT结果" "\n")
    for i in f0:
        l0.append(i.split()[0])
    for j in f1:
        l1.append(j)
    for i in range(len(l0)):
        if i<9:
            ff.write(str(i + 1) + '      ' + ((l0[i].split('\\'))[-1]).split('.')[0] +'    ' + l1[i])
        elif i>=9 and i <99:
            ff.write(str(i + 1) + '     ' + ((l0[i].split('\\'))[-1]).split('.')[0] + '    ' + l1[i])
        elif i>=99 and i < 999:
            ff.write(str(i + 1) + '    ' + ((l0[i].split('\\'))[-1]).split('.')[0] + '    ' + l1[i])
        elif i>=999 and i <9999:
            ff.write(str(i + 1) + '   ' + ((l0[i].split('\\'))[-1]).split('.')[0] + '    ' + l1[i])
        elif i>=9999 and i<99999:
            ff.write(str(i + 1) + '  ' + ((l0[i].split('\\'))[-1]).split('.')[0] + '    ' + l1[i])
        else:
            ff.write(str(i + 1) + ' ' + ((l0[i].split('\\'))[-1]).split('.')[0] + '    ' + l1[i])
        # print(str(i) + '\t'+l0[i] + '\t'+((l0[i].split('\\'))[-1]).split('.')[0] +'\t' + l1[i])
    f0.close()
    f1.close()
    ff.close()


if __name__ == '__main__':

        # 图像大小
        IMAGE_HEIGHT = 130
        IMAGE_WIDTH = 85
        IMAGE_CHANNEL = 3

        LABELNUM = 3

        date = "0712"
        inputFileDir = "D:\\AI4Medical\\project0315\\data180712"

        label_file_list = './test_list' + date + '.txt'
        generateList(label_file_list, inputFileDir, 'w')

        root_path = './'
        tf_file_last = 'MedicalTestData_'+ date + '.tfrecodes'
        modelname = 'crack_capcha.model-2018Jul11-160300-22410'
        # modelfile = root_path +'model/'+ modelname
        modelfile =  '../model_3class/' + modelname
        codeFilePath = root_path + tf_file_last
        # if codeFilePath == None:
        cnt = encode_to_tfrecode(label_file_list, codeFilePath, IMAGE_WIDTH, IMAGE_HEIGHT)

        test_image, test_label = decode_from_tfrecode(codeFilePath)
        test_batch_x, test_batch_y = get_batch_queue(test_image, test_label, batch_size=1)

        X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNEL])
        Y = tf.placeholder(tf.float32, [None, LABELNUM])
        keep_prob = tf.placeholder(tf.float32)  # dropout

        crack_captcha(test_batch_x, cnt, modelfile, IMAGE_WIDTH, IMAGE_HEIGHT)

        generateReport(label_file_list, './resultTemReport.txt', date)


