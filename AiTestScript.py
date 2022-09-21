# -*- coding:utf-8 -*-
import sys
import tensorflow as tf
import cv2
import numpy as np

def encode_to_tfrecode(dir_list, path, IMAGE_WIDTH = None, IMAGE_HEIGHT = None):
    # path = os.path.abspath(data_root + new_name + '.tfrecodes')
    writer = tf.python_io.TFRecordWriter(path)
    num_example = 0
    for d in dir_list:
        image = cv2.imread(d)

        if IMAGE_WIDTH and IMAGE_HEIGHT is not None:
            image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
        height, width, nchannel = image.shape

        label = -99

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
    # with open(label_file, 'r') as f:
    #     for l in f.readlines():
    #         l = l.split()
    #         # print(l[0])
    #         image = cv2.imread(l[0])
    #
    #         if IMAGE_WIDTH and IMAGE_HEIGHT is not None:
    #             image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
    #         height, width, nchannel = image.shape
    #
    #         label = int(l[1])
    #
    #         example = tf.train.Example(features=tf.train.Features(feature = {
    #             'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
    #             'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
    #             'nchannel': tf.train.Feature(int64_list=tf.train.Int64List(value=[nchannel])),
    #             'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tobytes()])),
    #             'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
    #         }))
    #         serialized = example.SerializeToString()
    #         writer.write(serialized)
    #         num_example += 1
    print( "样本数据量:", num_example)
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
    init = tf.initialize_all_variables()
    # config = tf.ConfigProto(allow_soft_placement=True)
    # # 最多占gpu资源的70%
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    # # 开始不会给tensorflow全部gpu资源 而是按需增加
    # config.gpu_options.allow_growth = True
    # sess = tf.Session(config=config)
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.9
    # session = tf.Session(config=config)
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
        resTxt = []
        for n in range(numberOfImage):
            batch_x_r, batch_y_r = sess.run([captcha_image, test_batch_y])

            imageshow = batch_x_r.reshape(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL)

            text_list = sess.run(predict, feed_dict={X: batch_x_r, keep_prob: 1})
            text = text_list[0].tolist()

            resTxt.append(text) #0或1加入list，保存为一个TXT

            if text == 0:
                text = 'Yin'
            elif text == 1:
                text = 'Yang'

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
            elif label == -99 and text == "Yin":
                yisi2yin += 1
            elif label == -99 and text == "Yang":
                yisi2yang += 1

            if isinstance(label, int) == False:
                if label != text:
                    imgname = label + '_' + text + '___' + str(n) + "___.jpg"
                    pathname = "./imgres/" + imgname
                    imageshow = cv2.resize(imageshow, (IMAGE_WIDTH, IMAGE_HEIGHT))
                    cv2.imwrite(pathname, imageshow)
        if yin == 0 and yang != 0:
            print("正确数：", nn, "循环数：", n + 1, "准确率：", nn / (n + 1 - yisi), "阳性准确率：", 1 - error2yin / yang)
        elif yang == 0 and yin != 0:
            print("正确数：", nn, "循环数：", n + 1, "准确率：", nn / (n + 1 - yisi), "阴性准确率：", 1 - error2yang / yin)
        elif yin != 0 and yang != 0:
            print("正确数：", nn, "循环数：", n + 1, "准确率：", nn / (n + 1 - yisi), "阴性准确率：", 1 - error2yang / yin, "阳性准确率：",
                  1 - error2yin / yang)
        elif yin == 0 and yang == 0 and yisi != 0:
            print("疑似数量：", yisi, "预测为阳性：", yisi2yang, "预测为阴性；", yisi2yin)
        print("yin:", yin, "yang:", yang, "yisi:", yisi)
        print(resTxt)

        txtname = 'AiTestResult.txt'
        with open(txtname, 'w') as f:  # 如果txtname不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据！
            for i in resTxt:
                f.write(str(i)+'\n')
            # f.write(resTxt)

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':

        # 图像大小
        IMAGE_HEIGHT = 130
        IMAGE_WIDTH = 85
        IMAGE_CHANNEL = 3

        LABELNUM = 2

        # date = "0416"
        # label_file_list = './test_list'+date+'.txt'

        dirList00 = ["D:\\AI4Medical\\project0315\\data180418\\20180410--41-4.JPG",
                   "D:\\AI4Medical\\project0315\\data180418\\20180410--41-5.JPG",
                   "D:\\AI4Medical\\project0315\\data180418\\20180410--41-6.JPG",
                   "D:\\AI4Medical\\project0315\\data180418\\20180410--41-7.JPG",
                   "D:\\AI4Medical\\project0315\\data180418\\20180410--42-1.JPG",
                   "D:\\AI4Medical\\project0315\\data180418\\20180410--42-2.JPG"]
        dirList = []
        # infile = open(sys.argv[1])
        # for each_line in infile:
        for each_line in dirList00:
            dirList.append(each_line)
        print(dirList)
        root_path = './'
        tf_file_last = 'MedicalTestData_'+ '__999__' + '.tfrecodes'
        modelname = 'crack_capcha.model-2018Apr10-103-3000'
        # modelfile = root_path +'model/'+ modelname
        modelfile =  '../model/' + modelname
        codeFilePath = root_path + tf_file_last
        # if codeFilePath == None:
        # cnt = encode_to_tfrecode(label_file_list, codeFilePath, IMAGE_WIDTH, IMAGE_HEIGHT)
        cnt = encode_to_tfrecode(dirList, codeFilePath, IMAGE_WIDTH, IMAGE_HEIGHT)

        test_image, test_label = decode_from_tfrecode(codeFilePath)

        test_batch_x, test_batch_y = get_batch_queue(test_image, test_label, batch_size=1)

        X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNEL])
        Y = tf.placeholder(tf.float32, [None, LABELNUM])
        keep_prob = tf.placeholder(tf.float32)  # dropout

        crack_captcha(test_batch_x, cnt, modelfile, IMAGE_WIDTH, IMAGE_HEIGHT)


