# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import cv2

def encode_to_tfrecode(label_file, data_path, new_name='MedicalTrainData_0921.tfrecodes', resize=(85,130)):
    writer = tf.python_io.TFRecordWriter(data_path+'\\'+ new_name)
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

if __name__ == '__main__':
    label_file_list = 'D:\\AI4Medical\\project0315\\Train3.00\\train_list0921.txt'
    data_path = 'D:\\AI4Medical\\project0315\\Train3.00'

    encode_to_tfrecode(label_file_list, data_path)