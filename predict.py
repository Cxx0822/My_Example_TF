import time
import datetime

from create_tf_record import *
from create_labels_files import *
from model import *
import matplotlib.pyplot as plt

import yaml

with open("info.yml") as stream:
    my_data = yaml.load(stream, Loader=yaml.FullLoader)

N_CLASSES = my_data['n_classes']
BATCH_SIZE = 1
MAX_STEP = 10
IMG_SIZE = my_data['resize_height'] 
resize_height = my_data['resize_height']
logs_dir = my_data['logs_dir']

my_labels = my_data['my_labels']
val_tfrecord = my_data['val_tfrecord']


def my_predict_queue():
    record_file = val_tfrecord

    sess = tf.Session()
    # 加载数据集
    tf_image, tf_label = read_records(record_file, IMG_SIZE, IMG_SIZE, type='normalization')
    image_val_batch, label_val_batch = get_batch_images(tf_image, tf_label, batch_size=BATCH_SIZE,
                                                        labels_nums=N_CLASSES, one_hot=False, shuffle=True)

    # 预测模型
    val_logits = inference_op(image_val_batch, N_CLASSES)
    val_logits = tf.nn.softmax(val_logits)  

    # 加载模型
    saver = tf.train.Saver()
    print('\nLoading checkpoints...')
    ckpt = tf.train.get_checkpoint_state(logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Successful loading, global_step = %s\n' % global_step)
    else:
        print('No checkpoints')

    # 启动queue
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:    
        # 预测结果
        for step in range(MAX_STEP):
            if coord.should_stop():
                break

            # 启动图片、预测
            image, prediction = sess.run([image_val_batch, val_logits])  
            max_index = np.argmax(prediction)     

            # 判断类别
            if max_index == 0:
                label = '%.2f%% ' % (prediction[0][0] * 100) + 'is a ' + str(my_labels[0]) + '.'
            elif max_index == 1:
                label = '%.2f%% ' % (prediction[0][1] * 100) + 'is a ' + str(my_labels[1]) + '.'
            elif max_index == 2:
                label = '%.2f%% ' % (prediction[0][2] * 100) + 'is a ' + str(my_labels[2]) + '.'
            elif max_index == 3:
                label = '%.2f%% ' % (prediction[0][3] * 100) + 'is a ' + str(my_labels[3]) + '.'
            elif max_index == 4:
                label = '%.2f%% ' % (prediction[0][4] * 100) + 'is a ' + str(my_labels[4]) + '.'

            # 显示图片
            plt.imshow(image[0])
            plt.title(label)
            plt.show()

    except tf.errors.OutOfRangeError:
        print('Done.')
    finally:
        coord.request_stop()

    coord.join(threads=threads)
    sess.close()


def my_predict_tfData():
    filename = tf.compat.v1.placeholder(tf.string, shape=[None])

    with tf.compat.v1.Session() as sess:
        # 加载数据集
        val_dataset = tf.data.TFRecordDataset(filename)

        val_dataset = val_dataset.map(parse_records).shuffle(10000).batch(1).repeat(MAX_STEP)

        iterator = val_dataset.make_initializable_iterator()   # 生成迭代器
        val_images, val_labels = iterator.get_next()

        val_filename = []
        val_filename.append(val_tfrecord)   # 将tfrecords文件添加到文件名列表中（run里面需要列表格式，直接使用字符串会报错）

        # 预测模型
        val_logits = inference_op(val_images, N_CLASSES)
        val_logits = tf.nn.softmax(val_logits)  

        # 启动线程
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(iterator.initializer, feed_dict={filename: val_filename})

        # 加载模型
        saver = tf.compat.v1.train.Saver()
        print('\nLoading checkpoints...')
        ckpt = tf.train.get_checkpoint_state(logs_dir)
        if ckpt and ckpt.model_checkpoint_path:
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Successful loading, global_step = %s\n' % global_step)
        else:
            print('No checkpoints')

        # 预测结果
        for step in range(MAX_STEP):  
            # 启动图片、预测
            image, prediction = sess.run([val_images, val_logits]) 
            max_index = np.argmax(prediction)     

            # 判断类别
            if max_index == 0:
                label = '%.2f%% ' % (prediction[0][0] * 100) + 'is a ' + str(my_labels[0]) + '.'
            elif max_index == 1:
                label = '%.2f%% ' % (prediction[0][1] * 100) + 'is a ' + str(my_labels[1]) + '.'
            elif max_index == 2:
                label = '%.2f%% ' % (prediction[0][2] * 100) + 'is a ' + str(my_labels[2]) + '.'
            elif max_index == 3:
                label = '%.2f%% ' % (prediction[0][3] * 100) + 'is a ' + str(my_labels[3]) + '.'
            elif max_index == 4:
                label = '%.2f%% ' % (prediction[0][4] * 100) + 'is a ' + str(my_labels[4]) + '.'

            # 显示图片
            plt.imshow(image[0])
            plt.title(label)
            plt.show()


if __name__ == '__main__':
    my_predict_queue()
    # my_predict_tfData()
