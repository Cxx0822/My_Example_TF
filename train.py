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
BATCH_SIZE = my_data['batch_size']
MAX_STEP = my_data['max_step'] 
IMG_SIZE = my_data['resize_height'] 
LEARNING_RATE = my_data['LEARNING_RATE']

resize_height = my_data['resize_height']
resize_width = my_data['resize_width']
shuffle = my_data['shuffle']
logs_dir = my_data['logs_dir']

train_tfrecord = my_data['train_tfrecord']


def my_train_queue():
    # 加载数据集
    record_file = 'dataSet/record/train{}.tfrecords'.format(IMG_SIZE)
    tf_image, tf_label = read_records(record_file, IMG_SIZE, IMG_SIZE, type='normalization')
    image_train_batch, label_train_batch = get_batch_images(tf_image, tf_label, batch_size=BATCH_SIZE,
                                                            labels_nums=N_CLASSES, one_hot=False, shuffle=True)

    sess = tf.Session()

    # 加载模型，并计算损失、正确率等
    train_logits = inference_op(image_train_batch, N_CLASSES)
    train_loss = losses(train_logits, label_train_batch)
    train_acc = evaluation(train_logits, label_train_batch)

    # 优化器迭代学习
    train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(train_loss)

    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())

    # 启动queue
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # 计时
    start_time = time.time()
    now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    try:
        for step in range(MAX_STEP):
            if coord.should_stop():
                break
            # 启动模型、损失、正确率
            _, loss, acc = sess.run([train_op, train_loss, train_acc])

            # 每10步显示，可以自己更改
            if step % 10 == 0:  
                runtime = time.time() - start_time
                print('Step: %6d, now_time: %s, loss: %.8f, accuracy: %.2f%%, time:%.2fs, time left: %.2fhours'
                      % (step, str(now_time), loss, acc * 100, runtime, (MAX_STEP - step) * runtime / 3600))
                start_time = time.time()

            # 保存最后一步的模型，可以自己更改
            if step == MAX_STEP - 1:  
                checkpoint_path = os.path.join(logs_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

    except tf.errors.OutOfRangeError:
        print('Done.')
    finally:
        coord.request_stop()

    coord.join(threads=threads)
    sess.close()


def my_train_tfData():
    # 加载数据集
    filename = tf.compat.v1.placeholder(tf.string, shape=[None])

    train_dataset = tf.data.TFRecordDataset(filename)
    train_dataset = train_dataset.map(parse_records).shuffle(10000).batch(BATCH_SIZE).repeat(MAX_STEP)

    iterator = tf.compat.v1.data.make_initializable_iterator(train_dataset)     # 生成迭代器
    train_images, train_labels = iterator.get_next()

    train_filename = []
    train_filename.append(train_tfrecord)   # 将tfrecords文件添加到文件名列表中（run里面需要列表格式，直接使用字符串会报错）

    # 加载模型，并计算损失、正确率等
    train_logits = inference_op(train_images, N_CLASSES)
    train_loss = losses(train_logits, train_labels)
    train_acc = evaluation(train_logits, train_labels)

    # 优化器迭代学习
    train_op = tf.compat.v1.train.AdamOptimizer(LEARNING_RATE).minimize(train_loss)

    # 启动线程
    saver = tf.compat.v1.train.Saver()
    with tf.compat.v1.Session() as sess:
        # 启动初始化和迭代器
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(iterator.initializer, feed_dict={filename: train_filename})

        # 计时
        start_time = time.time()
        now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')   

        # 迭代学习
        for step in range(MAX_STEP):   
            # 启动模型、损失、正确率
            _, loss, acc = sess.run([train_op, train_loss, train_acc])

            # 每10步显示，可以自己更改
            if step % 10 == 0:  
                runtime = time.time() - start_time
                print('Step: %6d, now_time: %s, loss: %.8f, accuracy: %.2f%%, time:%.2fs, time left: %.2fhours'
                      % (step, str(now_time), loss, acc * 100, runtime, (MAX_STEP - step) * runtime / 3600))
                start_time = time.time()

            # 保存最后一步的模型，可以自己更改
            if step == MAX_STEP - 1:  
                checkpoint_path = os.path.join(logs_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

if __name__ == '__main__':
    my_train_queue()
    # my_train_tfData()
