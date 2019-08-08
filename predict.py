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


def my_predict():
    CAPACITY = 200
    record_file = 'dataSet/record/val{}.tfrecords'.format(resize_height)

    sess = tf.Session()

    tf_image, tf_label = read_records(record_file, IMG_SIZE, IMG_SIZE, type='normalization')
    image_train_batch, label_train_batch = get_batch_images(tf_image, tf_label, batch_size=BATCH_SIZE,
                                                            labels_nums=N_CLASSES, one_hot=False, shuffle=False)

    train_logits = inference_op(image_train_batch, 0.5, N_CLASSES)
    train_logits = tf.nn.softmax(train_logits)  

    saver = tf.train.Saver()
    print('\nLoading checkpoints...')
    ckpt = tf.train.get_checkpoint_state(logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Successful loading, global_step = %s\n' % global_step)
    else:
        print('No checkpoints')
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:    
        for step in range(MAX_STEP):
            if coord.should_stop():
                break

            image, prediction = sess.run([image_train_batch, train_logits])  
            max_index = np.argmax(prediction)     

            if max_index == 0:
                label = '%.2f%% is a' + str(my_labels[0]) + '.' % (prediction[0][0] * 100)
            elif max_index == 1:
                label = '%.2f%% is a' + str(my_labels[1]) + '.' % (prediction[0][1] * 100)
            elif max_index == 2:
                label = '%.2f%% is a' + str(my_labels[2]) + '.' % (prediction[0][2] * 100)
            elif max_index == 3:
                label = '%.2f%% is a' + str(my_labels[3]) + '.' % (prediction[0][3] * 100)
            elif max_index == 4:
                label = '%.2f%% is a' + str(my_labels[4]) + '.' % (prediction[0][4] * 100)

            plt.imshow(image[0])
            plt.title(label)
            plt.show()

    except tf.errors.OutOfRangeError:
        print('Done.')
    finally:
        coord.request_stop()

    coord.join(threads=threads)
    sess.close()


if __name__ == '__main__':
    my_predict()
