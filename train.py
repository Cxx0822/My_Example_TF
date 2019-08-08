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

resize_height = my_data['resize_height']
resize_width = my_data['resize_width']
shuffle = my_data['shuffle']
logs_dir = my_data['logs_dir']


def my_train():
    CAPACITY = 200
    LEARNING_RATE = 1e-4

    record_file = 'dataSet/record/train{}.tfrecords'.format(IMG_SIZE)
    tf_image, tf_label = read_records(record_file, IMG_SIZE, IMG_SIZE, type='normalization')
    image_train_batch, label_train_batch = get_batch_images(tf_image, tf_label, batch_size=BATCH_SIZE,
                                                            labels_nums=N_CLASSES, one_hot=False, shuffle=False)

    sess = tf.Session()

    train_logits = inference_op(image_train_batch, 0.5, N_CLASSES)
    train_loss = losses(train_logits, label_train_batch)
    train_acc = evaluation(train_logits, label_train_batch)
    train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(train_loss)

    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    start_time = time.time()
    now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    try:
        for step in range(MAX_STEP):
            if coord.should_stop():
                break

            _, loss, acc = sess.run([train_op, train_loss, train_acc])

            if step % 1 == 0:  
                runtime = time.time() - start_time
                print('Step: %6d, now_time: %s, loss: %.8f, accuracy: %.2f%%, time:%.2fs, time left: %.2fhours'
                      % (step, str(now_time), loss, acc * 100, runtime, (MAX_STEP - step) * runtime / 3600))
                start_time = time.time()

            if step % 1000 == 0 or step == MAX_STEP - 1:  
                checkpoint_path = os.path.join(logs_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

    except tf.errors.OutOfRangeError:
        print('Done.')
    finally:
        coord.request_stop()

    coord.join(threads=threads)
    sess.close()


if __name__ == '__main__':
    my_train()
