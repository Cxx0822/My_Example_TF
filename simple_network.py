import numpy as np
import tensorflow as tf

import os
import os.path

# 基本信息配置
input_dim = 320
label_dim = 2

max_epochs = 50
minibatch_size = 25

logs_dir = 'logs_1'
filename = "data.txt"


def load_data(filename, input_dim):
    # 读取数据集
    with open(filename, "r") as fr:
        arrayOfLines = fr.readlines()
        numberOfLines = len(arrayOfLines)

        X = np.zeros((numberOfLines, input_dim))
        Y = []
        index = 0  
        for line in arrayOfLines:
            line = line.strip()
            listFromLine = line.split(' ')
            X[index, :] = listFromLine[0:input_dim]      # 读取特征列

            if listFromLine[-1] == '0':                  # 读取标签列
                Y.append(0)
            elif listFromLine[-1] == '1':
                Y.append(1)     

            index += 1  

    # 将标签列的格式转为one-hot类型，即[0 0 0 1 .... 0 0]
    from sklearn.preprocessing import LabelEncoder
    import pandas as pd
    encoder = LabelEncoder()
    Y = encoder.fit_transform(Y)
    Y = pd.get_dummies(Y).values

    return X, Y


def random_batch(X_train, Y_train, batch_size):
    # 随机选取batch_size个数据
    rnd_indices = np.random.randint(0, len(X_train), batch_size)
    X_batch = X_train[rnd_indices]

    Y_batch = Y_train[rnd_indices]

    return X_batch, Y_batch


def create_model_single_layer(input_features, input_dim, label_dim):
    # 创建单层神经网络
    weight = tf.Variable(tf.random_normal([input_dim, label_dim]), name="weight")
    bias = tf.Variable(tf.zeros([label_dim]), name="bias")
    output = tf.add(tf.matmul(input_features, weight), bias)
    tf.add_to_collection("output", output)

    return output


def create_model_multiple_layer(input_features, input_dim, label_dim):
    # 创建多层神经网络
    hidden_1 = 256
    hidden_2 = 256
    weights = {
        "h1": tf.Variable(tf.random_normal([input_dim, hidden_1]), name="h1"),
        "h2": tf.Variable(tf.random_normal([hidden_1, hidden_2]), name="h2"),
        "out": tf.Variable(tf.random_normal([hidden_2, label_dim]), name="weights_out")
    }
    biases = {
        "b1": tf.Variable(tf.random_normal([hidden_1]), name="b1"),
        "b2": tf.Variable(tf.random_normal([hidden_2]), name="b2"),
        "out": tf.Variable(tf.random_normal([label_dim]), name="biases_out")
    }

    layer_1 = tf.add(tf.matmul(input_features, weights["h1"]), biases["b1"])
    layer_1 = tf.nn.relu(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1, weights["h2"]), biases["b2"])
    layer_2 = tf.nn.relu(layer_2)

    out_layer = tf.add(tf.matmul(layer_2, weights["out"]), biases["out"])  
    tf.add_to_collection("output", out_layer)

    return out_layer


def train():
    # 训练模型
    X, Y = load_data(filename, input_dim)    # 加载数据集

    # 生成测试集和训练集
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=0)

    # 保存测试集和训练集以便后续预测分析
    with open("X_test.txt", "w") as fr:
        for i in range(len(X_test)):
            for j in range(len(X_test[i])):
                fr.write(str(X_test[i][j]) + '\t')    
            fr.write('\n')    

    with open("Y_test.txt", "w") as fr:
        for i in range(len(Y_test)):
            for j in range(len(Y_test[i])):
                fr.write(str(Y_test[i][j]) + '\t')    
            fr.write('\n')

    input_features = tf.placeholder(tf.float32, [None, input_dim], name="input_features")
    input_labels = tf.placeholder(tf.float32, [None, label_dim], name="input_labels") 

    # 模型输出
    output = create_model_single_layer(input_features, input_dim, label_dim)   # 单层神经网络
    # output = create_model_multiple_layer(input_features, input_dim, label_dim)   # 多层神经网络

    # 计算错误率
    predict_value = tf.argmax(tf.nn.softmax(output), axis=1)
    true_value = tf.argmax(input_labels, axis=1)
    error = tf.count_nonzero(predict_value - true_value)

    # 计算损失
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=input_labels, logits=output)
    loss = tf.reduce_mean(cross_entropy)

    # 优化器
    optimizer = tf.train.AdamOptimizer(0.04)
    train = optimizer.minimize(loss)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        total_batch = np.int32(len(X_train) / minibatch_size)
        # 迭代训练
        for epoch in range(max_epochs):
            sum_err = 0
            for _ in range(total_batch):                   
                x, y = random_batch(X_train, Y_train, minibatch_size)

                _, lossval, _, errval = sess.run([train, loss, output, error], feed_dict={input_features: x, input_labels: y})
                sum_err += (errval / minibatch_size)
            # 打印信息
            print("epoch: ", epoch, "loss: ", "{:.4f}".format(lossval), "error: ", "{:.4f}".format(sum_err / total_batch))

            # 保存模型
            if epoch == max_epochs - 1:  
                checkpoint_path = os.path.join(logs_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=epoch)


def predict(X_test): 
    # 预测模型  
    with tf.Session() as sess:
        # 加载模型
        ckpt = tf.train.get_checkpoint_state(logs_dir)
        saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
        saver.restore(sess, ckpt.model_checkpoint_path)
        graph = tf.get_default_graph()

        # 导出训练时保存的名字
        input_features = graph.get_operation_by_name('input_features').outputs[0]
        output = tf.get_collection("output")[0]

        prediction = sess.run(output, feed_dict={input_features: X_test})
        # 选取最大值为预测结果
        result = np.argmax(prediction)   
        print('result: ', result)  


if __name__ == "__main__":    
    train()   

    '''
    # 打开测试集数据
    X_tests = []   
    with open("X_test.txt", "r") as fr:
        for line in fr.readlines():    
            curLine = line.strip().split("\t")    
            floatLine = np.matrix(list(map(float, curLine)), dtype="float32")  
            X_tests.append(floatLine) 

    predict(X_tests[0])      # 选择测试哪个，并和Y_test.txt对照
    '''
