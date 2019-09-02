# My_Example_TF

# 平台
&emsp;&emsp;Windows 10    
&emsp;&emsp;Tensorflow 1.14    
&emsp;&emsp;python 3.5.4   

## Simple Network
&emsp;&emsp;利用单层和多层`BP`神经网络对文本数据集分类。

### 文件
&emsp;&emsp;data.txt      
&emsp;&emsp;simple_network.py 

### 数据集
&emsp;&emsp;首先确保数据集的格式和`data.txt`中的格式一致，即特征+标签。如果不一致，需要更改`simple_network.py`中的`load_data()`函数。其最终返回值为特征和`one-hot`编码的标签。    

### 模型
&emsp;&emsp;一共有2个模型，即`create_model_single_layer()`和`create_model_multiple_layer`分别为单层和多层`BP`神经网络模型。     

### 训练
&emsp;&emsp;首先在最上面的基本配置栏中，将信息更改为自己的，并在`train()`中模型输出中选择模型，最后在`main`函数中打开`train()`，开始训练。  
&emsp;&emsp;训练结束，会在文件夹内生成`X_test.txt`，`Y_test.txt`和`logs_1`文件夹，分别为测试集特征，测试集标签和模型日志文件。   

### 预测
&emsp;&emsp;在`main`中关闭训练函数，打开测试函数，并选择测试集索引即可。    

## CNN
&emsp;&emsp;利用CNN对图片数据集分类。 

### 文件  
&emsp;&emsp;dataSet   
&emsp;&emsp;create_labels_files.py       
&emsp;&emsp;create_tf_record.py    
&emsp;&emsp;info.yml      
&emsp;&emsp;model.py    
&emsp;&emsp;train.py   
&emsp;&emsp;predict.py     

### 配置
&emsp;&emsp;首先打开`info.yml`，更改里面的配置信息。 

### 数据集
&emsp;&emsp;数据集采用`tfrecords`文件格式，首先将自己的图片放入`dataSet`文件夹中，然后打开`create_labels_files.py`文件，更改`labels`信息，然后运行即可。最后再打开`create_tf_record.py`生成`tfrecords`文件。  

### 模型
&emsp;&emsp;采用普通的卷积，池化和全连接层。    

### 训练
&emsp;&emsp;有2个训练函数，即`my_train_queue()`和`my_train_tfData()`，分别为采用队列和`tfData`读取数据。    

### 预测
&emsp;&emsp;预测的函数对应于训练的函数。选择相应的函数运行即可。    

