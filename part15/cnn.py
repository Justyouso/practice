# -*- coding: utf-8 -*-
# @Time: 19-4-18 下午4:23

from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from keras.models import load_model
from config import model_path

# 加载数据
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 网络架构
# network = models.Sequential()
# network.add(layers.Dense(512, activation="relu", input_shape=(28 * 28,)))
# network.add(layers.Dense(10, activation="softmax"))
#
# # 编译训练模型
# network.compile(optimizer="rmsprop", loss="categorical_crossentropy",
#                 metrics=["accuracy"])
#
# # 准备图像数据(图像28*28,取值区间[0-1])
# train_images = train_images.reshape((60000, 28*28))
# train_images = train_images.astype("float32") / 255
#
# test_images = test_images.reshape((10000, 28*28))
# test_images = test_images.astype("float32") / 255
#
# # 准备标签
# train_labels = to_categorical(train_labels)
# test_labels = to_categorical(test_labels)
#
# # 训练模型
# history =network.fit(train_images, train_labels, epochs=5, batch_size=128,
#             validation_data=(test_images, test_labels))
#
#
#
# # 取出精度
# history_dict = history.history
#
# loss_values = history_dict["loss"]
# val_loss_values = history_dict["val_loss"]
# epochs = range(1,len(loss_values)+1)
#
# # 绘制训练损失和验证损失
# plt.plot(epochs,loss_values,'bo',label='Training loss')
# plt.plot(epochs,val_loss_values,'b',label='Validation loss')
# plt.title('Training and Validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
#
# # 清空图像
# plt.clf()
# acc = history_dict['acc']
# val_acc = history_dict['val_acc']
# plt.plot(epochs,acc,'bo',label='Training acc')
# plt.plot(epochs,val_acc,'b',label='Validation acc')
# plt.title('Training and Validation Acc')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()
# 测试模型

# 导入模型
model = load_model(model_path+"/cnn.m")


# 取出test_images中的第一个图像做测试

# np.array将数据转化为数组 np.reshape将一维数组reshape成(28*28)  mnist.train.images[1]取出第二张图片 dtype转换为int8数据类型
im_data = np.array(np.reshape(test_images[0], (28, 28)) * 255, dtype=np.int8)  # 取第一张图片的 数组
print(im_data)
# 将数组还原成图片 Image.fromarray方法 传入数组 和 通道
img = Image.fromarray(im_data, 'L')
img.save('1.jpg')
img.show()  # 显示图片

# 拿对应的标签
arr_data = test_labels[0]
print(arr_data)  # one-hot形式

# 将数组转换成(1,784,)的形状
test= np.reshape(test_images[0],(1,784,))
result = model.predict(test)

# 解码one_hot(解码成数字)
k=np.argmax(result,axis=1)
print(k)

# test_loss, test_acc = network.evaluate(test_images, test_labels)