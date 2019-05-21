# -*- coding: utf-8 -*-

from sklearn.linear_model import LogisticRegression
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.externals import joblib
from sklearn.cluster import KMeans
from sklearn import svm, tree
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report  # 生产报告
from sklearn.metrics import confusion_matrix

from sklearn import metrics
from sklearn import neighbors
import tensorflow as tf

from config import model_path,part12_data_path


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


# 获取MNIST数据集
mnist = input_data.read_data_sets(part12_data_path, one_hot=False)

y_train1 = []
y_test1 = []
x_train, y_train = mnist.train.next_batch(50000)

x_test, y_test = mnist.test.next_batch(2000)


#
# LogisticRegression
def logistic():
    #
    # clf = LogisticRegression()
    # clf.fit(x_train, y_train)
    # joblib.dump(clf, "models/logistic.m")

    # 导入模型
    clf = joblib.load(model_path + "/logistic.m")
    y_pred = clf.predict(x_test)
    sum = 0.0
    for i in range(2000):
        if (y_pred[i] == y_test[i]):
            sum = sum + 1

    print('LogisticRegression Test set score: %f' % (sum / 2000.))

    fpr0, tpr0, threshold0 = roc_curve(y_test, y_pred,
                                       pos_label=0)  ###计算真正率和假正率
    roc_auc0 = auc(fpr0, tpr0)  ###计算auc的值
    fpr1, tpr1, threshold1 = roc_curve(y_test, y_pred,
                                       pos_label=1)  ###计算真正率和假正率
    roc_auc1 = auc(fpr1, tpr1)  ###计算auc的值
    fpr2, tpr2, threshold2 = roc_curve(y_test, y_pred,
                                       pos_label=2)  ###计算真正率和假正率
    roc_auc2 = auc(fpr2, tpr2)  ###计算auc的值
    fpr3, tpr3, threshold3 = roc_curve(y_test, y_pred,
                                       pos_label=3)  ###计算真正率和假正率
    roc_auc3 = auc(fpr3, tpr3)  ###计算auc的值
    fpr4, tpr4, threshold4 = roc_curve(y_test, y_pred,
                                       pos_label=4)  ###计算真正率和假正率
    roc_auc4 = auc(fpr4, tpr4)  ###计算auc的值
    fpr5, tpr5, threshold5 = roc_curve(y_test, y_pred,
                                       pos_label=5)  ###计算真正率和假正率
    roc_auc5 = auc(fpr5, tpr5)  ###计算auc的值
    fpr6, tpr6, threshold6 = roc_curve(y_test, y_pred,
                                       pos_label=6)  ###计算真正率和假正率
    roc_auc6 = auc(fpr6, tpr6)  ###计算auc的值
    fpr7, tpr7, threshold7 = roc_curve(y_test, y_pred,
                                       pos_label=7)  ###计算真正率和假正率
    roc_auc7 = auc(fpr7, tpr7)  ###计算auc的值
    fpr8, tpr8, threshold8 = roc_curve(y_test, y_pred,
                                       pos_label=8)  ###计算真正率和假正率
    roc_auc8 = auc(fpr8, tpr8)  ###计算auc的值
    fpr9, tpr9, threshold9 = roc_curve(y_test, y_pred,
                                       pos_label=9)  ###计算真正率和假正率
    roc_auc9 = auc(fpr9, tpr9)  ###计算auc的值

    fpr = (
              fpr0 + fpr1 + fpr2 + fpr3 + fpr4 + fpr5 + fpr6 + fpr7 + fpr8 + fpr9) / 10
    tpr = (
              tpr0 + tpr1 + tpr2 + tpr3 + tpr4 + tpr5 + tpr6 + tpr7 + tpr8 + tpr9) / 10
    roc_auc = (
                  roc_auc0 + roc_auc1 + roc_auc2 + roc_auc3 + roc_auc4 + roc_auc5 + roc_auc6 + roc_auc7 + roc_auc8 + roc_auc9) / 10
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw,
             label='ROC curve (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


# knn
def deal_knn():  # KNN
    # 定义模型
    # knn_model = neighbors.KNeighborsClassifier()
    # knn_model.fit(x_train, y_train)
    # joblib.dump(knn_model, "models/knn.m")

    # 导入模型
    knn_model = joblib.load(model_path + "/knn.m")
    y_pred1 = knn_model.predict(x_test)
    sum1 = 0.0
    for i in range(2000):
        if (y_pred1[i] == y_test[i]):
            sum1 = sum1 + 1
    print('KNeighborsClassifier Test set score: %f' % (sum1 / 2000.))


# kmeans
def deal_kmeans():
    # 定义模型
    # kmeans = KMeans(n_clusters=10)
    # kmeans.fit(x_train)
    # joblib.dump(kmeans, "models/kmeans.m")

    # 导入模型
    kmeans = joblib.load(model_path + "/kmeans.m")
    y_pred2 = kmeans.predict(x_test)
    print (
    'KMeans Test set score:', metrics.adjusted_rand_score(y_test, y_pred2))


def deal_svm():
    # 定义模型
    # svc = svm.SVC(C=100.0, kernel='rbf', gamma=0.03)
    # svc.fit(x_train, y_train)
    # joblib.dump(svc, "models/svm.m")

    # 导入模型
    svc = joblib.load(model_path + "/svm.m")

    y_pred3 = svc.predict(x_test)
    sum2 = 0.0
    for i in range(2000):
        if (y_pred3[i] == y_test[i]):
            sum2 = sum2 + 1
    print('svm Test set score: %f' % (sum2 / 2000.))

    fpr0, tpr0, threshold0 = roc_curve(y_test, y_pred3,
                                       pos_label=0)  ###计算真正率和假正率
    roc_auc0 = auc(fpr0, tpr0)  ###计算auc的值
    fpr1, tpr1, threshold1 = roc_curve(y_test, y_pred3,
                                       pos_label=1)  ###计算真正率和假正率
    roc_auc1 = auc(fpr1, tpr1)  ###计算auc的值
    fpr2, tpr2, threshold2 = roc_curve(y_test, y_pred3,
                                       pos_label=2)  ###计算真正率和假正率
    roc_auc2 = auc(fpr2, tpr2)  ###计算auc的值
    fpr3, tpr3, threshold3 = roc_curve(y_test, y_pred3,
                                       pos_label=3)  ###计算真正率和假正率
    roc_auc3 = auc(fpr3, tpr3)  ###计算auc的值
    fpr4, tpr4, threshold4 = roc_curve(y_test, y_pred3,
                                       pos_label=4)  ###计算真正率和假正率
    roc_auc4 = auc(fpr4, tpr4)  ###计算auc的值
    fpr5, tpr5, threshold5 = roc_curve(y_test, y_pred3,
                                       pos_label=5)  ###计算真正率和假正率
    roc_auc5 = auc(fpr5, tpr5)  ###计算auc的值
    fpr6, tpr6, threshold6 = roc_curve(y_test, y_pred3,
                                       pos_label=6)  ###计算真正率和假正率
    roc_auc6 = auc(fpr6, tpr6)  ###计算auc的值
    fpr7, tpr7, threshold7 = roc_curve(y_test, y_pred3,
                                       pos_label=7)  ###计算真正率和假正率
    roc_auc7 = auc(fpr7, tpr7)  ###计算auc的值
    fpr8, tpr8, threshold8 = roc_curve(y_test, y_pred3,
                                       pos_label=8)  ###计算真正率和假正率
    roc_auc8 = auc(fpr8, tpr8)  ###计算auc的值
    fpr9, tpr9, threshold9 = roc_curve(y_test, y_pred3,
                                       pos_label=9)  ###计算真正率和假正率
    roc_auc9 = auc(fpr9, tpr9)  ###计算auc的值

    fpr = (
              fpr0 + fpr1 + fpr2 + fpr3 + fpr4 + fpr5 + fpr6 + fpr7 + fpr8 + fpr9) / 10
    tpr = (
              tpr0 + tpr1 + tpr2 + tpr3 + tpr4 + tpr5 + tpr6 + tpr7 + tpr8 + tpr9) / 10
    roc_auc = (
                  roc_auc0 + roc_auc1 + roc_auc2 + roc_auc3 + roc_auc4 + roc_auc5 + roc_auc6 + roc_auc7 + roc_auc8 + roc_auc9) / 10
    lw = 2
    plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw,
             label='ROC curve (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


def deal_tree():
    # 定义模型
    # Tree = tree.DecisionTreeClassifier()
    # Tree.fit(x_train, y_train)
    # joblib.dump(Tree, "models/tree.m")

    # 导入模型
    Tree = joblib.load(model_path + "/tree.m")
    y_pred4 = Tree.predict(x_test)
    sum3 = 0.0
    for i in range(2000):
        if (y_pred4[i] == y_test[i]):
            sum3 = sum3 + 1
    print('Tree Test set score: %f' % (sum3 / 2000.))


def deal_random_forest_classifier():
    # 定义模型
    # rfc = RandomForestClassifier(n_jobs=-1)
    # rfc.fit(x_train, y_train)
    # joblib.dump(rfc, "models/random_forest.m")

    # 导入模型
    rfc = joblib.load(model_path + "/random_forest.m")
    y_pred5 = rfc.predict(x_test)
    report = classification_report(y_test, y_pred5)
    # confusion_mat = confusion_matrix(y_test1, y_pred5)
    print(report)
    sum4 = 0.0
    for i in range(2000):
        if (y_pred5[i] == y_test[i]):
            sum4 = sum4 + 1
    print(' RandomForestClassifier Test set score: %f' % (sum4 / 2000.))


def deal_mnb():
    # 定义模型
    # mnb = MultinomialNB()  # 使用默认配置初始化朴素贝叶斯
    # mnb.fit(x_train, y_train)  # 利用训练数据对模型参数进行估计
    # joblib.dump(mnb, "models/mnb.m")

    # 导入模型
    mnb = joblib.load(model_path + "/mnb.m")
    y_pred6 = mnb.predict(x_test)  # 对参数进行预测

    sum5 = 0.0
    for i in range(2000):
        if (y_pred6[i] == y_test[i]):
            sum5 = sum5 + 1
    print('MultinomialNB Test set score: %f' % (sum5 / 2000.))


def deal_mlp():
    # 定义模型
    # MLP = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(30, 20),
    #                     random_state=1)
    # MLP.fit(x_train, y_train)
    # joblib.dump(MLP, "models/mlp.m")

    # 导入模型
    MLP = joblib.load(model_path + "/mlp.m")
    y_pred7 = MLP.predict(x_test)
    sum6 = 0.0
    for i in range(2000):
        if (y_pred7[i] == y_test[i]):
            sum6 = sum6 + 1
    print('MLPClassifier Test set score: %f' % (sum6 / 2000.))


if __name__ == "__main__":
    print("logistic")
    logistic()
    print("knn")
    deal_knn()
    print("kmeans")
    deal_kmeans()
    print("svm")
    deal_svm()
    print("tree")
    deal_tree()
    print("forest")
    deal_random_forest_classifier()
    print("mnb")
    deal_mnb()
    print("mlp")
    deal_mlp()
