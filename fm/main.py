import tensorflow as tf
import numpy as np
from scipy.sparse import csr
import pandas as pd
from tqdm import tqdm

from model.fm_model import FM


def dic2matrix(dic, ix=None, p=None, n=0, g=0):
    """
    将字典转化为FM所需的矩阵
    :param dic: 一个含有多个特征的字典
    :param ix: 下标的生成字典
    :param n: 记录条数
    :param g: 特征种类
    :return: 一个二值矩阵
    """
    if ix is None:
        ix = dict()

    # 矩阵中1的个数
    nz = n * g

    col_ix = np.empty(nz, dtype=int)

    # 勾选特征
    i = 0
    for k, lis in dic.items():
        for t in range(len(lis)):
            ix[str(lis[t]) + str(k)] = ix.get(str(lis[t]) + str(k), 0) + 1
            col_ix[i + g * t] = (lis[t] - 1) * g + i
        i += 1

    # 特征空间，即矩阵的列数
    if p is None:
        p = np.max(col_ix) + 1

    row_ix = np.repeat(np.arange(0, n), g)
    data = np.ones(nz)
    ixx = np.where(col_ix < p)
    return csr.csr_matrix((data[ixx], (row_ix[ixx], col_ix[ixx])), shape=(n, p)), ix


def main():
    cols = ['user', 'item', 'rating', 'timestamp']

    train = pd.read_csv('data/ua.base', delimiter='\t', names=cols)
    test = pd.read_csv('data/ua.test', delimiter='\t', names=cols)

    x_train, ix = dic2matrix({'users': train['user'].values,
                              'items': train['item'].values}, n=len(train.index), g=2)

    x_test, ix = dic2matrix({'users': test['user'].values,
                             'items': test['item'].values}, ix, x_train.shape[1], n=len(test.index), g=2)

    y_train = train['rating'].values.astype(np.float32)
    y_test = test['rating'].values.astype(np.float32)

    x_train = x_train.todense().astype(np.float32)
    x_test = x_test.todense().astype(np.float32)

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(1000)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(1000)

    model = FM(10, x_train.shape[1])
    loss_object = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam()

    train_loss = tf.keras.metrics.Mean(name='train_loss')

    test_loss = tf.keras.metrics.Mean(name='test_loss')

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            predictions = model(x)
            loss = loss_object(y, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)

    @tf.function
    def test_step(x, y):
        predictions = model(x)
        t_loss = loss_object(y, predictions)

        test_loss(t_loss)

    EPOCH = 100

    for epoch in tqdm(range(EPOCH)):
        for x, y in train_ds:
            train_step(x, y)

        for test_x, test_y in test_ds:
            test_step(test_x, test_y)

        print(
            f'Epoch {epoch + 1}, '
            f'Loss: {train_loss.result()}, '
            f'Test Loss: {test_loss.result()}, ')


if __name__ == '__main__':
    main()
