# -*- coding: utf-8 -*-

from keras.models import Model
from keras.layers import Dense, Activation, Input, concatenate
from keras.utils import to_categorical
from keras.optimizers import SGD
from keras import metrics
import numpy as np


class My_LLGMN():
    def __init__(self, n_class, input_dim, n_component):
        self.n_class = n_class
        self.n_component = n_component
        self.h = int(1 + input_dim * (input_dim + 3) / 2)

    def build_network(self):
        main_input = Input(shape=(self.h,), dtype="float32", name="main_input")

        layer = Dense(self.n_class * self.n_component - 1, use_bias=False)
        middle_layer_ = layer(main_input)

        auxiliary_input = Input(shape=(1,), name="aux_input")
        middle_layer = concatenate([middle_layer_, auxiliary_input])

        acctivate = Activation("softmax")
        middle_layer2 = acctivate(middle_layer)

        layer = Dense(self.n_class, use_bias=False, name="main_output")
        layer.trainable = False
        main_output = layer(middle_layer2)

        self.model = Model(
            inputs=[main_input, auxiliary_input], outputs=[main_output])

        self.model.summary()

        # 同じクラスの各コンポーネント成分の総和を実現するための重みを作成
        # 学習はしないで固定する (trainable=False)
        weights = self.model.get_weights()
        w2 = np.zeros(weights[1].shape)
        for c in range(self.n_class):
            for j in range(self.n_component):
                w2[self.n_component * c + j, c] = 1
        weights[1] = w2
        self.model.set_weights(weights=weights)

    def convert_labels(self, labels):
        labels = labels.astype(int)
        return to_categorical(labels, self.n_class)

    def nonlinear_transform(self, data):
        d = data.copy()
        tmp = np.zeros([1, d.shape[0]])
        for i in range(d.shape[1]):
            for j in range(i, d.shape[1]):
                tmp = np.vstack([tmp, d[:, i] * d[:, j]])
        _ = np.ones([d.shape[0], 1])
        d = np.hstack([_, d])
        d = np.hstack([d, tmp.T[:, 1:]])
        return d

    def learn(
        self, x_train, y_train, is_mini_batch=False, n_batch_size=128, n_epoch=30
    ):

        # データの下処理
        nonlinear_x_train = self.nonlinear_transform(x_train)
        aux_train = np.zeros([x_train.shape[0], 1])
        y_train = self.convert_labels(y_train)

        sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

        # 2クラス or 多クラス識別でLossを変えている
        if self.n_class == 2:
            self.model.compile(
                optimizer=sgd, loss="binary_crossentropy", metrics=[metrics.binary_accuracy]
            )
        else:
            self.model.compile(
                optimizer=sgd,
                loss="categorical_crossentropy",
                metrics=[metrics.categorical_accuracy],
            )

        if is_mini_batch is True:
            self.model.fit(
                {"main_input": nonlinear_x_train, "aux_input": aux_train},
                {"main_output": y_train},
                epochs=n_epoch,
                batch_size=n_batch_size,
                verbose=1,  # 0:学習過程を非表示，1:表示
            )
        else:
            self.model.fit(
                {"main_input": nonlinear_x_train, "aux_input": aux_train},
                {"main_output": y_train},
                epochs=n_epoch,
                batch_size=x_train.shape[1],
                verbose=1,  # 0:学習過程を非表示，1:表示
            )

    def predict(self, x_test):
        nonlinear_x_test = self.nonlinear_transform(x_test)
        aux_test = np.zeros([x_test.shape[0], 1])
        pred = self.model.predict(
            {"main_input": nonlinear_x_test, "aux_input": aux_test})
        return pred
