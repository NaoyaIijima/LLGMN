from My_LLGMN import My_LLGMN
import numpy as np


# 学習データの作成（2変数，3クラスのデータを計100サンプル）
C = 2
d = 2
N = 100
x_train = np.random.randn(N, d)
x_train[0: int(N / C), :] += 2
# x_train[int(N/C): int(2*N/C), :] -= 3

y_train = np.ones((N, 1))
y_train[: int(N / C), :] -= 1.0
# y_train[int(N/C): int(2*N/C), :] += 1.0

# テストデータの作成（10サンプル，すべてクラス0に近いものを生成）
x_test = np.random.randn(10, d) + 2


my_llgmn = My_LLGMN(n_class=C, input_dim=d, n_component=1)
my_llgmn.build_network()
my_llgmn.learn(x_train=x_train, y_train=y_train,
               is_mini_batch=False, n_epoch=50)
pred = my_llgmn.predict(x_test=x_test)
print(np.argmax(pred, axis=1))
print(y_train)
