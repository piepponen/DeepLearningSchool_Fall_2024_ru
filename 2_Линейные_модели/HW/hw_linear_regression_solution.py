#Заполните пропуски в функциях grad_f и grad_descent_2d

import numpy as np


def f(x):
    """
    :param x: np.array(np.float) размерности 2
    :return: float
    """
    return np.sum(np.sin(x) ** 2, axis=0)


def grad_f(x):
    """
    Градиент функции f, определенной выше.
    :param x: np.array[2]: float вектор длины 2
    :return: np.array[2]: float вектор длины 2
    """
    return np.sin(2 * x)  # производная sin^2(x) = sin(2x)


def grad_descent_2d(f, grad_f, lr, num_iter=100, x0=None):
    """
    Функция, которая реализует градиентный спуск в минимум для функции f от двух переменных.
        :param f: скалярная функция двух переменных
        :param grad_f: функция, возвращающая градиент функции f (устроена как реализованная выше grad_f)
        :param lr: learning rate алгоритма
        :param num_iter: количество итераций градиентного спуска
        :return: np.array[num_iter, 2] пар вида (x, f(x))
    """
    if x0 is None:
        x0 = np.random.random(2)

    # будем сохранять значения аргументов и значений функции
    # в процессе град. спуска в переменную history
    history = []

    # итерация цикла -- шаг градиентного спуска
    curr_x = x0.copy()
    for iter_num in range(num_iter):
        entry = np.hstack((curr_x, f(curr_x)))
        history.append(entry)

        # обновляем x с помощью градиентного шага
        curr_x -= lr * grad_f(curr_x)

    return np.vstack(history)
#Заполните пропуски в функции generate_batches

def generate_batches(X, y, batch_size):
    """
    param X: np.array[n_objects, n_features] --- матрица объекты-признаки
    param y: np.array[n_objects] --- вектор целевых переменных
    """
    assert len(X) == len(y)
    np.random.seed(42)
    X = np.array(X)
    y = np.array(y)

    # перемешиваем индексы данных
    perm = np.random.permutation(len(X))

    # итерируем по индексам с шагом batch_size
    for batch_start in range(0, len(X), batch_size):
        # выбираем индексы для текущего батча
        batch_indices = perm[batch_start:batch_start + batch_size]

        # проверяем, чтобы размер батча соответствовал batch_size
        if len(batch_indices) == batch_size:
            yield X[batch_indices], y[batch_indices]
#Реализуйте методы fit и get_grad класса MyLogisticRegression.
#Напоминаем формулы:
#Функцию generate_batches, которую нужно использовать внутри .fit(),  мы уже реализовали за вас.
def logit(x, w):
    return np.dot(x, w)

def sigmoid(h):
    return 1. / (1 + np.exp(-h))

class MyLogisticRegression(object):
    def __init__(self):
        self.w = None

    def fit(self, X, y, epochs=10, lr=0.1, batch_size=100):
        n, k = X.shape
        if self.w is None:
            np.random.seed(42)
            # Вектор столбец в качестве весов
            self.w = np.random.randn(k + 1)

        # Добавляем вектор единиц для BIAS
        X_train = np.concatenate((np.ones((n, 1)), X), axis=1)

        losses = []

        for i in range(epochs):
            for X_batch, y_batch in generate_batches(X_train, y, batch_size):
                # Предсказываем вероятности для батча
                predictions = self._predict_proba_internal(X_batch)

                # Вычисляем лосс
                loss = self.__loss(y_batch, predictions)
                assert (np.array(loss).shape == tuple()), "Лосс должен быть скаляром!"

                losses.append(loss)

                # Обновляем веса с использованием градиента
                grad = self.get_grad(X_batch, y_batch, predictions)
                self.w -= lr * grad

        return losses

    def get_grad(self, X_batch, y_batch, predictions):
        """
        param X_batch: np.array[batch_size, n_features + 1] --- матрица объекты-признаки (с добавленным bias)
        param y_batch: np.array[batch_size] --- батч целевых переменных
        param predictions: np.array[batch_size] --- батч вероятностей классов
        """
        # Градиент логистической регрессии для всей партии данных
        errors = predictions - y_batch  # разница между предсказаниями и истинными метками
        grad_basic = X_batch.T @ errors  # сумма градиентов по всем объектам батча
        assert grad_basic.shape == (X_batch.shape[1],), "Градиенты должны быть столбцом из k_features + 1 элементов"

        return grad_basic

    def predict_proba(self, X):
        n, k = X.shape
        X_ = np.concatenate((np.ones((n, 1)), X), axis=1)
        return sigmoid(logit(X_, self.w))

    def _predict_proba_internal(self, X):
        return sigmoid(logit(X, self.w))

    def predict(self, X, threshold=0.5):
        return self.predict_proba(X) >= threshold

    def get_weights(self):
        return self.w.copy()

    def __loss(self, y, p):
        # Обрезаем вероятности, чтобы избежать логарифмов от 0
        p = np.clip(p, 1e-10, 1 - 1e-10)
        return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
#Реализуйте класс логистической регрессии с обеими регуляризациями и оптимизацией с помощью SGD.
#Обратите внимание, что реализация ElasticNet отличается от реализации
# LogisticRegression только функцией потерь для оптимизации.
# Поэтому единственная функция, которая будет отличаться у двух методов, это self.get_grad.
#Поэтому в данном случае естественно применить паттерн наследования.
# Весь синтаксис наследования мы прописали за вас.
# Единственное, что вам осталось сделать, это переопределить метод get_grad
# в отнаследованном классе MyElasticLogisticRegression.
class MyElasticLogisticRegression(MyLogisticRegression):
    def __init__(self, l1_coef, l2_coef):
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.w = None

    def get_grad(self, X_batch, y_batch, predictions):
        """
        Принимает на вход X_batch с уже добавленной колонкой единиц.
        Выдаёт градиент функции потерь в логистической регрессии с регуляризаторами
        как сумму градиентов функции потерь на всех объектах батча + регуляризационное слагаемое
        """

        # Основной градиент логистической регрессии (без регуляризации)
        errors = predictions - y_batch  # ошибки
        grad_basic = X_batch.T @ errors  # основной градиент
        # Обратите внимание: компонент bias в grad_basic уже вычислен

        # Градиент для L1-регуляризации (включаем все веса, кроме bias)
        grad_l1 = self.l1_coef * np.sign(self.w)
        grad_l1[0] = 0  # не включаем bias в регуляризацию

        # Градиент для L2-регуляризации (включаем все веса, кроме bias)
        grad_l2 = 2 * self.l2_coef * self.w
        grad_l2[0] = 0  # не включаем bias в регуляризацию

        # Проверяем, что bias не входит в регуляризационные слагаемые
        assert grad_l1[0] == grad_l2[0] == 0, "Bias в регуляризационные слагаемые не входит!"
        assert grad_basic.shape == grad_l1.shape == grad_l2.shape == (X_batch.shape[1],), "Градиенты должны быть столбцом из k_features + 1 элементов"

        # Возвращаем сумму всех градиентов
        return grad_basic + grad_l1 + grad_l2
