---
title: TensorFlow RNN入门
date: 2017-10-30 17:00:00
tags: tensorflow
categories: tensorflow
---

循环神经网络, Recurrent neural network, RNN
<!-- more -->

# RNN简介

起源于1982年的霍普菲尔德网络。

RNN的主要用途是处理和预测序列数据。应用于语音识别、语言模型、机器翻译以及时序分析等问题。

CNN中隐藏层中的节点是无连接的。RNN隐藏层的输入不仅包括输入层的输出，也包括上一时刻隐藏层的输出。

## 前向传播

损失函数为所有时刻上损失函数的总和。

使用numpy库模拟前向传播

示意图：

![img]()

代码：

```python
import numpy as np

# 不同时刻的输入，此处为2个时刻
X = [1, 2]
# 初始的状态，即上一个循环体的输出
state = [0.0, 0.0]

# 循环体参数state为state的系数，input为X的系数
w_cell_state = np.asarray([[0.1, 0.2], [0.3, 0.4]])
w_cell_input = np.asarray([0.5, 0.6])
b_cell = np.asarray([0.1, -0.1])

# 用于当前时刻从循环体输出的参数
w_output = np.asarray([[1.0], [2.0]])
b_output = 0.1

for i in range(len(X)):
    before_activation = np.dot(state, w_cell_state) + \
        X[i] * w_cell_input + b_cell
    state = np.tanh(before_activation)
    final_output = np.dot(state, w_output) + b_output
    print("before activation: ", before_activation)
    print("state: ", state)
    print("output: ", final_output)
```

> 当循环体过多时，会发生梯度消失问题。

# LTSM

RNN中一个重要结构：长短时记忆网络, long short-term memory, LSTM

1997年提出，为了解决长期依赖问题(long-term dependencies)

LSTM是一种特殊的循环体结构，拥有三个门：输入门、遗忘门、输出门。

之所以是门，是因为使用了全连接网络+sigmoid激活函数，通过[0,1]内的输出来决定有多少信息可以通过这个门。

遗忘门的作用是使得RNN忘记之前没有用的信息，遗忘门根据当前的输入x(t)、上一时刻的状态c(t-1)和上一时刻的输出h(t-1)来决定哪一部分记忆需要被遗忘。

输入门的作用是补充最新的记忆，输入们同样通过当前的输入x(t)、上一时刻的状态c(t-1)和上一时刻的输出h(t-1)来决定哪一部分当前信息进入这一时刻的状态中。

通过输入门和遗忘门，LSTM结构可以更加有效的决定哪些信息应该被遗忘，哪些信息应该得到保留，即得到当前状态c(t)。

输出门会根据当前的输入x(t)、这一时刻的状态c(t)和上一时刻的输出h(t-1)来决定这一时刻的输出h(t)

伪代码如下：

```python
lstm = rnn_cell.BasicLSTMCell(lstm_hidden_size)

state = lstm.zero_state(batch_size, tf.float32)

loss = 0.0
for i in range(num_steps):
    # 在第一个时刻申明的LSTM结构中使用的变量，后续要重复利用
    if i > 0: tf.get_variable_scrope().reuse_variables()

    # 每一步处理一个时刻，将当前输入和前一时刻状态传入定义的LSTM
    lstm_output, state = lstm(current_input, state)
    final_output = fully_connected(lstm_output)

    loss += calc_loss(final_output, expected_output)
```

# RNN的变种

## 双向RNN和深层RNN

双向循环神经网络bidirectional RNN，使得当前时刻的输出不仅与上一时刻的状态有关，也与后一时刻的状态有关。使用2个RNN进行组合。

![img]()

深层循环神经网络deepRNN，将每一时刻的循环体重复多次。不同层的参数不同，而不同时刻同一层的参数相同。

```python
lstm = rnn_cell.BasicLSTMCell(lstm_size)

stacked_lstm = rnn_cell.MultiRNNCell([lstm]*number_of_layers)

state = stacked_lstm.zero_state(batch_size, tf.float32)

for i in range(len(num_steps)):
    if i > 0: tf.get_variable_scrope().reuse_variables()
    stacked_lstm_output, state = stacked_lstm(current_input, state)
    final_output = fully_connected(stacked_lstm_output)
    loss += cal_loss(final_output, expected_output)
```

## RNN的Dropout

![img]()

```python
lstm = rnn_cell.BasicLSTMCell(lstm_size)

dropout_lstm = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=0.5)

stacked_lstm = rnn_cell.MultiRNNCell([dropout_lstm] * number_of_layers)
```

# RNN样例应用

## 自然语言建模

一个句子看作是一个单词的序列，一个句子出现的概率为

```
p(S)=p(w1,w2,w3,...,wm)
    =p(w1)p(w2|w1)p(w3|w1,w2)...p(wm|w1,w2,...,wm-1)
```

以上的每一个p均为语言模型的一个参数，为了估计参数取值，常见方法有：n-gram方法、决策树、最大熵模型、条件随机场、神经网络语言模型等等。

n-gram中的n一般取1，2，3，分别称为unigram、bigram（常用）、trigram。

语言模型的评价指标为复杂度（**perplexity**），表示平均分支系数(average branch factor)，模型预测下一个词时的平均可选择数量。

**PTB文本数据集**

PTB(Penn Treebank Dataset) [下载地址](http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz)

使用`tar zxvf simple-examples.tgz`解压

下载[`reader.py`](https://github.com/tensorflow/models/blob/master/tutorials/rnn/ptb/reader.py)

数据使用[代码](https://github.com/caicloud/tensorflow-tutorial/blob/master/Deep_Learning_with_TensorFlow/1.0.0/Chapter08/2.%20PTB%E6%95%B0%E6%8D%AE%E9%9B%86%E4%BB%8B%E7%BB%8D.ipynb)

```python
import tensorflow as tf
import reader

DATA_PATH = "simple-examples/data"
train_data, valid_data, test_data, _ = reader.ptb_raw_data(DATA_PATH)

print(len(train_data))
print(train_data[:100])

# ptb_producer返回的为一个二维的tuple数据。4为batch大小，5为截断长度
result = reader.ptb_producer(train_data, 4, 5)

# 通过队列依次读取batch。
with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for i in range(3):
        x, y = sess.run(result)
        print("X{}: {}".format(i, x))
        print("Y{}: {}".format(i, y))
    coord.request_stop()
    coord.join(threads)
```

输出如下：

```
929589
[9970, 9971, 9972, 9974, 9975, 9976, 9980, 9981, 9982, 9983, 9984, 9986, 9987, 9988, 9989, 9991, 9992, 9993, 9994, 9995, 9996, 9997, 9998, 9999, 2, 9256, 1, 3, 72, 393, 33, 2133, 0, 146, 19, 6, 9207, 276, 407, 3, 2, 23, 1, 13, 141, 4, 1, 5465, 0, 3081, 1596, 96, 2, 7682, 1, 3, 72, 393, 8, 337, 141, 4, 2477, 657, 2170, 955, 24, 521, 6, 9207, 276, 4, 39, 303, 438, 3684, 2, 6, 942, 4, 3150, 496, 263, 5, 138, 6092, 4241, 6036, 30, 988, 6, 241, 760, 4, 1015, 2786, 211, 6, 96, 4]
X0: [[9970 9971 9972 9974 9975]
 [ 332 7147  328 1452 8595]    
 [1969    0   98   89 2254]    
 [   3    3    2   14   24]]   
Y0: [[9971 9972 9974 9975 9976]
 [7147  328 1452 8595   59]    
 [   0   98   89 2254    0]    
 [   3    2   14   24  198]]   
X1: [[9976 9980 9981 9982 9983]
 [  59 1569  105 2231    1]    
 [   0  312 1641    4 1063]    
 [ 198  150 2262   10    0]]   
Y1: [[9980 9981 9982 9983 9984]
 [1569  105 2231    1  895]    
 [ 312 1641    4 1063    8]    
 [ 150 2262   10    0  507]]   
X2: [[9984 9986 9987 9988 9989]
 [ 895    1 5574    4  618]    
 [   8  713    0  264  820]    
 [ 507   74 2619    0    1]]   
Y2: [[9986 9987 9988 9989 9991]
 [   1 5574    4  618    2]    
 [ 713    0  264  820    2]    
 [  74 2619    0    1    8]]   
```

- 数据集中包含929589个单词
- 每次单词有特有的ID，每个句子的结束标识为2
- 截断大小为5：代表RNN的输入最多只有5个元素，过多会导致梯度消失
- batch_size为4：代表得到几批数据
- 会自动生成每个batch对应的答案，代表当前单词的后一个单词

完整的RNN语言模型代码

```python
import numpy as np
import tensorflow as tf
import reader

# 读取数据并打印长度及前100位数据
DATA_PATH = "simple-examples/data"

# 网络参数
HIDDEN_SIZE = 200  # 隐藏层规模
NUM_LAYERS = 2  # 深层RNN中LSTM结构的层数
VOCAB_SIZE = 10000  # 词典规模

LEARNING_RATE = 1.0  # 学习速率
TRAIN_BATCH_SIZE = 20  # 训练数据batch的大小
TRAIN_NUM_STEP = 35  # 训练数据截断长度

# 在测试时不需截断，可以将测试数据看作超长序列
EVAL_BATCH_SIZE = 1  # 测试数据batch的大小
EVAL_NUM_STEP = 1  # 测试数据截断长度
NUM_EPOCH = 2  # 使用训练数据的轮数
KEEP_PROB = 0.5  # 节点不被dropout的概率
MAX_GRAD_NORM = 5  # 用于控制梯度膨胀的参数


class PTBModel(object):
    def __init__(self, is_training, batch_size, num_steps):
        # 记录使用的batch大小和截断长度
        self.batch_size = batch_size
        self.num_steps = num_steps

        # 定义输入层，可以看到输入层的维度为batch_size*num_steps
        # 这和ptb_producter函数输出的训练数据batch是一致的
        self.input_data = tf.placeholder(tf.int32, [batch_size, num_steps])

        # 定义预期输出
        self.targets = tf.placeholder(tf.int32, [batch_size, num_steps])

        # 定义使用LSTM结构为循环体结构且使用dropout的深层循环神经网络
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
        if is_training:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
                lstm_cell, output_keep_prob=KEEP_PROB)
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * NUM_LAYERS)

        # 初始化最初的状态
        self.initial_state = cell.zero_state(batch_size, tf.float32)
        # 将单词ID转换为单词向量。因为总共有VOCAB_SIZE个单词，每个单词向量的维度为
        # HIDDEN_SIZE, 所以embedding参数维度为VOCAB_SIZE*HIDDEN_SIZE
        embedding = tf.get_variable("embedding", [VOCAB_SIZE, HIDDEN_SIZE])

        # 将原本batch_size*num_steps个单词ID转换为单词向量，转化为的输入层维度
        # 为batch_size*num_steps*HIDDEN_SIZE
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)

        # 只在训练时dropout
        if is_training:
            inputs = tf.nn.dropout(inputs, KEEP_PROB)

        # 定义输出列表。在这里先将不同时刻LSTM结构的输出收集起来，再通过一个全连接得到最终的输出
        outputs = []
        # state存储不同batch中LSTM的状态，将其初始化为0
        state = self.initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                # 从输入数据中获取当前时刻的输入并传入LSTM
                cell_output, state = cell(inputs[:, time_step, :], state)
                # 将当前输出加入输出队列
                outputs.append(cell_output)

        # 把输出队列展开为[batch, hidden_size*num_steps]
        # 然后reshape成batch*nu_steps, hidden_size
        output = tf.reshape(tf.concat(outputs, 1), [-1, HIDDEN_SIZE])

        # 将从LSTM中得到的输出再经过一个全连接层得到最后的预测结果，最终的预测结果在每一个时刻上
        # 都是长度为VOCAB_SIZE的数组，经过softmax层表示下一个位置是不同单词的概率
        weight = tf.get_variable("weight", [HIDDEN_SIZE, VOCAB_SIZE])
        bias = tf.get_variable("bias", [VOCAB_SIZE])
        logits = tf.matmul(output, weight) + bias

        # 定义交叉熵损失函数
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(self.targets, [-1])],
            [tf.ones([batch_size * num_steps], dtype=tf.float32)])
        self.cost = tf.reduce_sum(loss) / batch_size
        self.final_state = state

        # 只在训练模型时定义反向传播操作。
        if not is_training:
            return
        trainable_variables = tf.trainable_variables()

        # 控制梯度大小，定义优化方法和训练步骤。
        grads, _ = tf.clip_by_global_norm(tf.gradients(
            self.cost, trainable_variables), MAX_GRAD_NORM)
        optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
        self.train_op = optimizer.apply_gradients(
            zip(grads, trainable_variables))


# 使用给定的模型model在数据data上运行train_op并返回在全部数据上的perplexity值
def run_epoch(session, model, data, train_op, output_log, epoch_size):
    total_costs = 0.0
    iters = 0
    state = session.run(model.initial_state)

    # 训练一个epoch。
    for step in range(epoch_size):
        x, y = session.run(data)
        cost, state, _ = session.run(
            [model.cost, model.final_state, train_op],
            {model.input_data: x,
             model.targets: y,
             model.initial_state: state})
        total_costs += cost
        iters += model.num_steps

        if output_log and step % 100 == 0:
            print("After %d steps, perplexity is %.3f" %
                  (step, np.exp(total_costs / iters)))
    return np.exp(total_costs / iters)


# 定义主函数并执行
def main():
    train_data, valid_data, test_data, _ = reader.ptb_raw_data(DATA_PATH)

    # 计算一个epoch需要训练的次数
    train_data_len = len(train_data)
    train_batch_len = train_data_len // TRAIN_BATCH_SIZE
    train_epoch_size = (train_batch_len - 1) // TRAIN_NUM_STEP

    valid_data_len = len(valid_data)
    valid_batch_len = valid_data_len // EVAL_BATCH_SIZE
    valid_epoch_size = (valid_batch_len - 1) // EVAL_NUM_STEP

    test_data_len = len(test_data)
    test_batch_len = test_data_len // EVAL_BATCH_SIZE
    test_epoch_size = (test_batch_len - 1) // EVAL_NUM_STEP

    initializer = tf.random_uniform_initializer(-0.05, 0.05)
    with tf.variable_scope("language_model", reuse=None,
                           initializer=initializer):
        train_model = PTBModel(True, TRAIN_BATCH_SIZE, TRAIN_NUM_STEP)

    with tf.variable_scope("language_model", reuse=True,
                           initializer=initializer):
        eval_model = PTBModel(False, EVAL_BATCH_SIZE, EVAL_NUM_STEP)

    # 训练模型。
    with tf.Session() as session:
        tf.global_variables_initializer().run()

        train_queue = reader.ptb_producer(
            train_data, train_model.batch_size, train_model.num_steps)
        eval_queue = reader.ptb_producer(
            valid_data, eval_model.batch_size, eval_model.num_steps)
        test_queue = reader.ptb_producer(
            test_data, eval_model.batch_size, eval_model.num_steps)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=session, coord=coord)

        for i in range(NUM_EPOCH):
            print("In iteration: %d" % (i + 1))
            run_epoch(session, train_model, train_queue,
                      train_model.train_op, True, train_epoch_size)

            valid_perplexity = run_epoch(
                session, eval_model, eval_queue, tf.no_op(),
                False, valid_epoch_size)
            print("Epoch: %d Validation Perplexity: %.3f" %
                  (i + 1, valid_perplexity))

        test_perplexity = run_epoch(
            session, eval_model, test_queue, tf.no_op(),
            False, test_epoch_size)
        print("Test Perplexity: %.3f" % test_perplexity)

        coord.request_stop()
        coord.join(threads)


if __name__ == "__main__":
    main()
```

## 时间序列预测

TFLearn IRIS代码实例

```python
from sklearn import model_selection
from sklearn import datasets
from sklearn import metrics
import tensorflow as tf
import numpy as np
from tensorflow.contrib.learn.python.learn.estimators.estimator import SKCompat
learn = tf.contrib.learn


# 自定义softmax回归模型
def my_model(features, target):
    target = tf.one_hot(target, 3, 1, 0)

    # 计算预测值及损失函数。
    logits = tf.contrib.layers.fully_connected(features, 3, tf.nn.softmax)
    loss = tf.losses.softmax_cross_entropy(target, logits)

    # 创建优化步骤。
    train_op = tf.contrib.layers.optimize_loss(
        loss,
        tf.contrib.framework.get_global_step(),
        optimizer='Adam',
        learning_rate=0.01)
    return tf.argmax(logits, 1), loss, train_op


# 读取数据并将数据转化成TensorFlow要求的float32格式
iris = datasets.load_iris()
x_train, x_test, y_train, y_test = model_selection.train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=0)

x_train, x_test = map(np.float32, [x_train, x_test])

classifier = SKCompat(learn.Estimator(
    model_fn=my_model, model_dir="Models/model_1"))
classifier.fit(x_train, y_train, steps=800)

y_predicted = [i for i in classifier.predict(x_test)]
score = metrics.accuracy_score(y_test, y_predicted)
print('Accuracy: %.2f%%' % (score * 100))
```

预测sin曲线

```python
import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.estimators.estimator import SKCompat
import matplotlib.pyplot as plt
learn = tf.contrib.learn

# 设置神经网络的参数
HIDDEN_SIZE = 30  # LSTM中隐藏节点的个数
NUM_LAYERS = 2  # LSTM的层数

TIMESTEPS = 10  # RNN的截断长度
TRAINING_STEPS = 3000  # 训练轮次
BATCH_SIZE = 32  # batch大小

TRAINING_EXAMPLES = 10000  # 训练数据个数
TESTING_EXAMPLES = 1000  # 测试数据个数
SAMPLE_GAP = 0.01  # 采样间隔


# 定义生成正弦数据的函数
def generate_data(seq):
    X = []
    y = []

    # 序列的第i项和后面的TIMESTEPS-1项合并在一起作为输入。
    # 第i+TIMESTPES项作为输出。
    # 即用sin函数前面的TIMESTEPS个点的信息，预测第i+TIMESTPES个点的函数值
    for i in range(len(seq) - TIMESTEPS - 1):
        X.append([seq[i: i + TIMESTEPS]])
        y.append([seq[i + TIMESTEPS]])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# 定义lstm模型
def lstm_model(X, y):
    def lstm_cell():
        return tf.contrib.rnn.BasicLSTMCell(HIDDEN_SIZE, state_is_tuple=True)
    cell = tf.contrib.rnn.MultiRNNCell(
        [lstm_cell() for _ in range(NUM_LAYERS)])

    output, _ = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    output = tf.reshape(output, [-1, HIDDEN_SIZE])

    # 通过无激活函数的全联接层计算线性回归，并将数据压缩成一维数组的结构。
    predictions = tf.contrib.layers.fully_connected(output, 1, None)

    # 将predictions和labels调整统一的shape
    labels = tf.reshape(y, [-1])
    predictions = tf.reshape(predictions, [-1])

    loss = tf.losses.mean_squared_error(predictions, labels)

    # 创建模型优化器并得到优化步骤
    train_op = tf.contrib.layers.optimize_loss(
        loss, tf.contrib.framework.get_global_step(),
        optimizer="Adagrad", learning_rate=0.1)

    return predictions, loss, train_op


# 进行训练
# 封装之前定义的lstm。
regressor = SKCompat(learn.Estimator(
    model_fn=lstm_model, model_dir="Models/model_2"))

# 生成数据。
test_start = TRAINING_EXAMPLES * SAMPLE_GAP
test_end = (TRAINING_EXAMPLES + TESTING_EXAMPLES) * SAMPLE_GAP
train_X, train_y = generate_data(np.sin(np.linspace(
    0, test_start, TRAINING_EXAMPLES, dtype=np.float32)))
test_X, test_y = generate_data(np.sin(np.linspace(
    test_start, test_end, TESTING_EXAMPLES, dtype=np.float32)))

# 拟合数据。
regressor.fit(train_X, train_y, batch_size=BATCH_SIZE, steps=TRAINING_STEPS)

# 计算预测值。
predicted = [[pred] for pred in regressor.predict(test_X)]

# 计算MSE。
rmse = np.sqrt(((predicted - test_y) ** 2).mean(axis=0))
print("Mean Square Error is: %f" % rmse[0])

plot_predicted, = plt.plot(predicted, label='predicted')
plot_test, = plt.plot(test_y, label='real_sin')
plt.legend([plot_predicted, plot_test], ['predicted', 'real_sin'])
plt.show()
```