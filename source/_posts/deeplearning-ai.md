---
title: deeplearning.ai
date: 2017-11-06 23:00:00
tags: 深度学习
categories: 机器学习
---

课程在[网易云课堂](https://mooc.study.163.com/course/deeplearning_ai-2001281002#/info)上免费观看，作业题如下：加粗为个人见解，非答案。

<!-- more -->

# 第一周

1. What does the analogy “AI is the new electricity” refer to?
    1. **Similar to electricity starting about 100 years ago, AI is transforming multiple industries.**
    1. Through the “smart grid”, AI is delivering a new wave of electricity.
    1. AI runs on computers and is thus powered by electricity, but it is letting computers do things not possible before.
    1. AI is powering personal devices in our homes and offices, similar to electricity.
1. Which of these are reasons for Deep Learning recently taking off? (Check the three options that apply.)
    1. **Deep learning has resulted in significant improvements in important applications such as online advertising, speech recognition, and image recognition.**
    1. **We have access to a lot more data.**
    1. **We have access to a lot more computational power.**
    1. Neural Networks are a brand new field.
1. Recall this diagram of iterating over different ML ideas. Which of the statements below are true? (Check all that apply.)   
    ![](http://outz1n6zr.bkt.clouddn.com/20171106212040.png)
    1. **Being able to try out ideas quickly allows deep learning engineers to iterate more quickly.**
    1. **Faster computation can help speed up how long a team takes to iterate to a good idea.**
    1. It is faster to train on a big dataset than a small dataset.
    1. **Recent progress in deep learning algorithms has allowed us to train good models faster (even without changing the CPU/GPU hardware).**
1. When an experienced deep learning engineer works on a new problem, they can usually use insight from previous problems to train a good model on the first try, without needing to iterate multiple times through different models. True/False?
    1. True
    1. **False**
1. Which one of these plots represents a ReLU activation function?
    1. Figure 1:   
    ![](http://outz1n6zr.bkt.clouddn.com/20171106212525.png)
    1. Figure 2:   
    ![](http://outz1n6zr.bkt.clouddn.com/20171106212535.png)
    1. **Figure 3:**   
    ![](http://outz1n6zr.bkt.clouddn.com/20171106212541.png)
    1. Figure 4:   
    ![](http://outz1n6zr.bkt.clouddn.com/20171106212548.png)
1. Images for cat recognition is an example of “structured” data, because it is represented as a structured array in a computer. True/False?
    1. True
    1. **False**
1. A demographic dataset with statistics on different cities' population, GDP per capita, economic growth is an example of “unstructured” data because it contains data coming from different sources. True/False?
    1. True
    1. **False**
1. Why is an RNN (Recurrent Neural Network) used for machine translation, say translating English to French? (Check all that apply.)
    1. **It can be trained as a supervised learning problem.**
    1. It is strictly more powerful than a Convolutional Neural Network (CNN).
    1. **It is applicable when the input/output is a sequence (e.g., a sequence of words).**
    1. RNNs represent the recurrent process of Idea->Code->Experiment->Idea->....
1. In this diagram which we hand-drew in lecture, what do the horizontal axis (x-axis) and vertical axis (y-axis) represent?  
    ![](http://outz1n6zr.bkt.clouddn.com/20171106212556.png)
    1.  - x-axis is the performance of the algorithm
        - y-axis (vertical axis) is the amount of data.
    1.  - x-axis is the input to the algorithm
        - y-axis is outputs.
    1.  - x-axis is the amount of data
        - y-axis is the size of the model you train.
    1.  - **x-axis is the amount of data**
        - **y-axis (vertical axis) is the performance of the algorithm.**
1. Assuming the trends described in the previous question's figure are accurate (and hoping you got the axis labels right), which of the following are true? (Check all that apply.)
    1. Decreasing the training set size generally does not hurt an algorithm’s performance, and it may help significantly.
    1. Decreasing the size of a neural network generally does not hurt an algorithm’s performance, and it may help significantly.
    1. **Increasing the training set size generally does not hurt an algorithm’s performance, and it may help significantly.**
    1. **Increasing the size of a neural network generally does not hurt an algorithm’s performance, and it may help significantly.**

# 第二周

符号约定

![](http://outz1n6zr.bkt.clouddn.com/20171106212859.png)

![](http://outz1n6zr.bkt.clouddn.com/20171106213014.png)

## Neural-Network-Basics

1. What does a neuron compute?
    1. A neuron computes an activation function followed by a linear function (z = Wx + b)
    1. **A neuron computes a linear function (z = Wx + b) followed by an activation function**
    1. A neuron computes a function g that scales the input x linearly (Wx + b)
    1. A neuron computes the mean of all features before applying the output to an activation function
1. Which of these is the "Logistic Loss"?
    1. $L^{(i)}(\hat{y}^{(i)},y^{(i)}) = -(y^{(i)}\log(\hat{y}^{(i)}) + (1-y^{(i)})\log(1-\hat{y}^{(i)}))$ **True**
    1. $L^{(i)}(\hat{y}^{(i)},y^{(i)}) = |y^{(i)} - \hat{y}^{(i)}|$
    1. $L^{(i)}(\hat{y}^{(i)},y^{(i)}) = \max(0, y^{(i)} - \hat{y}^{(i)})$
    1. $L^{(i)}(\hat{y}^{(i)},y^{(i)}) = |y^{(i)} - \hat{y}^{(i)}|^2$
1. Suppose img is a (32,32,3) array, representing a 32x32 image with 3 color channels red, green and blue. How do you reshape this into a column vector?
    1. **`x = img.reshape((32*32*3,1))`**
    1. `x = img.reshape((32*32,3))`
    1. `x = img.reshape((1,32*32,*3))`
    1. `x = img.reshape((3,32*32))`
1. Consider the two following random arrays "a" and "b":
    ```python
    a = np.random.randn(2, 3) # a.shape = (2, 3)
    b = np.random.randn(2, 1) # b.shape = (2, 1)
    c = a + b
    ```
    What will be the shape of "c"?
    1. c.shape = (2, 1)
    1. c.shape = (3, 2)
    1. **c.shape = (2, 3)**
    1. The computation cannot happen because the sizes don't match. It's going to be "Error"!
1. Consider the two following random arrays "a" and "b":
    ```python
    a = np.random.rand(4, 3) # a.shape = (4, 3)
    b = np.random.rand(3, 2) # a.shape = (3, 2)
    c = a*b
    ```
    What will be the shape of "c"?
    1. c.shape = (4, 2)
    1. c.shape = (4, 3)
    1. **The computation cannot happen because the sizes don't match. It's going to be "Error"!**
    1. c.shape = (3, 3)
1. Suppose you have $n_x$ input features per example. Recall that $X=[x^{(2)}x^{(m)}...x^{(1)}]$. What is the dimension of X?
    1. (m, 1)
    1. **($n_x$, m)**
    1. (m, $n_x$)
    1. (1, m)
1. Recall that "np.dot(a,b)" performs a matrix multiplication on a and b, whereas "a*b" performs an element-wise multiplication.  
    Consider the two following random arrays "a" and "b":
    ```python
    a = np.random.randn(12288, 150) # a.shape = (12288, 150)
    b = np.random.randn(150, 45) # b.shape = (150, 45)
    c = np.dot(a, b)
    ```
    What is the shape of c?
    1. The computation cannot happen because the sizes don't match. It's going to be "Error"!
    1. c.shape = (12288, 150)
    1. **c.shape = (12288, 45)**
    1. c.shape = (150,150)
1. Consider the following code snippet:
    ```python
    # a.shape = (3, 4)
    # b.shape = (4, 1)

    for i in range(3):
        for j in range(4):
            c[i][j] = a[i][j] + b[j]
    ```
    How do you vectorize this?
    1. c = a.T + b
    1. c = a.T + b.T
    1. **c = a + b.T**
    1. c = a + b
1. Consider the following code:
    ```python
    a = np.random.randn(3, 3)
    b = np.random.randn(3, 1)
    c = a * b
    ```
    What will be c? (If you’re not sure, feel free to run this in python to find out).
    1. **This will invoke broadcasting, so b is copied three times to become (3,3), and ∗ is an element-wise product so c.shape will be (3, 3)**
    1. This will invoke broadcasting, so b is copied three times to become (3, 3), and ∗ invokes a matrix multiplication operation of two 3x3 matrices so c.shape will be (3, 3)
    1. This will multiply a 3x3 matrix a with a 3x1 vector, thus resulting in a 3x1 vector. That is, c.shape = (3,1).
    1. It will lead to an error since you cannot use “*” to operate on these two matrices. You need to instead use np.dot(a,b)
1. Consider the following computation graph.  
    ![](...)  
    What is the output J?
    1. `J = (c - 1)*(b + a)`
    1. **`J = (a - 1) * (b + c)`**
    1. `J = a*b + b*c + a*c`
    1. `J = (b - 1) * (c + a)`

## Logistic-Regression-with-a-Neural-Network-mindset

相关数据集和输出见github

```python
# 1 - Packages
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage
from lr_utils import load_dataset

# %matplotlib inline

# 2 - Overview of the Problem set

# Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# Example of a picture
index = 25
plt.imshow(train_set_x_orig[index])
# plt.show()
print("y = " + str(train_set_y[:, index]) + ", it's a '" +
      classes[np.squeeze(train_set_y[:, index])].decode("utf-8") + "' picture.")


# START CODE HERE ### (≈ 3 lines of code)
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]
# END CODE HERE ###

print("Number of training examples: m_train = " + str(m_train))
print("Number of testing examples: m_test = " + str(m_test))
print("Height/Width of each image: num_px = " + str(num_px))
print("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print("train_set_x shape: " + str(train_set_x_orig.shape))
print("train_set_y shape: " + str(train_set_y.shape))
print("test_set_x shape: " + str(test_set_x_orig.shape))
print("test_set_y shape: " + str(test_set_y.shape))

# Reshape the training and test examples

# START CODE HERE ### (≈ 2 lines of code)
train_set_x_flatten = train_set_x_orig.reshape(m_train, -1).T
test_set_x_flatten = test_set_x_orig.reshape(m_test, -1).T
# END CODE HERE ###

print("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print("train_set_y shape: " + str(train_set_y.shape))
print("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print("test_set_y shape: " + str(test_set_y.shape))
print("sanity check after reshaping: " + str(train_set_x_flatten[0:5, 0]))

train_set_x = train_set_x_flatten / 255.
test_set_x = test_set_x_flatten / 255.

# 3 - General Architecture of the learning algorithm
# 4 - Building the parts of our algorithm


# 4.1 - Helper functions
# GRADED FUNCTION: sigmoid
def sigmoid(z):
    """
    Compute the sigmoid of z
​
    Arguments:
    z -- A scalar or numpy array of any size.
​
    Return:
    s -- sigmoid(z)
    """

    # START CODE HERE ### (≈ 1 line of code)
    s = 1 / (1 + np.exp(-z))
    # END CODE HERE ###

    return s


print("sigmoid([0, 2]) = " + str(sigmoid(np.array([0, 2]))))


# 4.2 - Initializing parameters
# GRADED FUNCTION: initialize_with_zeros
def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.

    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)

    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """

    # START CODE HERE ### (≈ 1 line of code)
    w = np.zeros((dim, 1))
    b = 0

    # END CODE HERE ###

    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))

    return w, b


dim = 2
w, b = initialize_with_zeros(dim)
print("w = " + str(w))
print("b = " + str(b))


# 4.3 - Forward and Backward propagation
# GRADED FUNCTION: propagate
def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation explained above
​
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)
​
    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b

    Tips:
    - Write your code step by step for the propagation. np.log(), np.dot()
    """

    m = X.shape[1]

    # FORWARD PROPAGATION (FROM X TO COST)
    # START CODE HERE ### (≈ 2 lines of code)
    A = sigmoid(np.dot(w.T, X) + b)                                # compute activation
    cost = np.sum(Y * np.log(A) - (1 - Y) * np.log(1 - A)) / (-m)  # compute cost
    # END CODE HERE ###

    # BACKWARD PROPAGATION (TO FIND GRAD)
    # START CODE HERE ### (≈ 2 lines of code)
    # dw = da * dz * dw
    dw = np.dot(X, (A - Y).T) / m
    db = np.sum(A - Y) / m

    # END CODE HERE ###
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())

    grads = {"dw": dw,
             "db": db}

    return grads, cost


w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1, 2], [3, 4]]), np.array([[1, 0]])
grads, cost = propagate(w, b, X, Y)
print("dw = " + str(grads["dw"]))
print("db = " + str(grads["db"]))
print("cost = " + str(cost))


# GRADED FUNCTION: optimize
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    """
    This function optimizes w and b by running a gradient descent algorithm

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps

    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.

    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for w and b.
    """

    costs = []

    for i in range(num_iterations):

        # Cost and gradient calculation (≈ 1-4 lines of code)
        # START CODE HERE ###
        grads, cost = propagate(w, b, X, Y)
        # END CODE HERE ###

        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]

        # update rule (≈ 2 lines of code)
        # START CODE HERE ###
        w = w - learning_rate * dw
        b = b - learning_rate * db
        # END CODE HERE ###

        # Record the costs
        if i % 100 == 0:
            costs.append(cost)

        # Print the cost every 100 training examples
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs


params, grads, costs = optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False)

print("w = " + str(params["w"]))
print("b = " + str(params["b"]))
print("dw = " + str(grads["dw"]))
print("db = " + str(grads["db"]))


# GRADED FUNCTION: predict
def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)

    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''

    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    # START CODE HERE ### (≈ 1 line of code)
    A = sigmoid(np.dot(w.T, X) + b)
    # END CODE HERE ###

    for i in range(A.shape[1]):

        # Convert probabilities a[0,i] to actual predictions p[0,i]
        # START CODE HERE ### (≈ 4 lines of code)
        if (A[0, i] > 0.5):
            Y_prediction[0][i] = 1
        else:
            Y_prediction[0][i] = 0
        # END CODE HERE ###

    assert(Y_prediction.shape == (1, m))

    return Y_prediction


print("predictions = " + str(predict(w, b, X)))


# 5 - Merge all functions into a model
# GRADED FUNCTION: model
def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    """
    Builds the logistic regression model by calling the function you've implemented previously

    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations

    Returns:
    d -- dictionary containing information about the model.
    """

    # START CODE HERE ###

    # initialize parameters with zeros (≈ 1 line of code)
    w, b = initialize_with_zeros(X_train.shape[0])

    # Gradient descent (≈ 1 line of code)
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]

    # Predict test/train set examples (≈ 2 lines of code)
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    # END CODE HERE ###

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d


d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=0.005, print_cost=True)

# Example of a picture that was wrongly classified.
index = 1
plt.imshow(test_set_x[:, index].reshape((num_px, num_px, 3)))
#plt.show()
print("y = " + str(test_set_y[0, index]) + ", you predicted that it is a \"" +
      classes[int(d["Y_prediction_test"][0, index])].decode("utf-8") + "\" picture.")

# Plot learning curve (with costs)
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
#plt.show()

learning_rates = [0.01, 0.001, 0.0001]
models = {}
for i in learning_rates:
    print("learning rate is: " + str(i))
    models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y,
                           num_iterations=1500, learning_rate=i, print_cost=False)
    print('\n' + "-------------------------------------------------------" + '\n')

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label=str(models[str(i)]["learning_rate"]))

plt.ylabel('cost')
plt.xlabel('iterations')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
#plt.show()

# START CODE HERE ## (PUT YOUR IMAGE NAME)
my_image = "Lion_waiting_in_Namibia.jpg"   # change this to the name of your image file
# END CODE HERE ##

# We preprocess the image to fit your algorithm.
fname = "images/" + my_image
image = np.array(ndimage.imread(fname, flatten=False))
my_image = scipy.misc.imresize(image, size=(num_px, num_px)).reshape((1, num_px * num_px * 3)).T
my_predicted_image = predict(d["w"], d["b"], my_image)

plt.imshow(image)
print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" +
      classes[int(np.squeeze(my_predicted_image)), ].decode("utf-8") + "\" picture.")
```

# 第三周

## Shallow Neural Networks

1. Which of the following are true? (Check all that apply.)
    1. **$a^{[2]}$ denotes the activation vector of the $2^{nd}$ layer.**
    1. **$a^{[2](12)}$ denotes the activation vector of the $2^{nd}$ layer for the $12^{th}$ training example.**
    1. $X$ is a matrix in which each row is one training example.
    1. $a^{[2](12)}$ denotes activation vector of the $12^{th}$ layer on the $2^{nd}$ training example.
    1. **$X$ is a matrix in which each column is one training example.**
    1. **$a^{[2]}_4$ is the activation output by the $4^{th}$ neuron of the $2^{nd}$ layer**
    1. $a^{[2]}_4$ is the activation output of the $2^{nd}$ layer for the $4^{th}$ training example
1. The tanh activation usually works better than sigmoid activation function for hidden units because the mean of its output is closer to zero, and so it centers the data better for the next layer. True/False?
    1. **True**
    1. False
1. Which of these is a correct vectorized implementation of forward propagation for layer $l$, where $1\leq{}l\leq{}L$?
    1. - $Z^{[l]}=W^{[l]}A^{[l]}+b^{[l]}$
       - $A^{[l+1]}=g^{[l+1]}(Z^{[l]})$
    1. - $Z^{[l]}=W^{[l]}A^{[l-1]}+b^{[l]}$ **True**
       - $A^{[l]}=g^{[l]}(Z^{[l]})$ **True**
    1. - $Z^{[l]}=W^{[l]}A^{[l]}+b^{[l]}$
       - $A^{[l+1]}=g^{[l]}(Z^{[l]})$
    1. - $Z^{[l]}=W^{[l-1]}A^{[l]}+b^{[l-1]}$
       - $A^{[l]}=g^{[l]}(Z^{[l]})$
1. You are building a binary classifier for recognizing cucumbers (y=1) vs. watermelons (y=0). Which one of these activation functions would you recommend using for the output layer?
    1. ReLU
    1. Leaky ReLU
    1. **sigmoid**
    1. tanh
1. Consider the following code:
    ```python
    A = np.random.randn(4, 3)
    B = np.sum(A, axis = 1, keepdims = True)
    ```
    What will be B.shape? (If you’re not sure, feel free to run this in python to find out).
    1. **(4, 1)**
    1. (4, )
    1. (1, 3)
    1. (, 3)
1. Suppose you have built a neural network. You decide to initialize the weights and biases to be zero. Which of the following statements is true?
    1. **Each neuron in the first hidden layer will perform the same computation. So even after multiple iterations of gradient descent each neuron in the layer will be computing the same thing as other neurons.**
    1. Each neuron in the first hidden layer will perform the same computation in the first iteration. But after one iteration of gradient descent they will learn to compute different things because we have “broken symmetry”.
    1. Each neuron in the first hidden layer will compute the same thing, but neurons in different layers will compute different things, thus we have accomplished “symmetry breaking” as described in lecture.
    1. The first hidden layer’s neurons will perform different computations from each other even in the first iteration; their parameters will thus keep evolving in their own way.
1. Logistic regression’s weights w should be initialized randomly rather than to all zeros, because if you initialize to all zeros, then logistic regression will fail to learn a useful decision boundary because it will fail to “break symmetry”, True/False?
    1. True
    1. False
1. You have built a network using the tanh activation for all the hidden units. You initialize the weights to relative large values, using np.random.randn(..,..)*1000. What will happen?
    1. **This will cause the inputs of the tanh to also be very large, thus causing gradients to be close to zero. The optimization algorithm will thus become slow.**
    1. This will cause the inputs of the tanh to also be very large, causing the units to be “highly activated” and thus speed up learning compared to if the weights had to start from small values.
    1. It doesn’t matter. So long as you initialize the weights randomly gradient descent is not affected by whether the weights are large or small.
    1. This will cause the inputs of the tanh to also be very large, thus causing gradients to also become large. You therefore have to set α to be very small to prevent divergence; this will slow down learning.
1. Consider the following 1 hidden layer neural network:  
    ![](...)
    Which of the following statements are True? (Check all that apply).
    1. $W^{[1]}$ will have shape (2, 4)
    1. **$b^{[1]}$ will have shape (4, 1)**
    1. **$W^{[1]}$ will have shape (4, 2)**
    1. $b^{[1]}$ will have shape (2, 1)
    1. **$W^{[2]}$ will have shape (1, 4)**
    1. $b^{[2]}$ will have shape (4, 1)
    1. $W^{[2]}$ will have shape (4, 1)
    1. **$b^{[2]}$ will have shape (1, 1)**
1. In the same network as the previous question, what are the dimensions of $Z^{[1]}$ and $A^{[1]}$?
    1. $Z^{[1]}$ and $A^{[1]}$ are (1, 4)
    1. $Z^{[1]}$ and $A^{[1]}$ are (4, 2)
    1. $Z^{[1]}$ and $A^{[1]}$ are (4, m)
    1. **$Z^{[1]}$ and $A^{[1]}$ are (4, 1)**
    
## Planar data classification with one hidden layer

```python
# Package imports
import numpy as np
import matplotlib.pyplot as plt
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

np.random.seed(1)  # set a seed so that the results are consistent

# 2 - Dataset
X, Y = load_planar_dataset()

# Visualize the data:
plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral)
# plt.show()

# START CODE HERE ### (≈ 3 lines of code)
shape_X = X.shape
shape_Y = Y.shape
m = shape_X[1]  # training set size
# END CODE HERE ###

print('The shape of X is: ' + str(shape_X))
print('The shape of Y is: ' + str(shape_Y))
print('I have m = %d training examples!' % (m))

# 3 - Simple Logistic Regression
# Train the logistic regression classifier
clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X.T, Y.T)

# Plot the decision boundary for logistic regression
plot_decision_boundary(lambda x: clf.predict(x), X, Y)
plt.title("Logistic Regression")
# plt.show()

# Print accuracy
LR_predictions = clf.predict(X.T)
print('Accuracy of logistic regression: %d ' %
      float((np.dot(Y, LR_predictions) +
             np.dot(1 - Y, 1 - LR_predictions)) / float(Y.size) * 100) +
      '% ' + "(percentage of correctly labelled datapoints)")

# 4 - Neural Network model
# 4.1 - Defining the neural network structure
# GRADED FUNCTION: layer_sizes


def layer_sizes(X, Y):
    """
    Arguments:
    X -- input dataset of shape (input size, number of examples)
    Y -- labels of shape (output size, number of examples)

    Returns:
    n_x -- the size of the input layer
    n_h -- the size of the hidden layer
    n_y -- the size of the output layer
    """
    # START CODE HERE ### (≈ 3 lines of code)
    n_x = X.shape[0]  # size of input layer
    n_h = X.shape[1]
    n_y = Y.shape[0]  # size of output layer
    # END CODE HERE ###
    return (n_x, n_h, n_y)


X_assess, Y_assess = layer_sizes_test_case()
(n_x, n_h, n_y) = layer_sizes(X_assess, Y_assess)
print("The size of the input layer is: n_x = " + str(n_x))
print("The size of the hidden layer is: n_h = " + str(n_h))
print("The size of the output layer is: n_y = " + str(n_y))

# 4.2 - Initialize the model's parameters
# GRADED FUNCTION: initialize_parameters


def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer

    Returns:
    params -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """

    np.random.seed(2)  # we set up a seed so that your output matches ours although the initialization is random.

    # START CODE HERE ### (≈ 4 lines of code)
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    # END CODE HERE ###

    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


n_x, n_h, n_y = initialize_parameters_test_case()

parameters = initialize_parameters(n_x, n_h, n_y)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))


# 4.3 - The Loop
# GRADED FUNCTION: forward_propagation
def forward_propagation(X, parameters):
    """
    Argument:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing your parameters (output of initialization function)

    Returns:
    A2 -- The sigmoid output of the second activation
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
    """
    # Retrieve each parameter from the dictionary "parameters"
    # START CODE HERE ### (≈ 4 lines of code)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    # END CODE HERE ###

    # Implement Forward Propagation to calculate A2 (probabilities)
    # START CODE HERE ### (≈ 4 lines of code)
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    # END CODE HERE ###

    assert(A2.shape == (1, X.shape[1]))

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}

    return A2, cache


X_assess, parameters = forward_propagation_test_case()

A2, cache = forward_propagation(X_assess, parameters)

# Note: we use the mean here just to make sure that your output matches ours.
print(np.mean(cache['Z1']), np.mean(cache['A1']), np.mean(cache['Z2']), np.mean(cache['A2']))


# GRADED FUNCTION: compute_cost
def compute_cost(A2, Y, parameters):
    """
    Computes the cross-entropy cost given in equation (13)

    Arguments:
    A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    parameters -- python dictionary containing your parameters W1, b1, W2 and b2

    Returns:
    cost -- cross-entropy cost given equation (13)
    """

    m = Y.shape[1]  # number of example

    # Compute the cross-entropy cost
    # START CODE HERE ### (≈ 2 lines of code)
    logprobs = Y * np.log(A2) + (1 - Y) * np.log(1 - A2)
    cost = np.sum(logprobs) / -m
    # END CODE HERE ###

    cost = np.squeeze(cost)     # makes sure cost is the dimension we expect.
    # E.g., turns [[17]] into 17
    assert(isinstance(cost, float))

    return cost


A2, Y_assess, parameters = compute_cost_test_case()

print("cost = " + str(compute_cost(A2, Y_assess, parameters)))


# GRADED FUNCTION: backward_propagation
def backward_propagation(parameters, cache, X, Y):
    """
    Implement the backward propagation using the instructions above.

    Arguments:
    parameters -- python dictionary containing our parameters
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
    X -- input data of shape (2, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)

    Returns:
    grads -- python dictionary containing your gradients with respect to different parameters
    """
    m = X.shape[1]

    # First, retrieve W1 and W2 from the dictionary "parameters".
    # START CODE HERE ### (≈ 2 lines of code)
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    # END CODE HERE ###

    # Retrieve also A1 and A2 from dictionary "cache".
    # START CODE HERE ### (≈ 2 lines of code)
    A1 = cache["A1"]
    A2 = cache["A2"]
    # END CODE HERE ###

    # Backward propagation: calculate dW1, db1, dW2, db2.
    # START CODE HERE ### (≈ 6 lines of code, corresponding to 6 equations on slide above)
    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    dZ1 = np.dot(W2.T, dZ2) * (1 - A1**2)
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m
    # END CODE HERE ###

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return grads


parameters, cache, X_assess, Y_assess = backward_propagation_test_case()

grads = backward_propagation(parameters, cache, X_assess, Y_assess)
print("dW1 = " + str(grads["dW1"]))
print("db1 = " + str(grads["db1"]))
print("dW2 = " + str(grads["dW2"]))
print("db2 = " + str(grads["db2"]))


# GRADED FUNCTION: update_parameters
def update_parameters(parameters, grads, learning_rate=1.2):
    """
    Updates parameters using the gradient descent update rule given above

    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients 

    Returns:
    parameters -- python dictionary containing your updated parameters 
    """
    # Retrieve each parameter from the dictionary "parameters"
    # START CODE HERE ### (≈ 4 lines of code)
    W1 = None
    b1 = None
    W2 = None
    b2 = None
    ### END CODE HERE ###

    # Retrieve each gradient from the dictionary "grads"
    # START CODE HERE ### (≈ 4 lines of code)
    dW1 = None
    db1 = None
    dW2 = None
    db2 = None
    ## END CODE HERE ###

    # Update rule for each parameter
    # START CODE HERE ### (≈ 4 lines of code)
    W1 = None
    b1 = None
    W2 = None
    b2 = None
    ### END CODE HERE ###

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters
```