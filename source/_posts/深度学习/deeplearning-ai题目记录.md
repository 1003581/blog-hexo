---
title: deeplearning.ai题目记录
date: 2017-12-26 23:30:00
tags: 深度学习
categories: 深度学习
---

课程在[网易云课堂](https://study.163.com/provider/2001053000/index.htm)上免费观看，作业题如下：加粗为答案。
<!-- more -->
# 神经网络和深度学习

## 第一周 深度学习概论

10个选择题，原见[Github](https://github.com/liqiang311/deeplearning.ai/blob/master/1_Neural%20Networks%20and%20Deep%20Learning/Quiz-week1-Introduction%20to%20deep%20learning.pdf)

1. What does the analogy “AI is the new electricity” refer to?
    1. **Similar to electricity starting about 100 years ago, AI is transforming multiple industries.**
    1. Through the “smart grid”, AI is delivering a new wave of electricity.
    1. AI runs on computers and is thus powered by electricity, but it is letting computers do things not possible before.
    1. AI is powering personal devices in our homes and offices, similar to electricity.
1. Which of these are reasons for Deep Learning recently taking off? (Check the three options that apply.)
    1. Deep learning has resulted in significant improvements in important applications such as online advertising, speech recognition, and image recognition.
    1. **We have access to a lot more data.**
    1. **We have access to a lot more computational power.**
    1. Neural Networks are a brand new field.
1. Recall this diagram of iterating over different ML ideas. Which of the statements below are true? (Check all that apply.)   
    ![img](http://upload-images.jianshu.io/upload_images/5952841-c562c3e1edff606d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
    1. **Being able to try out ideas quickly allows deep learning engineers to iterate more quickly.**
    1. **Faster computation can help speed up how long a team takes to iterate to a good idea.**
    1. It is faster to train on a big dataset than a small dataset.
    1. **Recent progress in deep learning algorithms has allowed us to train good models faster (even without changing the CPU/GPU hardware).**
1. When an experienced deep learning engineer works on a new problem, they can usually use insight from previous problems to train a good model on the first try, without needing to iterate multiple times through different models. True/False?
    1. True
    1. **False**
1. Which one of these plots represents a ReLU activation function?
    1. Figure 1:   
    ![img](http://upload-images.jianshu.io/upload_images/5952841-ece3dc7648f76fcb.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
    1. Figure 2:   
    ![img](http://upload-images.jianshu.io/upload_images/5952841-3c39eeef8b4923cd.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
    1. **Figure 3:**   
    ![img](http://upload-images.jianshu.io/upload_images/5952841-f2002250600609c5.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
    1. Figure 4:   
    ![img](http://upload-images.jianshu.io/upload_images/5952841-2267b32e43582863.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
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
    ![img](http://upload-images.jianshu.io/upload_images/5952841-f3247c3703845750.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
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

## 第二周 Logistic Regression

### Neural-Network-Basics

10个选择题，原见[Github](https://github.com/liqiang311/deeplearning.ai/blob/master/1_Neural%20Networks%20and%20Deep%20Learning/Quiz-week2-Coursera%20_%20Online%20Courses%20From%20Top%20Universities.pdf)

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
    ![img](http://upload-images.jianshu.io/upload_images/5952841-421f376ba9a3105f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)  
    What is the output J?
    1. `J = (c - 1)*(b + a)`
    1. **`J = (a - 1) * (b + c)`**
    1. `J = a*b + b*c + a*c`
    1. `J = (b - 1) * (c + a)`

### Logistic-Regression-with-a-Neural-Network-mindset

相关数据集和输出见[github](https://github.com/liqiang311/deeplearning.ai/blob/master/1_Neural%20Networks%20and%20Deep%20Learning/week2/Logistic%20Regression%20as%20a%20Neural%20Network/my-Logistic%2BRegression%2Bwith%2Ba%2BNeural%2BNetwork%2Bmindset%2Bv3.ipynb)

## 第三周 浅层神经网络

### Shallow Neural Networks

10个选择题，原见[Github](https://github.com/liqiang311/deeplearning.ai/blob/master/1_Neural%20Networks%20and%20Deep%20Learning/Quiz-week3-Coursera%20_%20Online%20Courses%20From%20Top%20Universities.pdf)

1. Which of the following are true? (Check all that apply.)
    1. **$a^{[2]}$ denotes the activation vector of the $2^{nd}$ layer.**
    1. **$a^{[2] (12)}$ denotes the activation vector of the $2^{nd}$ layer for the $12^{th}$ training example.**
    1. $X$ is a matrix in which each row is one training example.
    1. $a^{[2] (12)}$ denotes activation vector of the $12^{th}$ layer on the $2^{nd}$ training example.
    1. **$X$ is a matrix in which each column is one training example.**
    1. **$a^{[2]}_4$ is the activation output by the $4^{th}$ neuron of the $2^{nd}$ layer**
    1. $a^{[2]}_4$ is the activation output of the $2^{nd}$ layer for the $4^{th}$ training example
1. The tanh activation usually works better than sigmoid activation function for hidden units because the mean of its output is closer to zero, and so it centers the data better for the next layer. True/False?
    1. **True**
    1. False
1. Which of these is a correct vectorized implementation of forward propagation for layer $l$, where $1\leq{}l\leq{}L$?
    1.  - $Z^{[l]}=W^{[l]}A^{[l]}+b^{[l]}$
        - $A^{[l+1]}=g^{[l+1]}(Z^{[l]})$
    1.  - $Z^{[l]}=W^{[l]}A^{[l-1]}+b^{[l]}$ **True**
        - $A^{[l]}=g^{[l]}(Z^{[l]})$ **True**
    1.  - $Z^{[l]}=W^{[l]}A^{[l]}+b^{[l]}$
        - $A^{[l+1]}=g^{[l]}(Z^{[l]})$
    1.  - $Z^{[l]}=W^{[l-1]}A^{[l]}+b^{[l-1]}$
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
    1. **False**
1. You have built a network using the tanh activation for all the hidden units. You initialize the weights to relative large values, using np.random.randn(..,..)*1000. What will happen?
    1. **This will cause the inputs of the tanh to also be very large, thus causing gradients to be close to zero. The optimization algorithm will thus become slow.**
    1. This will cause the inputs of the tanh to also be very large, causing the units to be “highly activated” and thus speed up learning compared to if the weights had to start from small values.
    1. It doesn’t matter. So long as you initialize the weights randomly gradient descent is not affected by whether the weights are large or small.
    1. This will cause the inputs of the tanh to also be very large, thus causing gradients to also become large. You therefore have to set α to be very small to prevent divergence; this will slow down learning.
1. Consider the following 1 hidden layer neural network:  
    ![img](http://upload-images.jianshu.io/upload_images/5952841-26659547ea2d3af5.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
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
    1. **$Z^{[1]}$ and $A^{[1]}$ are (4, m)**
    1. $Z^{[1]}$ and $A^{[1]}$ are (4, 1)
    
### Planar data classification with one hidden layer

相关数据集和输出见[github](https://github.com/liqiang311/deeplearning.ai/blob/master/1_Neural%20Networks%20and%20Deep%20Learning/week3/my-Planar%2Bdata%2Bclassification%2Bwith%2Bone%2Bhidden%2Blayer.ipynb)

## 第四周 深层神经网络

### Key concepts on Deep Neural Network

1. What is the "cache" used for in our implementation of forward propagation and backward propagation?
    1. **We use it to pass variables computed during forward propagation to the corresponding backward propagation step. It contains useful values for backward propagation to compute derivatives.**
    1. We use it to pass variables computed during backward propagation to the corresponding forward propagation step. It contains useful values for forward propagation to compute activations.
    1. It is used to cache the intermediate values of the cost function during training.
    1. It is used to keep track of the hyperparameters that we are searching over, to speed up computation.
1. Among the following, which ones are "hyperparameters"? (Check all that apply.)
    1. **size of the hidden layers $n^{[l]}$**
    1. **number of layers $L$ in the neural network**
    1. **learning rate $α$**
    1. activation values $a^{[l]}$
    1. **number of iterations**
    1. weight matrices $W^{[l]}$
    1. bias vectors $b^{[l]}$
1. Which of the following statements is true?
    1. **The deeper layers of a neural network are typically computing more complex features of the input than the earlier layers.**
    1. The earlier layers of a neural network are typically computing more complex features of the input than the deeper layers.
1. Vectorization allows you to compute forward propagation in an $L$-layer neural network without an explicit for-loop (or any other explicit iterative loop) over the layers l=1, 2, …,L. True/False?
    1. **True**
    1. False
1. Assume we store the values for $n^{[l]}$ in an array called layers, as follows: layer_dims = [$n_x$, 4,3,2,1]. So layer 1 has four hidden units, layer 2 has 3 hidden units and so on. Which of the following for-loops will allow you to initialize the parameters for the model?
    1. Code1
        ```python
        for (i in range(1, len(layer_dims)/2)):
            parameter['w' + str(i)] = np.random.randn(layers[i], layers[i-1]) * 0.01
            parameter['b' + str(i)] = np.random.randn(layers[i], 1) *　0.01
        ```
    1. Code2 
        ```python
        for (i in range(1, len(layer_dims)/2)):
            parameter['w' + str(i)] = np.random.randn(layers[i], layers[i-1]) * 0.01
            parameter['b' + str(i)] = np.random.randn(layers[i-1], 1) *　0.01
        ```
    1. Code3 
        ```python
        for (i in range(1, len(layer_dims))):
            parameter['w' + str(i)] = np.random.randn(layers[i-1], layers[i]) * 0.01
            parameter['b' + str(i)] = np.random.randn(layers[i], 1) *　0.01
        ```
    1. **Code4**
        ```python
        for (i in range(1, len(layer_dims))):
            parameter['w' + str(i)] = np.random.randn(layers[i], layers[i-1]) * 0.01
            parameter['b' + str(i)] = np.random.randn(layers[i], 1) *　0.01
        ```
1. Consider the following neural network.  
    [图片上传失败...(image-4d5dce-1513913878329)]  
    How many layers does this network have?
    1. **The number of layers $L$ is 4. The number of hidden layers is 3.**
    1. The number of layers $L$ is 3. The number of hidden layers is 3.
    1. The number of layers $L$ is 4. The number of hidden layers is 4.
    1. The number of layers $L$ is 5. The number of hidden layers is 4.
1. During forward propagation, in the forward function for a layer $l$ you need to know what is the activation function in a layer (Sigmoid, tanh, ReLU, etc.). During backpropagation, the corresponding backward function also needs to know what is the activation function for layer $l$, since the gradient depends on it. True/False?
    1. **True**
    1. False
1. There are certain functions with the following properties:  
    (i) To compute the function using a shallow network circuit, you will need a large network (where we measure size by the number of logic gates in the network), but (ii) To compute it using a deep network circuit, you need only an exponentially smaller network. True/False?
    1. **True**
    1. False
1. Consider the following 2 hidden layer neural network:  
    ![img](http://upload-images.jianshu.io/upload_images/5952841-7d263eb90f556000.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)  
    Which of the following statements are True? (Check all that apply).
    1. **W[1] will have shape (4, 4)**
    1. **b[1] will have shape (4, 1)**
    1. W[1] will have shape (3, 4)
    1. b[1] will have shape (3, 1)
    1. **W[2] will have shape (3, 4)**
    1. b[2] will have shape (1, 1)
    1. W[2] will have shape (3, 1)
    1. **b[2] will have shape (3, 1)**
    1. W[3] will have shape (3, 1)
    1. **b[3] will have shape (1, 1)**
    1. **W[3] will have shape (1, 3)**
    1. b[3] will have shape (3, 1)
1. Whereas the previous question used a specific network, in the general case what is the dimension of W^{[l]}, the weight matrix associated with layer $l$?
    1. **W[l] has shape (n[l],n[l−1])**
    1. W[l] has shape (n[l+1],n[l])
    1. W[l] has shape (n[l],n[l+1])
    1. W[l] has shape (n[l−1],n[l])

### Building your Deep Neural Network - Step by Step

[ipynb](https://github.com/liqiang311/deeplearning.ai/blob/master/1_Neural%20Networks%20and%20Deep%20Learning/week4/Building%20your%20Deep%20Neural%20Network%20-%20Step%20by%20Step/my-Building%2Byour%2BDeep%2BNeural%2BNetwork%2B-%2BStep%2Bby%2BStep.ipynb)

### Deep Neural Network - Application

[ipynb](https://github.com/liqiang311/deeplearning.ai/blob/master/1_Neural%20Networks%20and%20Deep%20Learning/week4/Deep%20Neural%20Network%20Application%20Image%20Classification/my-Deep%2BNeural%2BNetwork%2B-%2BApplication.ipynb)

# 改善深层神经网络：超参数调试、正则化以及优化

[网址](https://mooc.study.163.com/course/2001281003?tid=2001391036#/info)

## 第一周 深度学习的实用层面

### Practical aspects of deep learning

1. If you have 10,000,000 examples, how would you split the train/dev/test set?
    1. 60% train . 20% dev . 20% test
    1. **98% train . 1% dev . 1% test**
    1. 33% train . 33% dev . 33% test
1. The dev and test set should:
    1. **Come from the same distribution**
    1. Come from different distributions
    1. Be identical to each other (same (x,y) pairs)
    1. Have the same number of examples
1. If your Neural Network model seems to have high variance, what of the following would be promising things to try?
    1. **Add regularization**        
    1. Make the Neural Network deeper
    1. Increase the number of units in each hidden layer
    1. Get more test data
    1. **Get more training data**
1. You are working on an automated check-out kiosk for a supermarket, and are building a classifier for apples, bananas, and oranges. Suppose your classifier obtains a training set error of 0.5%, and a dev set error of 7%, Which of the following are promising things to try to improve your classfier? (Check all that apply.)
    1. **Increase the regularization parameter lambda**
    1. Decrease the regularization parameter lambda
    1. **Get more training data**
    1. Use a bigger neural network
1. What is weight decay?
    1. Gradual corruption of the weights in the neural network if it is trained on noisy data.
    1. A technique to avoid vanishing gradient by imposing a ceiling on the values of the weights.
    1. **A regularization technique (such as L2 regularization) that results in gradient descent shrinking the weights on every iteration.**
    1. The process of gradually decreasing the learning rate during training.
1. What happens when you increase the regularization hyperparameter lambda?
    1. **Weights are pushed toward becoming smaller (closer to 0)**
    1. Weights are pushed toward becoming bigger (further from 0)
    1. Doubling lambda should roughly result in doubling the weights
    1. Gradient descent taking bigger steps with each iteration (proportional to lambda)
1. With the inverted dropout technique, at test time:
    1. You apply dropout (randomly eliminating units) and do not keep the 1/keep_prob factor in the calculations used in training
    1. You do not apply dropout (do not randomly eliminate units), but keep the 1/keep_prob factor in the calculations used in training.
    1. **You do not apply dropout (do not randomly eliminate units) and do not keep the 1/keep_prob factor in the calculations used in training**
    1. You apply dropout (randomly eliminating units) but keep the 1/keep_prob factor in the calculations used in training.
1. Increasing the parameter keep_prob from (say) 0.5 to 0.6 will likely cause the following: (Check the two that apply)
    1. Increasing the regularization effect
    1. **Reducing the regularization effect**
    1. Causing the neural network to end up with a higher training set error
    1. **Causing the neural network to end up with a lower training set error**
1. Which of these techniques are useful for reducing variance (reducing overfitting)? (Check all that apply.)
    1. **Data augmentation**
    1. Gradient Checking
    1. Exploding gradient
    1. **L2 regularization**
    1. **Dropout**
    1. Vanishing gradient
    1. Xavier initialization
1. Why do we normalize the inputs x?
    1. It makes it easier to visualize the data
    1. It makes the parameter initialization faster
    1. Normalization is another word for regularization--It helps to reduce variance
    1. **It makes the cost function faster to optimize**

### Regularization

[ipynb](https://github.com/liqiang311/deeplearning.ai/blob/master/2_Improving%20Deep%20Neural%20Networks/week1_Regularization/my-Regularization.ipynb)

### Initialization

[ipynb](https://github.com/liqiang311/deeplearning.ai/blob/master/2_Improving%20Deep%20Neural%20Networks/week1_initialization/my-Initialization.ipynb)

### Gradient_Checking

[ipynb](https://github.com/liqiang311/deeplearning.ai/blob/master/2_Improving%20Deep%20Neural%20Networks/week1_Gradient_Checking/my-Gradient%2BChecking.ipynb)

## 第二周 优化算法

### Optimization algorithms

1. Which notation would you use to denote the 3rd layer’s activations when the input is the 7th example from the 8th minibatch?
    1. a[8]{3}(7)
    1. a[3]{7}(8)
    1. a[8]{7}(3)
    1. **a[3]{8}(7)**
1. Which of these statements about mini-batch gradient descent do you agree with?
    1. Training one epoch (one pass through the training set) using minibatch gradient descent is faster than training one epoch using batch gradient descent.
    1. **One iteration of mini-batch gradient descent (computing on a single mini-batch) is faster than one iteration of batch gradient descent.**
    1. You should implement mini-batch gradient descent without an explicit for-loop over different mini-batches, so that the algorithm processes all mini-batches at the same time(vectorization).
1. Why is the best mini-batch size usually not 1 and not m, but instead something in-between?
    1. **If the mini-batch size is m, you end up with batch gradient descent, which has to process the whole training set before making progress.**
    1. If the mini-batch size is m, you end up with stochastic gradient descent, which is usually slower than mini-batch gradient descent.
    1. **If the mini-batch size is 1, you lose the benefits of vectorization across examples in the mini-batch.**
    1. If the mini-batch size is 1, you end up having to process the entire training set before making any progress.
1. Suppose your learning algorithm’s cost , plotted as a function of the number of iterations, looks like this:  
    ![img](http://upload-images.jianshu.io/upload_images/5952841-aaeec5fbb925308e.PNG?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)  
    Which of the following do you agree with?
    1. Whether you’re using batch gradient descent or mini-batch gradient descent, this looks acceptable.
    1. **If you’re using mini-batch gradient descent, this looks acceptable. But if you’re using batch gradient descent, something is wrong.**
    1. Whether you’re using batch gradient descent or mini-batch gradient descent, something is wrong.
    1. If you’re using mini-batch gradient descent, something is wrong. But if you’re using batch gradient descent, this looks acceptable.
1. Suppose the temperature in Casablanca over the first three days of January are the same:  
    Jan 1st: $\theta_1 = 10^{\circ}C$  
    Jan 2nd: $\theta_2 = 10^{\circ}C$  
    (We used Fahrenheit in lecture, so will use Celsius here in honor of the metric world.)  
    Say you use an exponentially weighted average with $\beta=0.5$ to track the temperature: $v_0=0,v_t=\beta v_{t-1}+(1-\beta)\theta_t$. If $v_2$ is the value computed after day 2 without bias correction, and $v_2^{corrected}$ is the value you compute with bias correction. What are these values? (You might be able to do this without a calculator, but you don't actually need one. Remember what is bias correction doing.)
    1. $v_2=7.5,v_2^{corrected}=7.5$
    1. $v_2=10,v_2^{corrected}=7.5$
    1. $v_2=10,v_2^{corrected}=10$
    1. $v_2=7.5,v_2^{corrected}=10$ **True**
1. Which of these is NOT a good learning rate decay scheme? Here, t is the epoch number.
    1. $\alpha=\frac{1}{1+2*t}\alpha_0$
    1. $\alpha=e^t\alpha_0$ **True**
    1. $\alpha=0.95^t\alpha_0$
    1. $\alpha=\frac{1}{\sqrt{t}}\alpha_0$
1. You use an exponentially weighted average on the London temperature dataset. You use the following to track the temperature: $v_t=\beta v_{t-1}+(1-\beta)\theta_t$. The red line below was computed using $\beta=0.9$. What would happen to your red curve as you vary $\beta$? (Check the two that apply)  
    ![img](http://upload-images.jianshu.io/upload_images/5952841-c3b2aada9df88a4c.PNG?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
    1. Decreasing $\beta$ will shift the red line slightly to the right.
    1. **Increasing $\beta$ will shift the red line slightly to the right.**
    1. Decreasing $\beta$ will create more oscillation within the red line.
    1. Increasing $\beta$ will create more oscillations within the red line.
1. Consider this figure:  
    ![img](http://upload-images.jianshu.io/upload_images/5952841-6a15f8632ca8cf24.PNG?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)  
    These plots were generated with gradient descent; with gradient descent with momentum ($\beta$ = 0.5) and gradient descent with momentum ($\beta$ = 0.9). Which curve corresponds to which algorithm?
    1. (1) is gradient descent with momentum (small $\beta$). (2) is gradient descent. (3) is gradient descent with momentum (large $\beta$)
    1. (1) is gradient descent. (2) is gradient descent with momentum (large $\beta$) . (3) is gradient descent with momentum (small $\beta$)
    1. **(1) is gradient descent. (2) is gradient descent with momentum (small $\beta$). (3) is gradient descent with momentum (large $\beta$)**
    1. (1) is gradient descent with momentum (small $\beta$), (2) is gradient descent with momentum (small $\beta$), (3) is gradient descent
1. Suppose batch gradient descent in a deep network is taking excessively long to find a value of the parameters that achieves a small value for the cost function $J(W^{[1]},b^{[1]},...,W^{[L]},b^{[L]})$. Which of the following techniques could help find parameter values that attain a small value for $J$? (Check all that apply)
    1. Try initializing all the weights to zero
    1. **Try tuning the learning rate $\alpha$**
    1. **Try better random initialization for the weights**
    1. **Try using Adam**
    1. **Try mini-batch gradient descent**
1. Which of the following statements about Adam is False?
    1. **Adam should be used with batch gradient computations, not with mini-batches.**
    1. We usually use “default” values for the hyperparameters and in Adam ($\beta_1=0.9$,$\beta_2=0.999$,$\varepsilon=10^{-8}$)
    1. The learning rate hyperparameter $\alpha$ in Adam usually needs to be tuned.
    1. Adam combines the advantages of RMSProp and momentum

## 第三周 超参数调试、Batch正则化和程序框架

### Hyperparameter tuning, Batch Normalization, Programming Frameworks

1. If searching among a large number of hyperparameters, you should try values in a grid rather than random values, so that you can carry out the search more systematically and not rely on chance. True or False?
    1. True
    1. **False**
1. Every hyperparameter, if set poorly, can have a huge negative impact on training, and so all hyperparameters are about equally important to tune well. True or False?
    1. True
    1. **False**
1. During hyperparameter search, whether you try to babysit one model (“Panda” strategy) or train a lot of models in parallel (“Caviar”) is largely determined by:
    1. Whether you use batch or mini-batch optimization
    1. The presence of local minima (and saddle points) in your neural network
    1. **The amount of computational power you can access**
    1. The number of hyperparameters you have to tune
1. If you think $\beta$(hyperparameter for momentum) is between on 0.9 and 0.99, which of the following is the recommended way to sample a value for beta?
    1. Code1
        ```python
        r = np.random.rand()
        beta = r*0.09 + 0.9
        ```
    1. **Code2**
        ```python
        r = np.random.rand()
        beta = 1-10**(- r - 1)
        ```
    1. Code3
        ```python
        r = np.random.rand()
        beta = 1-10**(- r + 1)
        ```
    1. Code4
        ```python
        r = np.random.rand()
        beta = r*0.9 + 0.09
        ```
1. Finding good hyperparameter values is very time-consuming. So typically you should do it once at the start of the project, and try to find very good hyperparameters so that you don’t ever have to revisit tuning them again. True or false?
    1. True
    1. **False**
1. In batch normalization as presented in the videos, if you apply it on the $l$th layer of your neural network, what are you normalizing?
    1. $z^{[l]}$**True**
    1. $W^{[l]}$
    1. $a^{[l]}$
    1. $b^{[l]}$
1. In the normalization formula $z^{(i)}_{norm} =\frac{z^{(i)}-\mu}{\sqrt{\sigma^2+\varepsilon}}$, why do we use epsilon?
    1. To have a more accurate normalization
    1. **To avoid division by zero**
    1. In case μ is too small
    1. To speed up convergence
1. Which of the following statements about $\gamma$ and $\beta$ in Batch Norm are true?
    1. **They can be learned using Adam, Gradient descent with momentum, or RMSprop, not just with gradient descent.**
    1. β and γ are hyperparameters of the algorithm, which we tune via random sampling.
    1. **They set the mean and variance of the linear variable z[ l] of a given layer.**
    1. There is one global value of $\gamma \in R$ and one global value of $\beta \in R$ for each layer, and applies to all the hidden units in that layer.
    1. The optimal values are γ = $\sqrt{\sigma^2+\varepsilon}$, and β
1. After training a neural network with Batch Norm, at test time, to evaluate the neural network on a new example you should:
    1. Use the most recent mini-batch’s value of μ and σ2 to perform the needed normalizations
    1. If you implemented Batch Norm on mini-batches of (say) 256 examples, then to evaluate on one test example, duplicate that example 256 times so that you’re working with a mini-batch the same size as during training.
    1. **Perform the needed normalizations, use μ and σ2 estimated using an exponentially weighted average across mini-batches seen during training.**
    1. Skip the step where you normalize using and since a single test example cannot be normalized.
1. Which of these statements about deep learning programming frameworks are true? (Check all that apply)
    1. **Even if a project is currently open source, good governance of the project helps ensure that the it remains open even in the long term, rather than become closed or modified to benifit only one company.**
    1. **A programming framework allows you to code up deep learning algorithms with typically fewer lines of code than a lower-level language such as Python.**

### TensorFlow Tutorial

[ipynb]()

# 结构化机器学习项目

## 机器学习(ML)策略1

## 机器学习(ML)策略2

# 卷积神经网络

## 第一周 卷积神经网络

## 第二周 深度卷积神经网络
