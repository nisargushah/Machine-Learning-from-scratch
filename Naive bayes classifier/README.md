# Naive Bayes 

This document will guide you through how the naive bayes works and how it is implemented in the code. Please use the Table of contents if you want to skip a part



# Table of Contents

* [Theory behing Naive Bayes](#theory-behind-naive-bayes)
  * [What is a Classification problem ?](#what-is-classification-problem)
  * [Bayes Classifer](#bayes-classifier)
  * [The Naive in Naive Bayes Classifier](#the-naive-in-naive-bayes-classifier)
* [Code Explanation](#code-explaination)
  
  
 
 # Theory behind Naive bayes
 
 In this part I will try to explain how naive bayes work and why do we use it. If you want to jump right back at the code, [click here](#code-explaination)
 
 ## What is a classification problem?
 
 While trying to understand what Naive Bayes Classifier does, we need to first understand that does a classifier mean. 
 Well according to [Wikipedia](https://en.wikipedia.org/wiki/Statistical_classification) classifier mean
 > "An algorithm that implements classification, especially in a concrete implementation, is known as a classifier. The term "classifier" sometimes also refers to the mathematical function, implemented by a classification algorithm, that maps input data to a category."
    

In other terms, classification problem is like choosing which basket to put our oranges in or classifying if the given image is a dog or a cat.

## Bayes Classifier

There are many ways to classifiy objects and exampels but the most optimal way to find which class a certain object or examples belong to is Bayes classfier.

But why ? Because Bayes classfication derives probability of a certain object, if that object were to be of, lets say, class A.

So Bayesian classifier or Bayes classifier find the probability that it will be class A given our example. 

Or in mathematical form using bayes rule, we can define the probability as: 

<a href="https://www.codecogs.com/eqnedit.php?latex=p(C_{k}|x)&space;=&space;\frac{p(x|C_{k})*p(C_{k})}{p(x)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p(C_{k}|x)&space;=&space;\frac{p(x|C_{k})*p(C_{k})}{p(x)}" title="p(C_{k}|x) = \frac{p(x|C_{k})*p(C_{k})}{p(x)}" /></a>

We find the probablity of all classes k , and then we find the class that has the maximum probability. This class will be our final answer. If you think about it this should be the the correct and the most optimal answer since we need to find the P(C<sub>k</sub>|x) anyways. This is our problem statement and if we can just find which class has the max probability in this, we can get the correct answer 100%.

So why don't we just use it every time ? So as it turns out its quite diffuclt to find P(x|C<sub>k</sub>). To solve this we nneed to find the probability of x in all the classes. But more often than not, we cannot find this or it is incredibly difficult to solve this. There is the problem that we cannot computer P(x|C<sub>k</sub>) unless every attribute in a class is independent of each other and then we need to have enough data for each class to compute this. And we can never be 100% sure if we know we have all the possible classses like ,let say, for facial recognition. Its quite difficult to get images of all 8 billion or so people living on earth. So bayes Classifier is not used in almost all the cases. Also computation can only be done if all the classes are independent of each other. So if lets say we are trying to classify is the temperature is going to be higher than 25 degree C in Sahara in the summer season, we know that Sahara desert is extremely hot in the summer. SO if we were to bet on the temperature going high than 25 degree C, we will say most certainly yes because we know from past experience or knowledge that the temperature reaches 40 or even 50+ degreee C inn Sahara during the summers. But this knowledge makes it difficult to almost impossible to computer Bayesian classifisation because if we have any prior knowledge, the temperature is no longer independent each day. The "attributes" or the temperatiure each day are not independent of each other.

What do I mean by independent ? lets take the same example. If I say that for the past 10 days the temperature at a location X has been higher than 10 degreee C. Will we bet on the temperature will be more than 10 degree C today as well ? Well most certainly yes. But I were to say at a location Y, whats the probability that the temperature will go above 10 degree C ?, We will say we don't know. We don't know if its Antartica in winters ot Sahara in Summers. We cannot definetly say. This means that the temperature's everyday are independent of each other. As you can tell, this slims down real life problems we can do to just a small fraction.

## The Naive in Naive Bayes Classifier

As we discussed earlier, we can only calculate the bayesian probability if we have independent attributes. So naiver bayes assumes that the attributes and classes are independent of each other. Why do we do that ? A) Now we can compute the Bayesian probasbility even if its not exactly the same but still we can computer it
B) For some dataset we don't know if the attributes are independent off each other. We can try and see what are the results we get. If we get enough accuracy on both train and test, than why not use it ? So how do we calculate that ?

So we have this formula : 
<a href="https://www.codecogs.com/eqnedit.php?latex=p(C_{k}|x)&space;=&space;\frac{p(x|C_{k})*p(C_{k})}{p(x)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p(C_{k}|x)&space;=&space;\frac{p(x|C_{k})*p(C_{k})}{p(x)}" title="p(C_{k}|x) = \frac{p(x|C_{k})*p(C_{k})}{p(x)}" /></a>

Now we can find the prior probability with the help of our data. We can do this by: 

<a href="https://www.codecogs.com/eqnedit.php?latex=Prior&space;Probability&space;for&space;C_{k}&space;=\frac{len(C_{k})}{len(TrainData)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Prior&space;Probability&space;for&space;C_{k}&space;=\frac{len(C_{k})}{len(TrainData)}" title="Prior Probability for C_{k} =\frac{len(C_{k})}{len(TrainData)}" /></a>

We can just take how many number of examples have C<sub>k</sub> as their class and divide it by total number of examples. 

Next is p(x|C<sub>k</sub>). Where x is the example. Since they are independent of each other, we can just multiply that amongst each other. 

For this we will use Gausssian pdistribution for each attrbute in each class. This makes computation easier and we still get to the same answer.
The gausssian FUnction is defined as : 

<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{150}&space;\large&space;N(x)&space;=&space;\frac{1}{\sigma*\sqrt[]{2\pi}}e^{-\frac{(x-\mu)^{2}}{2\sigma&space;^{2}}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;\large&space;N(x)&space;=&space;\frac{1}{\sigma*\sqrt[]{2\pi}}e^{-\frac{(x-\mu)^{2}}{2\sigma&space;^{2}}}" title="\large N(x) = \frac{1}{\sigma*\sqrt[]{2\pi}}e^{-\frac{(x-\mu)^{2}}{2\sigma ^{2}}}" /></a>


We will see how to implement it in the code section: 

## Code Explaination

The first step we do is to import all the essential libraries: 

```python

import sys
import math
import numpy as np
from statistics import stdev as stdev
```
So lets create out function naive_bayes. We will take the location of train file and test file as an argument so that we can read the files.

```python
def naive_bayes(train_file, test_file):
    try:
        import pandas as pd
        train_data = pd.read_csv(train_file, header = None)
        test_data = pd.read_csv(test_file, header=None)
        X_test  = train_data.iloc[:,:-1]
        X_test = X_test.astype(np.float)
        y_test  = train_data.iloc[:,-1]
        y_test = y_test.astype(np.int)
        X_train = test_data.iloc[:,-1]
        X_train = X_train.astype(np.float)
        y_train = test_data.iloc[:,-1]
        y_train = y_train.astype(np.int)
        #print("From Pandas")
    except:
        train_data = np.genfromtxt(train_file)
        test_data = np.genfromtxt(test_file)
        X_test = test_data[:, :-1]
        X_train = train_data[:, :-1]
        y_test = test_data[:, -1]
        y_train = train_data[:, -1]
        X_test = X_test.astype(np.float)
        y_test = y_test.astype(np.int)
        X_train = X_train.astype(np.float)
        y_train = y_train.astype(np.int)


```

now that a big chunk of code, lets break it down. 

Since we are keeping pandas as optional, its nice to check if the user has already installed pandas. it provides us with fabulous dataframes which are easy to navigate thorugh. But its not required. If you look closely, we are also seperating four variables, X_train, X_test, y_trai, y_test. As you might have guessed, X_train contains all the attributes except the last column ,which is our "class" column or the targt column, from the train file and same with the test file.
Also we know that class numbers are only integers, so we convert them to np.int format and the attributes are float, so comverting them respectively as well. Note we can do this in the same step. But I wanted to be as exlpicit as possible.

y_train and y_test contains list of all the class numbers for that index example in the train or test file. This is usally the first step, sperating the target column from train and test file.



The next step that we will do is calculate all the means and std deviation so that we can find the gaussains. 

So we will do : 

```python
    class_means = []
    class_std = []
    #print(X_train[0][1])
    indexes = []
    y_train = np.asarray(y_train)
    for i in range(1,11):
        x = np.where(y_train == i)
        #print(x)
        x = np.asarray(x)
        temp = []
        #print(x[0])
        for j in range(0, x.shape[1]):
            temp.append(X_train[x[0][j],:])
        #print(temp)
        temp = np.asarray(temp)
        for j in range(0,8):
            temp2 = temp[:,j]
            temp2 = np.asarray(temp2)
            mean = temp2.mean()
            #std  = temp2.std()
            std = stdev(temp2)

            """
            We don't want our standard deviation to be 0
            It can mess up out calculation

            """
            if std<=0.01:
                std = 0.01
            class_means.append(mean)
            class_std.append(std)
            #print("mean %.2f std %.2f" %(mean, std))
```

So lets break this chunk down. The first thing we need is a list to store all the mean and stdevs. Note that since we need to calculate the mean and stddev for all attributes of all classes, so lets say we have 10 classes and 8 attributes. So in total we need to find 80 mean and stdevs.

We need to gather all the indexes where lets say class number is 1 so that we can find its mean and stdevs. We do that by x =np.where(y_train==i) inside the loop where i goes from 1 to 10, like our class numbers. We find the attrbutes and then store them in their respective arrays. If the stdev is less than 0.01, we reolace it with 0.01 since we dont want to mess up with our gaussian in whichwe divide it by sigma.


After that lets define our main function. Its always a good practice in python to have this. Also we will be taking in the location of the train ffile and test file from the commannd line itself. So lets read that as well: 

[ ] To be completed

```python
if __name__ == "__main__":
    naive_bayes(sys.argv[1],sys.argv[2])
```
Here feew things are happing, sys.argv[1] and [2] will take the inputs and you will see that we have already deined a naive bayes function.


**IMPORTANT - Please note that this is just my imterpretation of Naive Bayes as I understand it. The content might differ from source to source. However, I have correctly implemented the code as is, atleast according to my class curriculum. I appreciate your attention on this**
