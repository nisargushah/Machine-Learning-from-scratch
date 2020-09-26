# Naive Bayes 

This document will guide you through how the naive bayes work and how it is implemented in the code. Please use the Table of content if you want to skip a part



#Table of Content

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


**IMPORTANT - Please note that this is just my imterpretation of Naive Bayes as I understand it. The content might differ from source to source. However, I have correctly implemented the code as is, atleast according to my class curriculum. I appreciate your attention on this**
