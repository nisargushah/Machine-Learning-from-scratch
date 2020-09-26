# About this Project 
The rise of python libraries in Machine Learning and Deep Learning is truly amazing. However according to my Professor, [Vassillis Athitos](http://vlm1.uta.edu/~athitsos/), getting a basic 
undertanding of how to implement various functions and algorithms is important to which I strongly agree. In this repository I will attempt to develop popular 
Machine Learning Algorithms for my [CSE 4309 - Fundamentals of Machine Learning](http://vlm1.uta.edu/~athitsos/courses/cse4309_fall2020/assignments/) class.

## Table of Content

* [Setting up the environment](#setting-up-the-environment)
  * [Setting up Anaconda](#setting-up-anaconda)
  * [For Winndows Users](#for-windows-user)
* [Cloning the repository](#cloning-the-repository)
* [Installing Dependencies](#installing-dependencies)
* [Running the program](#running-the-program)
  * [About Dataset](#about-dataset)
  * [Naive Bayes](#naive-bayes)
    * [How does Naive Bayes work ? ](#how-does-naive-bayes-work)
    * [How to run naive Bayes program](#how-to-run-the-program)

## Setting up the environment

We will be usinng Python3 for this project. Here's how to set up an environment. 

### Setting up Anaconda

[Anaconda](https://www.anaconda.com/) is an amazing package source for python and I highly recommend downloading it from [here](https://www.anaconda.com/products/individual)


After you have installed the Anaconda package, go to your terminal and navigate to where you 
write up the following commands..
```console
foo@bar:~$ conda create -n myenv python==3.8.3
```
After coping this command the terminal will ask you if you want to continue, press y and then enter
Windows users might have to go to their Anaconda bash terminal which they can go to using the search bar. Just type Anaconda bash and it should appear
Let me know if it doesnt work !


## Cloning the repository

This repository has MIT license. 

Go to your desired directory where you want to clone this repo in the terminal and then 

```console
foo@bar:~$ git clone https://github.com/nisargushah/Machine-Learning-from-scratch.git
```

Now we have succesfully cloned the repository. The next step is to activate the environment

## Installing Dependencies

We are trying to implement a raw version of this algorithms but we will use numpy and pandas libraries here to make our life a little easier while still maintianing 
the raw implementation.

To do this first we will activate the enviornmennt and then install them 

```console
foo:~$ conda activate myenv
```

After typing this you should see that (myenv) appears before your every bash line (Please note that it won't work if you have customized your terminal with powerline or any other tool )

```console
(myenv) foo@bar:~$ pip install numpy==1.18.5
```
After getting the message of successfully istallation we will install pandas.

```console
(myenv) foo@bar:~$ pip install pandas==1.0.5
```

## Runninng the Program

After we have successfully installed the dependecies the next step is to start running the programs!!

However due to various different hyperparameters, each implementation is different. However we will use the same dataset for all the functionality 

### About Dataset

This dataset is part of [UCI dataset](https://archive.ics.uci.edu/ml/datasets.php). This is an amzing dataset library maintained by 
[University of California at Irvine](https://uci.edu/). It is motly, according to my knowledge, public source. So anyone is free to clone the datasets. 

The Data set we are using is not a big dataset with just 7494 training objects and 3498 test objects with classification to be done on 10 classes.

Each row constitues as an object with its columns as feautes ( 16 in total ) with the last one being the class of its object or example. 

Both the training file and the test file are text files, containing data in tabular format. Each value is a number, and values are separated by white space. 
The i-th row and j-th column contain the value for the j-th dimension of the i-th object. The only exception is the LAST column, that stores the class label 
for each object

The above abstract of dataset is taken from my [course website](http://vlm1.uta.edu/~athitsos/courses/cse4309_fall2020/assignments/uci_datasets/dataset_description.html)
You can finnd the dataset there as well. I have some illustrative examples if the above text is not clear [here](https://github.com/nisargushah/Machine-Learning-from-scratch/blob/master/dataset/READme.md)


### Naive Bayes

#### How does Naive Bayes theorem work ? 



#### How to run the program
