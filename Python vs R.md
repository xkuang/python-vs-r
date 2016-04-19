Intro to Geospatial Data Analysis in Python 
==================

Brought to you by [Lesley Cordero](http://www.columbia.edu/~lc2958) and [ADI](https://adicu.com)

## Table of Contents

- [0.0 Setup](#00-setup)
    + [0.1 Python and Pip](#01-python-and-pip)
    + [0.2 R & R Studio](#02-r--r-studio)
    + [0.3 Other](#03-other)
- [1.0 Introduction](#10-introduction)
    + [1.1 ](#11-what-is-geospatial-data-analysis)
    + [1.2 ](#12-understanding-the-data)
- [2.0 Data Preparation](#20-data-preparation)
    + [2.1 Reading the Data](#21-reading-the-data)
        * [2.1.1 Python](#211-python)
        * [2.1.2 R](#212-r)
- [3.0 Plotting](#30-plotting)
    + [3.1 Scatter Plots](#31-scatter-plots)
    + [3.2 Clustering](#32-clustering)
    + [3.3 Random Forests](#33-random-forests)
- [4.0 Error Evaluation](#40-error-evaluation)
- [5.0 Differences Overview](#40-differences-overview)
    + [5.1 Statistical Support](#51-statistical-support)
    + [5.2 Non-Statistical Support](#52-non-statistical-support)
    + [5.3 Packages](#53-packages)
    + [5.4 Functional vs Object-Oriented](#54-functional-vs-object-oriented)
- [6.0 Final Words](#60-final-words)
    + [6.1 Resources](#61-resources)


## 0.0 Setup

This guide was written in Python 2.7 and R 3.2.3.

### 0.1 Python & Pip

Download [Python](https://www.python.org/downloads/) and [Pip](https://pip.pypa.io/en/stable/installing/).

### 0.2 R & R Studio

Install [R](https://www.r-project.org/) and [R Studio](https://www.rstudio.com/products/rstudio/download/).

### 0.3 Other

```
pip install pandas
pip install sklearn
pip install seaborn
```

## 1.0 Introduction

Python and R are two commonly used programming languages in the realm of data science. Some data scientists prefer R, others prefer R; regardless, both are useful programming languages to feel comfortable with if you're interested in Data Science. With that said, in this tutorial we'll go through a data analysis problem in both languages, making sure to highlight differences between the two languages. 

### 1.1 The Data 

Since it's election season, we'll get in the spirit and analyze some the Iowa Primary Election Results! You can download the dataset below:

[Primary Election Results]()

## 2.0 Data Preparation 

### 2.1 Reading the Data

The data is located within a csv file, so we'll start off by reading the data so that we can perform analysis.

#### 2.1.1 Python

The following snippet of uses the pandas module to easily open and read the file.

``` python
import pandas
    
results = pandas.read_csv("./results.csv")
```

#### 2.1.2 R 

Meanwhile, in R, we can do this in one line:

``` R
results <- read.csv("./results.csv")
```

The only real difference here is that Python requires a module to access the function that reads in a CSV file.

### 2.2 Splitting the Data

#### 2.2.1 Python Training & Test

``` python
training = results.sample(frac=0.8, random_state=1)
testing = results.loc[~results.index.isin(train.index)]

#### 2.2.2 R Training & Test
```
``` R
train_count <- floor(0.8 * nrow(results))
set.seed(1)
index <- sample(1:nrow(results), train_count)

training <- results[index,]
testing <- results[-index,]
```
## 3.0 Plotting

### 3.1 Scatter Plots

### 3.2 Clustering 

### 3.3 Random Forests

## 4.0 Error Evaluation

## 5.0 Differences Overview

### 5.1 Statistical Support

### 5.2 Non-Statistial Support

### 5.3 Packages

### 5.4 Functional vs Object-Oriented

## 6.0 Final Words

### 6.1 Resources

[Public Datasets](https://github.com/caesar0301/awesome-public-datasets) <br>
[Kaggle](https://www.kaggle.com/)

