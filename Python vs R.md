Python vs R
==================

Brought to you by [Lesley Cordero](http://www.columbia.edu/~lc2958) and [ADI](https://adicu.com)

This tutorial assumes at least some basic knowledge in one of the languages. The goal of this tutorial is not to teach both languages, but rather to teach how commonly used skills can be transferred from one to the other.

To get the most out of this tutorial, some basic understanding of machine learning is also helpful, though not necessarily required.

## Table of Contents

- [0.0 Setup](#00-setup)
    + [0.1 Python and Pip](#01-python-and-pip)
    + [0.2 R & R Studio](#02-r--r-studio)
    + [0.3 Other](#03-other)
- [1.0 Introduction](#10-introduction)
    + [1.1 The Data](#11-the-data)
- [2.0 Data Preparation & Basic Functionality](#20-data-preparation--basic-functionality)
    + [2.1 Reading the Data](#21-reading-the-data)
        * [2.1.1 CSV Files](#211-csv-files)
        * [2.1.2 Viewing the Data](#212-viewing-the-data)
        * [2.2.3 Simple Stats](#213-simple-stats)
    + [2.2 Splitting the Data](#22-splitting-the-data)
- [3.0 Data Visualization](#30-data-visualization)
    + [3.1 Scatter Plots](#31-scatter-plots)
        * [3.1.1 Python ggplot](#311-python-ggplot)
        * [3.1.2 R ggplot](#312-r-ggplot)
    + [3.2 Clustering](#32-clustering)
    + [3.3 Random Forests](#33-random-forests)
- [4.0 Error Evaluation](#40-error-evaluation)
- [5.0 Differences Overview](#40-differences-overview)
    + [5.1 Statistical Support](#51-statistical-support)
    + [5.2 Non-Statistical Support](#52-non-statistical-support)
    + [5.3 Packages](#53-packages)
    + [5.4 Functional vs Object-Oriented](#54-functional-vs-object-oriented)
    + [5.5 When do we use each then?](#55-when-do-we-use-each-then)
        * [5.5.1 Python](#551-python)
        * [5.5.2 R](#552-r)
- [6.0 Final Words](#60-final-words)
    + [6.1 Resources](#61-resources)


## 0.0 Setup

This guide was written in Python 2.7 and R 3.2.3.

### 0.1 Python & Pip

Download [Python](https://www.python.org/downloads/) and [Pip](https://pip.pypa.io/en/stable/installing/).

### 0.2 R & R Studio

Download [R](https://www.r-project.org/) and [R Studio](https://www.rstudio.com/products/rstudio/download/).

### 0.3 Other

``` R
install.packages('dplyr')
install.packages('data.table')
install.packages('lubridate')
install.packages('jsonlite')
install.packages('tidyr')
install.packages('ggplot2')
install.packages('compare')
```

## 1.0 Introduction

Python and R are two commonly used programming languages in the realm of data science. Some data scientists prefer R, others prefer Python; regardless, both are useful programming languages to feel comfortable with if you're interested in Data Science. With that said, in this tutorial we'll go through data analysis problems in both languages, making sure to highlight differences between the two languages. 

### 1.1 The Data 

Since it's election season, we'll get in the spirit and analyze some the Iowa Primary Election Results! You can download the dataset below:

[Primary Election Results]()

## 2.0 Data Preparation & Basic Functionality

### 2.1 Reading the Data

The data is located within a csv file, so we'll start off by reading the data so that we can we can perform analysis later in this tutorial.

#### 2.1.1 CSV Files

The following snippet of uses the pandas module to easily open and read the file.

``` python
import pandas
    
results = pandas.read_csv("./results.csv")
``` 

Meanwhile, in R, we can do this in one line:

``` R
results <- read.csv("./results.csv")
```

The only real difference here is that Python requires a module to access the function that reads in a CSV file; for R, this comes built-in.

#### 2.2.2 Viewing the Data

Now, let's take a look at the actual data through Python and R functionality. First, let's take a look at the header column and its first 10 rows.

In Python, we do this with: 

``` python
results.head(10)
```

Similarly, in R: 

``` R
head(results, 10)
```

Pretty straightforward!

#### 2.2.3 Simple Stats

One very simple thing we can do in just one line is find the average number of votes each candidate receieved in the Iowa Primary. 

```python 
results.mean()
```

``` R
sapply(results, mean, na.rm=TRUE)
```

### 2.2 Splitting the Data

#### 2.2.1 Python Training & Test

``` python
training = results.sample(frac=0.8, random_state=1)
testing = results.loc[~results.index.isin(train.index)]
```

#### 2.2.2 R Training & Test

``` R
train_count <- floor(0.8 * nrow(results))
set.seed(1)
index <- sample(1:nrow(results), train_count)

training <- results[index,]
testing <- results[-index,]
```

## 3.0 Data Visualization

### 3.1 

```python
from ggplot.exampledata import diamonds
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("white")
sns.lmplot("carat", "price", col="cut", data=diamonds, order=2)
```
And as usual, to show the visualization, enter:
``` python
plt.show()
```

If your data analysis needs integration with a web application or database, Python is probably your best bet. Compared to R, the support for these sorts of application is much better since it's more of a general-purpose language.

Whereas in R, you can do the exact same thing with these lines of code:

```R
library(ggplot2)
library(dplyr)
data(diamonds)
diamonds %>% 
ggplot(aes(x=carat,y=price)) + 
geom_point(alpha=0.5) +
facet_grid(~ cut) + 
stat_smooth(method = lm, formula = y ~ poly(x,2)) + 
theme_bw()
```

Meanwhile, if your data analysis demands standalone computing or exploratory work, R is a great choice because of its strong statistical support.

### 3.2 Scatter Plots

#### 3.2.1 Python ggplot

``` python 
from ggplot.exampledata import diamonds

import seaborn as sn

sn.set_style("white")
sn.lmplot("carat","price",col="cut",data=diamonds, order=2)
sn.plt.show()
```

#### 4.2.2 R ggplot2

``` R
library(ggplot2)
library(dplyr)
data(diamonds)
diamonds %>$
ggplot(aes(x=carat,y=price)) + 
geom_point(alpha=0.5) + facet_grid(~ cut) +
stat_smooth(method=lm,formula=y ~ ploy(x,2)) + theme_bw()
```

Notice that these visualizations have a much better scaling of the y-axis; this is because R automatically scales to the actual data rather than the model fit. 


## 4.0 Analysis

### 4.1 Time Series

As always, let's import the needed modules first. 

``` python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
```

Every data visualization is going to be customized to your liking, in terms of aesthetic. Here, we feed in the specifics of how our time series graph should look. 

``` python
sns.set()
sns.set_context('notebook', font_scale=1.5)
cp = sns.color_palette()
```

In this example, we'll be creating a random time series -- here we'll load in the data with pandas. 

``` python
ts = pd.read_csv('./ts.csv')
```

Here we cast our data to datetimes so we can make a timeseries plot.

``` python
ts = ts.assign(dt = pd.to_datetime(ts.dt))
```

The FacetGrid is an object that links a Pandas DataFrame to a matplotlib figure with a particular structure. Here, we're just initializing it. 

``` python
g = sns.FacetGrid(ts, hue='kind', size=5, aspect=1.5)
```

Using the map() function, we plot each subset and give it the final aesthetic details. 

``` python
g.map(plt.plot, 'dt', 'value').add_legend()
g.ax.set(xlabel='Date', ylabel='Value', title='Random Timeseries')
g.fig.autofmt_xdate()
```

Now let's take a look!

``` python
plt.show()
```


``` python
df = pd.read_csv('./iris.csv')
```

Once again, we call the FacetGrid object for our scatterplot and follow up by plotting the scatter plots with its aesthetical specifics. 
``` python
g = sns.FacetGrid(df, hue='species', size=7.5)
g.map(plt.scatter, 'petalLength', 'petalWidth').add_legend()
g.ax.set_title('Petal Width v. Length -- by Species')
```

Now let's take a look! 

``` python
plt.show()
```




### 4.2 Clustering 

### 4.3 Random Forests

``` python
import numpy as np
import pylab as pl

x = np.random.uniform(1, 100, 1000)
y = np.log(x) + np.random.normal(0, .3, 1000)

pl.scatter(x, y, s=1, label="log(x) with noise")
pl.plot(np.arange(1, 100), np.log(np.arange(1, 100)), c="b", label="log(x) true function")
pl.xlabel("x")
pl.ylabel("f(x) = log(x)")
pl.legend(loc="best")
pl.title("A Basic Log Function")
pl.show() 
```

``` python
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
df.head()

train, test = df[df['is_train']==True], df[df['is_train']==False]

features = df.columns[:4]
clf = RandomForestClassifier(n_jobs=2)
y, _ = pd.factorize(train['species'])
clf.fit(train[features], y)

preds = iris.target_names[clf.predict(test[features])]
pd.crosstab(test['species'], preds, rownames=['actual'], colnames=['preds'])
```

## 5.0 Differences Overview

### 5.1 Statistical Support

### 5.2 Non-Statistical Support

### 5.3 Packages

### 5.4 Functional vs Object-Oriented

### 5.5 When do we use each then? 

#### 5.5.1 Python 

If your data analysis needs integration with a web application or database, Python is probably your best bet. Compared to R, the support for these sorts of application is much better since it's more of a general-purpose language. 


#### 5.5.2 R 

Meanwhile, if your data analysis demands standalone computing or exploratory work, R is a great choice because of its strong statistical support. 


## 6.0 Final Words

### 6.1 Resources

[Public Datasets](https://github.com/caesar0301/awesome-public-datasets) <br>
[Kaggle](https://www.kaggle.com/)

