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

This guide was written in Python 3.5 and R 3.2.3.

### 0.1 Python & Pip

Make sure to download [Python](https://www.python.org/downloads/) and [Pip](https://pip.pypa.io/en/stable/installing/).

### 0.2 R & R Studio

Make sure to download [R](https://www.r-project.org/) and [R Studio](https://www.rstudio.com/products/rstudio/download/).

### 0.3 Other

We'll be using a variety of packages and modules, all of which you should have installed by entering the following:

``` 
pip3 install seaborn
pip3 install pandas
pip3 install sklearn
```

And R: 

``` R
install.packages('dplyr')
install.packages('data.table')
install.packages('GGally')
install.packages('jsonlite')
install.packages('tidyr')
install.packages('ggplot2')
install.packages('compare')
```

## 1.0 Introduction

Python and R are two commonly used programming languages in the realm of data science. Some data scientists prefer R, others prefer Python; regardless, both are useful programming languages to feel comfortable with if you're interested in Data Science. With that said, in this tutorial we'll go through data analysis problems in both languages, making sure to highlight differences between the two languages and why having both skillsets is important.  

### 1.1 The Data 

For this tutorial, we'll be analyzing a dataset of NBA players and their performance in the 2013-2014 season. You can download the dataset below:

[Primary Election Results](https://github.com/lesley2958/python-vs-r/blob/master/results.csv)

## 2.0 Data Preparation & Basic Functionality

### 2.1 Reading the Data

The data is located within a csv file, so we'll start off by reading the data so that we can we can perform analysis later in this tutorial.

#### 2.1.1 CSV Files

The following snippet of code uses the pandas module to easily open and read the file.

``` python
import pandas
nba = pandas.read_csv("nba_2013.csv")
``` 

Meanwhile, in R, we can do this in one line:

``` R
nba <- read.csv("nba_2013.csv")
```
The only real difference is that in Python, we need to import the pandas library to get access to Dataframes. Dataframes are available in both R and Python, and are two-dimensional arrays (matrices) where each column can be of a different datatype. At the end of this step, the csv file has been loaded by both languages into a dataframe.

#### 2.2.2 Viewing the Data

Now, let's take a look at the actual data through Python and R functionality. First, let's take a look at the header column and its first 5 rows.

In Python, we do this with: 

``` python
nba.head(5)
```

Similarly, in R: 

``` R
head(nba, 5)
```

Pretty straightforward!

#### 2.2.3 Simple Stats

One very simple thing we can do in just one line is mean of each attribute:

```python 
nba.mean()
```

``` R
sapply(nba, mean, na.rm=TRUE)
```

In both, we’re applying a function across the dataframe columns. But in python, the mean method on dataframes will find the mean of each column by default.

In R, taking the mean of string values will just result in NA – not available. However, we do need to ignore NA values when we take the mean (requiring us to pass na.rm=TRUE into the mean function). If we don’t, we end up with NA for the mean of columns like x3p.. This column is three point percentage. Some players didn’t take three point shots, so their percentage is missing. If we try the mean function in R, we get NA as a response, unless we specify na.rm=TRUE, which ignores NA values when taking the mean. The .mean() method in Python already ignores these values by default.


#### 2.3 Training & Test Data

Since we'll be doing supervised machine learning later in this workshop, it’s a good idea to split the data into training and testing sets so we don’t overfit.

``` R
trainRowCount <- floor(0.8 * nrow(nba))
set.seed(1)
trainIndex <- sample(1:nrow(nba), trainRowCount)
train <- nba[trainIndex,]
test <- nba[-trainIndex,]
```

``` python
train = nba.sample(frac=0.8, random_state=1)
test = nba.loc[~nba.index.isin(train.index)]
```

Notice that R has more data analysis focused builtins, like floor, sample, and set.seed, whereas these are called via packages in Python (math.floor, random.sample, random.seed). In Python, the recent version of pandas came with a sample method that returns a certain proportion of rows randomly sampled from a source dataframe – this makes the code much more concise. In R, there are packages to make sampling simpler, but aren’t much more concise than using the built-in sample function. In both cases, we set a random seed to make the results reproducible.



## 3.0 Data Visualizations

<<<<<<< HEAD
### 3.1 Pairwise Scatterplots

One common way to explore a dataset is to see how different columns correlate to others. We’ll compare the ast, fg, and trb columns.

In R, we'll do this using the GGally package.

=======
#### 2.2.2 R Training & Test
```
>>>>>>> 2f2993b4e7befff0c62ffbda8029a5fe874efbf8
``` R
library(GGally)
ggpairs(nba[,c("ast", "fg", "trb")])
```
And in python, we'll use seaborn and matplotlib:

``` python
import seaborn as sns
import matplotlib.pyplot as plt
sns.pairplot(nba[["ast", "fg", "trb"]])
plt.show()
```

We get very similar plots, but this shows how the R data science ecosystem has many smaller packages (GGally is a helper package for ggplot2, the most-used R plotting package), and many more visualization packages in general. In Python, matplotlib is the primary plotting package, and seaborn is a widely used layer over matplotlib. With visualization in Python, there is usually one main way to do something, whereas in R, there are many packages supporting different methods of doing things (there are at least a half dozen packages to make pair plots, for instance).

#### 3.2 Clusters

So now let's show which players are most similar with clustering.

In Python, we use the main Python machine learning package, scikit-learn, to fit a k-means clustering model and get our cluster labels. We perform very similar methods to prepare the data that we used in R, except we use the get_numeric_data and dropna methods to remove non-numeric columns and columns with missing values.

``` python
from sklearn.cluster import KMeans
kmeans_model = KMeans(n_clusters=5, random_state=1)
good_columns = nba._get_numeric_data().dropna(axis=1)
kmeans_model.fit(good_columns)
labels = kmeans_model.labels_
```

In order to cluster properly, we remove any non-numeric columns and columns with missing values (NA, Nan, etc). In R, we do this by applying a function across each column, and removing it if it has any missing values or isn’t numeric. We then use the cluster package to perform k-means and find 5 clusters in our data. We set a random seed using set.seed to be able to reproduce our results.

``` R
library(cluster)
set.seed(1)
isGoodCol <- function(col){
   sum(is.na(col)) == 0 && is.numeric(col) 
}
goodCols <- sapply(nba, isGoodCol)
clusters <- kmeans(nba[,goodCols], centers=5)
labels <- clusters$cluster
```

Now we can plot players by cluster to discover patterns. One way to do this is to first use PCA to make our data 2-dimensional, then plot it, and shade each point according to cluster association.

In Python, we used the PCA class in the scikit-learn library and matplotlib to create the plot!

``` python
from sklearn.decomposition import PCA
pca_2 = PCA(2)
plot_columns = pca_2.fit_transform(good_columns)
plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=labels)
plt.show()
```
In R, the clusplot function was used, which is part of the cluster library. We performed PCA via the pccomp function that is builtin to R.

``` R
nba2d <- prcomp(nba[,goodCols], center=TRUE)
twoColumns <- nba2d$x[,1:2]
clusplot(twoColumns, labels)
```

### 3.3 ggplot & diamonds

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

### 3.4 Scatter Plots

#### 3.4.1 Python ggplot

``` python 
from ggplot.exampledata import diamonds

import seaborn as sn

sn.set_style("white")
sn.lmplot("carat","price",col="cut",data=diamonds, order=2)
sn.plt.show()
```

#### 3.4.2 R ggplot2

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

### 4.1 Linear Regression

So now let’s build a classifier that will predict the number of assists per player from field goals made per player.


Scikit-learn has a linear regression model that we can fit and generate predictions from: 

``` python
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(train[["fg"]], train["ast"])
predictions = lr.predict(test[["fg"]])
```

R relies on the built-in lm and predict functions. predict will behave differently depending on the kind of fitted model that is passed into it – it can be used with a variety of fitted models.

``` R
fit <- lm(ast ~ fg, data=train)
predictions <- predict(fit, test)
```

#### 4.1.1 Summary Stats

If we want to get summary statistics about the fit, such as r-squared value, we’ll need to do a bit more in Python than in R. With R, we can use the built-in summary function to get the needed information. 

``` R
summary(fit)
```
With Python, we need to use the statsmodels package, which enables many statistical methods to be used in Python. We get similar results, although generally it’s a bit harder to do statistical analysis in Python, and some statistical methods that exist in R don’t exist in Python.

``` python
import statsmodels.formula.api as sm
model = sm.ols(formula='ast ~ fga', data=train)
fitted = model.fit()
fitted.summary()
```

### 4.2 Random Forests

Our linear regression worked well in the single variable case, but there may be nonlinearities in the data, so to take care of that, we want to fit a random forest model.

``` R
library(randomForest)
predictorColumns <- c("age", "mp", "fg", "trb", "stl", "blk")
rf <- randomForest(train[predictorColumns], train$ast, ntree=100)
predictions <- predict(rf, test[predictorColumns])
```

``` python
from sklearn.ensemble import RandomForestRegressor
predictor_columns = ["age", "mp", "fg", "trb", "stl", "blk"]
rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=3)
rf.fit(train[predictor_columns], train["ast"])
predictions = rf.predict(test[predictor_columns])
```

The main difference here is that we needed to use the randomForest library in R to use the algorithm, whereas it was built in to scikit-learn in Python. scikit-learn has a unified interface for working with many different machine learning algorithms in Python, and there’s usually only one main implementation of each algorithm in Python. With R, there are many smaller packages containing individual algorithms, often with inconsistent ways to access them. This results in a greater diversity of algorithms (many have several implementations, and many are fresh out of research labs), but with a bit of a usability hit.


#### 4.2.1 Error

``` R 
mean((test["ast"] - predictions)^2)
```

``` python
from sklearn.metrics import mean_squared_error
mean_squared_error(test["ast"], predictions)
```

In Python, the scikit-learn library has a variety of error metrics that we can use. In R, there are likely some smaller libraries that calculate MSE, but doing it manually is pretty easy in either language. There’s a small difference in errors that almost certainly due to parameter tuning, and isn’t a big deal.



### 4.3 Time Series

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


### 4.4 Clustering 

K Means Clustering is an unsupervised learning algorithm that clusts data based on their similarity. In k means clustering, we have to the specify the number of clusters we want the data to be grouped into. The algorithm randomly assigns each observation to a cluster, and finds the centroid of each cluster. Then, the algorithm iterates through two steps:

- Reassign data points to the cluster whose centroid is closest
- Calculate new centroid of each cluster

These two steps are repeated till the within cluster variation cannot be reduced any further. The within cluster variation is calculated as the sum of the euclidean distance between the data points and their respective cluster centroids.

#### 4.3.1 Data Exploration


As we mentioned before, the iris dataset contains information about sepal length, sepal width, petal length, and petal width of flowers of different species. As before we load in the data and take a look at what it looks like:

``` R
library(datasets)
library(ggplot)
head(iris)
```

We can try different visualizations to see if there are any interesting relationships between the four variables. So first we try sepal length vs sepal width: 

``` R
ggplot(iris, aes(Sepal.Length, Sepal.Width, color = Species)) + geom_point()
```

Nothing particularly interesting emerges, so let's try sepal length versus petal length!

``` R
ggplot(iris, aes(Sepal.Length, Petal.Width, color = Species)) + geom_point()
```

Now that's a little better! For the sake of showing more examples, let's try out another visualization:
``` R
ggplot(iris, aes(Petal.Length, Petal.Width, color = Species)) + geom_point()
```

Now those clusters emerge even more!


#### 4.4.2 Clustering

So now that we have seen what the data looks like, let's try to actualy cluster it. Since the initial cluster assignments are random, let us set the seed to ensure reproducibility. 

``` R
set.seed(20)
irisCluster <- kmeans(iris[, 3:4], 3, nstart = 20)
```

Since there are 3 species involved, we have the algorithm group the data into 3 clusters. And since the starting assignments are random, we specify nstart = 20, which means that R will try 20 random starting assignments and select the one with the lowest. So now let's compare the clusters with the species.

``` R
table(irisCluster$cluster, iris$Species)
```

As we can see, the data belonging to the setosa species got grouped into cluster 3, versicolor into cluster 2, and virginica into cluster 1. This means the algorithm wrongly classified two data points belonging to versicolor and six data points belonging to virginica.


### 4.5 Random Forests

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

Both Python and R have their own strengths and weaknesses. As we'll go into soon, Python is a much more dynamic language whereas R has great statistical support. Combining both language's strengths and weaknesses is the ideal scenario - we'll go into what that looks like soon. 

### 5.1 R Statistical Support

R was built as a statistical language, and it shows. `statsmodels` in Python and other packages provide decent coverage for statistical methods, but the R ecosystem is far larger. R has a rich ecosystem of cutting-edge packages and active community. Packages are available at CRAN, BioConductor and Github. You can search through all R packages at Rdocumentation.


### 5.2 Python Non-Statistical Support

Python is a general purpose language that is easy and intuitive. This gives it a relatively flat learning curve, and it increases the speed at which you can write a program.

Furthermore, the Python testing framework is a built-in, low-barrier-to-entry testing framework that encourages good test coverage. This guarantees your code is reusable and dependable.

### 5.3 Visualizations

Visualizations are an important criteria when choosing data analysis software. Although Python has some nice visualization libraries, such as Seaborn, Bokeh and Pygal, there are maybe too many options to choose from. Moreover, compared to R, visualizations are usually more convoluted, and the results are not always so pleasing to the eye.

In R, visualized data can often be understood more efficiently and effectively than the raw numbers alone. R and visualization are a perfect match. Some must-see visualization packages are ggplot2, ggvis, googleVis and rCharts.

### 5.4 Speed

R was developed for statisticians, so its code is not necessarily the most efficient. Although R can be experienced as slow, there are multiple packages to improve R’s performance: pqR, renjin and FastR, Riposte and more.


### 5.5 When do we use each then? 

#### 5.5.1 Python 

Use Python when your data analysis needs to be integrated with web apps or if your statistical code needs to be incorporated into a production database. Being a dynamic programming language, it’s a great tool to implement algorithms for production use. If your data analysis needs integration with a web application or database, Python is probably your best bet. Compared to R, the support for these sorts of application is much better since it's more of a general-purpose language. 

#### 5.5.2 R 

R is mainly for when the data analysis task requires standalone computing. It’s great for exploratory work and for almost any type of data analysis because of the huge number of packages and readily usable tests that often provide you with the necessary tools to get up and running quickly. R can even be part of a big data solution.

When getting started with R, a good first step is to install RStudio. Once this is done, you should continue to have a look at the following packages:

- dplyr, plyr and data.table to easily manipulate packages,
- stringr to manipulate strings,
- zoo to work with regular and irregular time series,
- ggvis, lattice, and ggplot2 to visualize data, and
- caret for machine learning

## 6.0 Final Words

### 6.1 Resources

[Public Datasets](https://github.com/caesar0301/awesome-public-datasets) <br>
[Kaggle](https://www.kaggle.com/) <br>
[The Art of R Programming](https://www.dropbox.com/s/cr7mg2h20yzvbq3/The_Art_Of_R_Programming.pdf?dl=0)<br>
[Python Data Visualization Cookbook](https://www.dropbox.com/s/iybhvjblkjymnw7/Python%20Data%20Visualization%20Cookbook%20-%20Milovanovic%2C%20Igor-signed.pdf?dl=0)

### 6.2 More!

Join us for more workshops! 

[Thursday, December 29th, 6:00pm: Deep Learning Meets NLP: Intro to word2vec](https://www.eventbrite.com/e/december-data-science-series-deep-learning-meets-nlp-intro-to-word2vec-tickets-30494609197?aff=erelpanelorg)


