nba <- read.csv("nba_2013.csv")
head(nba, 5)
trainRowCount <- floor(0.8 * nrow(nba))
set.seed(1)
trainIndex <- sample(1:nrow(nba), trainRowCount)
train <- nba[trainIndex,]
test <- nba[-trainIndex,]

library(GGally)
ggpairs(nba[,c("ast", "fg", "trb")])

library(cluster)
set.seed(1)
isGoodCol <- function(col){
  sum(is.na(col)) == 0 && is.numeric(col) 
}
goodCols <- sapply(nba, isGoodCol)
clusters <- kmeans(nba[,goodCols], centers=5)
labels <- clusters$cluster

nba2d <- prcomp(nba[,goodCols], center=TRUE)
twoColumns <- nba2d$x[,1:2]
clusplot(twoColumns, labels)

library(ggplot2)
library(dplyr)
data(diamonds)
diamonds %>% 
  ggplot(aes(x=carat,y=price)) + 
  geom_point(alpha=0.5) +
  facet_grid(~ cut) + 
  stat_smooth(method = lm, formula = y ~ poly(x,2)) + 
  theme_bw()

library(ggplot2)
library(dplyr)
data(diamonds)
diamonds %>%
  ggplot(aes(x=carat,y=price)) + 
  geom_point(alpha=0.5) + facet_grid(~ cut) +
  stat_smooth(method=lm,formula=y ~ ploy(x,2)) + theme_bw()

fit <- lm(ast ~ fg, data=train)
predictions <- predict(fit, test)

summary(fit)

predictions

library(randomForest)
predictorColumns <- c("age", "mp", "fg", "trb", "stl", "blk")
rf <- randomForest(train[predictorColumns], train$ast, ntree=100)
predictions <- predict(rf, test[predictorColumns])

predictions

mean((test["ast"] - predictions)^2)

sns.set()
sns.set_context('notebook', font_scale=1.5)
cp = sns.color_palette()

library(datasets)
library(ggplot)
head(iris)

ggplot(iris, aes(Sepal.Length, Sepal.Width, color = Species)) + geom_point()

ggplot(iris, aes(Sepal.Length, Petal.Width, color = Species)) + geom_point()


ggplot(iris, aes(Petal.Length, Petal.Width, color = Species)) + geom_point()

set.seed(20)
irisCluster <- kmeans(iris[, 3:4], 3, nstart = 20)

table(irisCluster$cluster, iris$Species)


