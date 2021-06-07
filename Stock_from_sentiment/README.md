# Business problem

I have always been fascinated by the stock market. I also traded for major part of 2020. I booked profits and suffered losses too. Along with a friend at work, we always tried to predict the movement of BANK NIFTY and a few other tickers. He went to do his MBA and I changed my plans from MBA to building a career in analytics. This project is my first attempt at trying to predict the stock market movement on a daily basis based on sentiment analysis of news. 

# Methodology

This is a long term project. Currently I have tested the waters with a comparison of sentiment based analysis between rule based and lexicon based models. 
The dataset used is Apple's news. In the rule based analysis I have trained a machine learning model on the data. 
In the lexicon based analysis I have gauged each news article's sentiment polarity and objectivity. Then I have run it through a sigmoid transformation to get a probability value.
When the probability is above a particular threshold, it predicts the positive class.

# Wordcloud vizualized

![image](https://i.ibb.co/1bcv5DM/aples.png)

# Model testing

## Rule based model performance

Model| Accuracy|	Precision|	Recall|	F1 score|
|---|---|---|---|---|
GaussianNB|	0.883871|	0.888646|	0.992683|	0.937788|
Random Forest|	0.877419|	0.881210|	0.995122|	0.934708|
SVM|	0.873118|	0.887417|	0.980488|	0.931634|
GBM|	0.875269|	0.909302|	0.953659|	0.930952|
XGB|	0.868817|	0.888641|	0.973171|	0.928987|
DecisionTree|	0.851613|	0.891954|	0.946341|	0.918343|
Adaboost|	0.851613|	0.895592|	0.941463|	0.917955|
BernoulliNB|	0.855914|	0.927681|	0.907317|	0.917386|
Logistic|	0.843011|	0.896471|	0.929268|	0.912575|
KNN|	0.763441|	0.926136|	0.795122|	0.855643|

We'll use GBM based on the above table as it scores >=90 in precision recall and F1 score. Also, a very high recall and stagnant precision score indicates that those models predict the negative class for almost all the observations in the test set.

`Confusion matrix for GBM`

||True positive|True negative|
|-|--|--|
|Predicted positive|16|  39|
|Predicted negative|19| 391|
       
## Lexicon based model performance

After each sentiment is run through the sigmoid transformation:

![prob](https://i.ibb.co/SrX5hfv/prob.png)

I have used a threshold of 0.5 for the positive class. But very small changes drastically change the classifications.

The result of lexicon based sentiment analysis:

F1 score:  0.905885156063855
Precision:  0.8837749883774988
Accuracy:  0.8299612569952648

Confusion matrix:

||True positive|True negative|
|-|--|--|
|Predicted positive|27|  250|
|Predicted negative|145| 1901|

As we can see that the performance is very comparable. Lexicon based sentiment analysis will be very helpful in the cases of cold start and it is also easier to use. However it may require regular tweaking of the probability threshold to keep making money in the market.
