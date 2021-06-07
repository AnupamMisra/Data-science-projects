# Business problem:

Companies usually have a greater focus on customer acquisition than customer. However, it can cost anywhere between five to twenty five times more to attract a new customer than retain an existing one. Increasing customer retention rates by 5% can increase profits by 25%, according to a [research](https://hbr.org/2014/10/the-value-of-keeping-the-right-customers) done by Bain & Company.  

Churn is a metric that measures the no. of customers who stop doing business with a company. Through this metric, most businesses would try to understand the reason behind churn numbers and tackle those factors with reactive action plans.

But what if you could identify a customer who is likely to churn and take appropriate steps to prevent it from happening? The reasons that lead customers to the cancellation decision can be numerous, ranging from poor service quality to new competitors entering the market. Usually, there is no single reason, but a combination of factors that result to customer churn.

Although the customers have churned, their data is still available. Through machine learning we can sift through this valuable data to discover patterns and understand the combination of different factors which lead to customer churn.

Our goal in this project is to identify behavior among customers who are likely to churn. Subsequent to that we need to train a machine learning model to identify these signals from a customer before they churn. Once deployed, our model will identify customers who might churn and alert us to take necessary steps to prevent their churn.


# Data exploration

## ![Personal factors](https://i.ibb.co/5LqNr02/personal-factors.png)

`Observations:`

* Gender has no influence on churn.
* Single people are more likely to churn.
* People without dependents are more likely to churn.

![Contract](https://i.ibb.co/TkpT2bZ/contract.png)

`Observations:`

* Maximum people who churn are on a monthly contract and mostly bill in a paperless manner. Maybe these are tech savvy people who switch to a different carrier as soon as they find a better deal.

![Payment](https://i.ibb.co/r3509PR/payment.png)

`Observations:`

* Maximum people who pay electronically, churn. This supports our hypothesis that tech savvy people churn more often.

![Charges](https://i.ibb.co/80yXBzg/charges.png)

`Observations:`

* Most of the people who churn have low total charges with the carrier.
* Some people who churn are customers who have high monthly and total charges. Maybe these are corporate customers who are churn when they are offered a more competitive offer.

![Duration](https://i.ibb.co/V9fyTpv/duration.png)

`Observations:`

* Maximum customers churn during the early period of their subscription.

![img](https://i.ibb.co/y035fv3/img.png)

`Observations:`

* Customers with single phone service and no internet service churn the most. Maybe these are people who are not very well off.
* Among customers with internet service, they choose the faster Fibre optic without any protection/security or backup and churn the most.
* Among these customers who have churned, most have never contacted the tech support.

These customers are probably young tech savvy thrifty customers who change the subscription as soon as they spot a better offer.

![img2](https://i.ibb.co/wYgTP67/img2.png)

`Observations:`

* Maximum people who churn do not stream movies or TV, i.e. they are not dependent on the subscription for streamed media consumption.

# Data preparation

Data is split into train and test sets. The overrepresented class(0) is undersampled using `RandomUnderSampler`.

# Model training and evaluation

Five models were trained: `SVM` with polynomial kernel,`Random Forest`, `Logistic Regression`, `Gaussian Naive Bayes` and `KNN`. The model was picked at runtime after the input was given for the cost of churn and the cost to prevent churn. The comparison was also drawn with not using a model and using a random sample of 50% customers. The models were evaluated using F1 score and AUC_ROC.

# Model output (sample):

Model|	Revenue saved|	Predicted(True positive)(%)|	Missed(False negative)(%)|	F1 score|	ROC_AUC|
-----|---------------|--------------------------|-----------------------|---------|--------------|
Random Forest|	251000|	83.78|	16.22| 0.607487|	0.773085|
Logistic regression|	206500|	82.60|	17.40|	0.597015|	0.763914|
Naive Bayes|	183500|	87.32|	12.68|	0.558491|	0.737980|
K Nearest Neighbors|	169000|	85.25|	14.75|	0.565005|	0.741674|
Support Vector Machine|	-164000|	94.10|	5.90|	0.441522|	0.602744|

>Assumed cost of losing a customer:5000

>Assumed cost of effort to prevent churn:1500 

>Lost revenue if we do not prevent churn = Rs.93,45,000

Percentage of customers predicted by 'Random Forest' who were going to churn: 83.78%

Percentage of customers missed who were going to churn: 16.22%

Revenue saved by preventing churn with our model as compared to no model = Rs. 2,51,000


Total expenditure for preventing churn on random 50.0% of customers:52,83,000

Extra cost to prevent churn within random 50.0% of the customers = Rs.6,22,833

>Our 'Random Forest' model saves us Rs.8,73,833 on an average compared to a random selection of 50% customers
