# Business problem:

Companies usually have a greater focus on customer acquisition and keep retention as a secondary priority. However, it can cost five times more to attract a new customer than it does to retain an existing one. Increasing customer retention rates by 5% can increase profits by 25% to 95%, according to research done by Bain & Company.

Churn is a metric that shows customers who stop doing business with a company or a particular service, also known as customer attrition. By following this metric, what most businesses could do was try to understand the reason behind churn numbers and tackle those factors, with reactive action plans.

But what if you could know in advance that a specific customer is likely to leave your business, and have a chance to take proper actions in time to prevent it from happening? The reasons that lead customers to the cancellation decision can be numerous, coming from poor service quality, delay on customer support, prices, new competitors entering the market, and so on. Usually, there is no single reason, but a combination of events that somehow culminated in customer displeasure.

If your company were not capable to identify these signals and take actions prior to the cancel button click, there is no turning back, your customer is already gone. But you still have something valuable: the data. Your customer left very good clues about where you left to be desired. It can be a valuable source for meaningful insights and to train customer churn models. Learn from the past, and have strategic information at hand to improve future experiences, it’s all about machine learning.

Our goal in this project is to understand churn behaviour among customers. Subsequent to that we need to train a machine learning model to identify these signals from a customer before they churn. Once deployed, our model will identify customers who might churn and alert us to take necessary steps to prevent their churn.

# Data exploration

## ![Personal factors](https://i.ibb.co/5LqNr02/personal-factors.png)

`Observations:`

* Gender has no influence on churn
* Single people are more likely to churn
* People without dependents are more likely to churn

![Contract](https://i.ibb.co/TkpT2bZ/contract.png)

`Observations:`

* Maximum people who churn are on a monthly contract and mostly bill in a paperless manner. Maybe these are tech savvy people who switch to a different carrier as soon as they find a better deal

![Payment](https://i.ibb.co/r3509PR/payment.png)

`Observations:`

* Maximum people who pay electronically, churn. This supports our hypothesis that tech savvy people churn more often.

![Charges](https://i.ibb.co/80yXBzg/charges.png)

`Observations:`

* Most of the people who churn have low total charges with the carrier
* Some people who churn are customers who have high monthly and total charges. These are valuable customers whom we would want to retain

![Duration](https://i.ibb.co/V9fyTpv/duration.png)

`Observations:`

* Maximum customers who churn do so during the early period of their subscription. We have to try to reduce early discontinuations of our service

![img](https://i.ibb.co/y035fv3/img.png)

`Observations:`

* Customers with single phone service and no internet service churn the most
* Among customers with internet service, they choose the faster Fibre optic without any protection/security or backup and churn the most
* Among these customers who have churned, most have never contacted the tech support

These customers are young tech savvy thrifty customers who supposedly change the subscription as soon as they spot a better offer.

![img2](https://i.ibb.co/wYgTP67/img2.png)

`Observations:`

* Maximum people who churn do not stream movies or TV, i.e. they are not dependent on the subscription for streamed media consumption

# Data preparation

Data is split into train and test sets. The overrepresented class(0) is undersampled using `RandomUnderSampler`.

# Model training and evaluation

Four models were trained: `SVM` with polynomial kernel,`Random Forest`, `Logistic Regression`, `Gaussian Naive Bayes` and `KNN`. The model was selected at runtime based on the input give for cost of churn and cost to prevent churn. The comparison was also drawn with not using a model and using a random sample of 50% customers. The models were evaluated using F1 score and AUC_ROC.

# Model output (sample):

Model|	Revenue saved|	Predicted(True positive)(%)|	Missed(False negative)(%)|	F1 score|	ROC_AUC	Model|
-----|---------------|--------------------------|-----------------------|---------|--------------|
Random Forest|	277000|	85.25|	14.75|	0.607781|	0.775319|	(DecisionTreeClassifier(max_depth=10, max_feat...
Logistic regression|	204500|	82.89|	17.11|	0.594080|	0.762118|	LogisticRegression(C=0.01, solver='liblinear')
K Nearest Neighbors|	198500|	87.32|	12.68|	0.563810|	0.742653|	KNeighborsClassifier(n_neighbors=47)
Naive Bayes|	196500|	88.50| 11.50|	0.556586|	0.737338|	GaussianNB()
Support Vector Machine|	-168500|	94.10|	5.90|	0.440608|	0.601343|	SVC(C=0.001, kernel='poly')

>Assumed cost of losing a customer:5000

>Assumed cost of effort to prevent churn:1500 

>Lost revenue if we do not prevent churn = Rs.93,45,000

Percentage of customers predicted by 'Random Forest' who were going to churn: 85.25%

Percentage of customers missed who were going to churn: 14.75%

Revenue saved by preventing churn with our model as compared to no model = Rs. 2,77,000


Total expenditure for preventing churn on random 50.0% of customers:52,83,000

Extra cost to prevent churn within random 50.0% of the customers = Rs.6,04,667

>Our 'Random Forest' model saves us Rs.8,81,667 on an average compared to a random selection of 50% customers
