# Customer Subscription Propensity Modeling Using Random Forest

## Overview

Direct marketing campaigns are widely used by banks to promote financial products such as term deposits. However, contacting every customer is costly and often results in low conversion rates.

This project develops a machine learning model to predict whether a customer is likely to subscribe to a term deposit based on demographic, financial, and campaign-related information. By identifying high-probability customers, banks can improve campaign efficiency, reduce acquisition costs, and increase conversion rates.

---

## Business Problem

Banks invest significant resources in telemarketing campaigns. The challenge is determining which customers are most likely to respond positively before allocating marketing resources.

### Objective

Build a predictive model that estimates the probability of a customer subscribing to a term deposit and ranks customers according to their likelihood of conversion.

### Business Value

* Improve marketing campaign ROI
* Reduce unnecessary customer outreach
* Increase subscription conversion rates
* Enable data-driven customer targeting
* Optimize resource allocation for sales teams

---

## Dataset

The dataset contains customer demographics, financial attributes, and historical marketing campaign information.

### Features

#### Customer Attributes

* Age
* Job Type
* Marital Status
* Education Level

#### Financial Attributes

* Account Balance
* Housing Loan Status
* Personal Loan Status
* Credit Default History

#### Campaign Attributes

* Contact Method
* Month of Contact
* Call Duration
* Number of Campaign Contacts
* Previous Campaign Outcome
* Days Since Previous Contact

### Target Variable

**Subscription Status (y)**

* 1 = Customer subscribed to a term deposit
* 0 = Customer did not subscribe

---

## Machine Learning Approach

### Data Preprocessing

* Missing value assessment
* Binary feature encoding
* One-hot encoding of categorical variables
* Train-test split with stratification
* Class imbalance handling

### Model Development

A Random Forest Classifier was selected due to its ability to:

* Capture non-linear relationships
* Handle mixed feature types
* Reduce overfitting through ensemble learning
* Provide robust classification performance
* Generate feature importance insights

### Hyperparameter Optimization

RandomizedSearchCV was used with 5-fold cross-validation to identify optimal model parameters.

Optimization Metric:

* ROC-AUC Score

---

## Model Performance

### Key Results

| Metric       | Score     |
| ------------ | --------- |
| ROC-AUC      | **0.923** |
| KS Statistic | **0.714** |

### Why ROC-AUC?

ROC-AUC measures the model's ability to distinguish between customers who subscribe and those who do not across different classification thresholds.

### Why KS Statistic?

The Kolmogorov-Smirnov (KS) Statistic is widely used in banking, credit risk, and customer analytics to evaluate a model's discriminatory power.

A KS score of **0.714** indicates strong separation between subscribers and non-subscribers, demonstrating the model's effectiveness in ranking customers by conversion likelihood.

---

## Project Workflow

```text
Data Collection
        ↓
Data Cleaning & Encoding
        ↓
Exploratory Data Analysis
        ↓
Train-Test Split
        ↓
Random Forest Training
        ↓
Hyperparameter Optimization
        ↓
Probability Prediction
        ↓
ROC-AUC Evaluation
        ↓
KS Score Validation
        ↓
Customer Ranking
```

---

## Technical Stack

### Programming Language

* Python

### Data Analysis

* Pandas
* NumPy

### Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-Learn

### Model Evaluation

* ROC-AUC
* KS Statistic
* Cross Validation
* Confusion Matrix

---

## Key Learnings

* Building classification models for business decision-making
* Handling imbalanced datasets
* Hyperparameter tuning using RandomizedSearchCV
* Evaluating predictive models using ROC-AUC
* Applying KS Statistic for customer propensity modeling
* Translating machine learning results into business value

---

## Future Enhancements

* Feature importance analysis
* SHAP-based model explainability
* XGBoost and LightGBM benchmarking
* Customer segmentation and propensity scoring
* Model deployment using Flask/FastAPI
* Real-time scoring pipeline

---

## Author

**Narmathaa Palanisamy**

Biotechnology professional transitioning into Data Science with experience in experimental research, data analysis, machine learning, and predictive modeling. Interested in applying data-driven approaches to solve problems in healthcare, life sciences, and business analytics.




