# hastings

- Issues identified with the data and how these were addressed
- Data cleansing
* Data cleaning - dodgy characters stripped
* Some features constant - dropped
* Incomplete coverage of features and inconsistent distribution - marked those as recent coverage only
* Latest weather not reliable - did not fix but could have used external data joined on date
* Missing Targets - explored how they were missing and then just dropped them. 
* Outliers in targets - explored using logged features instead but ultimately found capped_incurred has best r2 without further transformations or different model spec.

- Model specification and justification for selecting this model specification

* Approached as a continous prediction problem. Considered combining with high value binary flag and using different models to predict given whether its a high value predicted case or not
* Test train split done on date rather than random sampling to better assess chance to generalise to the future
* Produced two test train splits - one for full data in train up to 2014 to give 2014-2015Q1 to test on. This did mean not using a lot of the most recent best quality data so wasnt ideal but I wanted a full year of test data in case of seasonality being missed. The 2nd recent set was 2012-2014q3 leaving 2 most recent Qs to test on. This had better results and better coverage of some useful features
* Considered predictive accuracy above robust coefficient estimation. So used some boosting and bagging (rf and gbm) approaches as well as multiple regression. 
* GBM handle non-linear relationships, skew in target and predictor distributions, mixed data types, high levels of colinearity
* Extensive use of mean target encoding for categorical and multivariable counts although also compared predictive value of counts and dummy variables as appropriate
* All features were run as predictors for a simple linear model vs target in EDA and I prioritised those that had a visible relationship to target or a statistically significant coefficient.
* Ensemble trees were used to find hidden interactions efficiently. Hyper-parameter tuning was attempted but had little improvement over the feature selection. 
* Manual interaction terms were created based on terms that plausibly might interact. I opted for meant
* Applied stepwise regression with manual coefficient colinearity filtering to mine features that were predictive but not overly correlated amongst themselves. Was not as effective as allowing gradient boosting access to more columns
* Used RF feature importance to select variables for linear regression but again wasnt as effective as experimenting manually

- Assessment of your model's accuracy and model diagnostics
* Found a decent level of signal but without a model that has robust coefficient interpretation
* Earlier models had awful residuals but later they looked healthier where error could only be in one direction at the lowest points
* Tree based models were overly focussed around the mid range of predictions. Multiple regression spread more nicely

- Suggestions of how your model could be improved
* Bayesian causal approach with DAG and interpretable coefficient
* Ensembling multiple models
* 2 stage prediction, first a binary high value flag then incurred within that subset
* More exploration of scaling and transforming features to be normally distributed
* Try dimensionality reduction of large feature set
* Add external data e.g. weather


- Practical challenges for implementing your model
* Lots of grouped count variables - tried adding mean target encoding to summarise
* Large number of colinear features, modelling techniques handle this and creating handy ColumnMetaOrganiser class allowed me to quickly audition different combinations of conceptually grouped columns and encoding methods