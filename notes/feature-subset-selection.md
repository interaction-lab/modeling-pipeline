# Feature Subset Selection

## Recommended Reading

 - https://machinelearningmastery.com/an-introduction-to-feature-selection/
 - https://machinelearningmastery.com/rfe-feature-selection-in-python/
 - https://machinelearningmastery.com/feature-selection-machine-learning-python/
 - https://machinelearningmastery.com/feature-selection-with-real-and-categorical-data/
 
 Papers:
 - [Unsupervised feature selection for sensor time-series in pervasive computing applications](http://pages.di.unipi.it/bacciu/wp-content/uploads/sites/12/2016/04/nca2015.pdf)
 - [Feature subset selection and feature ranking for multivariate time series](https://ieeexplore.ieee.org/document/1490526)




## Feature Selection for Time Series Modeling

When trying to do time series classification with Multivariate Time Series (MTS), it is often better to use an unsupervised univariate selection method based on cross correlation between features rather than between features and the label, which may not be well correlated with the features directly.

### Features from Corr.

Here we iterate through features and provide a list of features which are not correlated with each other. We utilize df.corr methods for Pearson, Spearman, and Kendal correlation coefficients.

## Wrapper vs Filter Methods

Wrapper Methods
 - Recursive Feature Elimination - Iteratively trains an SVM which ranks feature importance
 - Maximum-Relevance Minimum-Redundancy (MRMR) - Doesn't work well with time series

Filter Methods
 - CleVer - exploits the properties of the principal components common to all the time-series to provide a ranking of the more informative features
 - Incremental Cross-correlation Filter (ICF)