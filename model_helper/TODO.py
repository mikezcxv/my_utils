# A lot
    # https://github.com/josephmisiti/awesome-machine-learning#python-kaggle


# Stacking
    # https://alexanderdyakonov.wordpress.com/2017/03/10/c%D1%82%D0%B5%D0%BA%D0%B8%D0%BD%D0%B3-stacking-%D0%B8-%D0%B1%D0%BB%D0%B5%D0%BD%D0%B4%D0%B8%D0%BD%D0%B3-blending/amp/

# Grid search
    # https://www.kaggle.com/tanitter/grid-search-xgboost-with-scikit-learn
    # https://www.kaggle.com/phunter/xgboost-with-gridsearchcv
    # https://github.com/fmfn/BayesianOptimization/blob/master/examples/visualization.ipynb

# Features selection
    # feature_selection.GenericUnivariateSelect([…])	Univariate feature selector with configurable strategy.
    # feature_selection.SelectPercentile([…])	Select features according to a percentile of the highest scores.
    # feature_selection.SelectKBest([score_func, k])	Select features according to the k highest scores.
    # feature_selection.SelectFpr([score_func, alpha])	Filter: Select the pvalues below alpha based on a FPR test.
    # feature_selection.SelectFdr([score_func, alpha])	Filter: Select the p-values for an estimated false discovery rate
    # feature_selection.SelectFromModel(estimator)	Meta-transformer for selecting features based on importance weights.
    # feature_selection.SelectFwe([score_func, alpha])	Filter: Select the p-values corresponding to Family-wise error rate
    # feature_selection.RFE(estimator[, …])	Feature ranking with recursive feature elimination.
    # feature_selection.RFECV(estimator[, step, …])	Feature ranking with recursive feature elimination and cross-validated selection of the best number of features.
    # feature_selection.VarianceThreshold([threshold])	Feature selector that removes all low-variance features.
    # feature_selection.chi2(X, y)	Compute chi-squared stats between each non-negative feature and class.
    # feature_selection.f_classif(X, y)	Compute the ANOVA F-value for the provided sample.
    # feature_selection.f_regression(X, y[, center])	Univariate linear regression tests.
    # feature_selection.mutual_info_classif(X, y)	Estimate mutual information for a discrete target variable.
    # feature_selection.mutual_info_regression(X, y)	Estimate mutual information for a continuous target variable.

# Cat encoding
    # https://roamanalytics.com/2016/10/28/are-categorical-variables-getting-lost-in-your-random-forests/
    # Target encoding
    # https://www.kaggle.com/aharless/xgboost-cv-lb-284

# Input string handling
    # https://stackoverflow.com/questions/22604564/how-to-create-a-pandas-dataframe-from-a-string

# Visualization
# Seaborn
    # https://seaborn.pydata.org/generated/seaborn.countplot.html
    # http://seaborn.pydata.org/tutorial/categorical.html
    # http://seaborn.pydata.org/tutorial/axis_grids.html
    # https://seaborn.pydata.org/tutorial/distributions.html
    # https://seaborn.pydata.org/generated/seaborn.regplot.html

# Time series prediction
    # https://habrahabr.ru/company/ods/blog/323730/


# Etc
    # https://github.com/uber/pyro



# log_client = MongoClient("mongodb://127.0.0.1:27017")
# logger = RunInfoRegression(log_client, 'us_leasing_exploration', 'runs')
# logger.save('After restart',
#             {'oot': metrics_oot, 'cv': cv_info}, model.get_fscore(),
#            list(df.columns.values), xgb_params, model_description='Draft model'
#            )