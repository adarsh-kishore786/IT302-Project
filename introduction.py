import dalex as dx
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

data = dx.datasets.load_apartments()

# one-hot encoding
data = pd.get_dummies(data)
X = data.drop(columns='m2_price')
y = data.m2_price

# create random forest regressor using sklearn
regr = RandomForestRegressor(max_depth=2, random_state=0)
regr.fit(X, y)

# explainer with dalex package
exp = dx.Explainer(regr, X, y)

mp = exp.model_parts(loss_function='rmse')

# data frame of results
mp

mp.plot()

# Here we explore PDP

# PDP profile for surface and construction.year variable
# rf_mprofile = exp.model_profile(variables=["surface", "construction_year"], type="partial")
#
# # plot the PDP
# # rf_mprofile.plot()
#
# #  create a Linear Regression Model
# from sklearn.linear_model import LinearRegression
# lm = LinearRegression()
# lm.fit(X, y)
#
# # create an explainer for the linear regression model
# exp_lm = dx.Explainer(lm, X, y)
#
# # PDP profile for surface and construction_year model
# lm_mprofile = exp_lm.model_profile(variables=["surface", "construction_year"], type="partial")
#
# # comparision for random forest and linear regression
# rf_mprofile.plot(lm_mprofile)
