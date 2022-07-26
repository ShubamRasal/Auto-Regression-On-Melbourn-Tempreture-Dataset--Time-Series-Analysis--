# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 09:53:24 2022

@author: Shubham
"""
#AUTOREGRESSION MODELS FOR TSA WITH PYTHON:
#-------------------------------------------

#AR is a time series model that uses observations from previous time steps as
#input to a regression equation to predict the value at the next time step.

#AUTOREGRESSION:
#----------------
#A regression model, such as linear regression, models an output value based on
#a linear combination of input values.

#yhat = b0 + b1*X1
#------------------
#Here yhat is the prediction.
#b0 and b1 are coefficients found by optimizing the model on training data
#X is an input value.

#This technique can be used on time series where input variables are taken as
#observation at previous time steps, called lag variables.

#we can predict the value for the next time step (t+1) given the observations at
#the last two 
#time steps (t-1 and t-2).
#As a regresssion model, this would look as follows:

# X(t+1) = b0 + b1*X(t-1) + b2*X(t-2)
#--------------------------------------
#Because the regression model uses data from the same input variable at previou
#time steps, it is referred to as an autoregression (regression of self).

#Autocorrelation:

#An autoregression model makes an assumption that the observations at previous
#time steps are useful to predict the value at the next time step.

#This relationship between variables is called correlation.

#If both variables change in the same direction(e.g. go up together or down
#together), this is called a positive correlation.

#If the variables move in opposite direction as values change (e.g. one goes up
#and one goes down), then this is called negative correlation.

#We can use statistical measures to calculate the correlation between the output
#variable and values at previous time
#steps at various different lags.
#The stronger the correlation beteween the outpu variable and specific lagged
#variable, the more weight that autoregression model can put on that variable
#when modeling.

#Because the correlation is calculated between the variable and itself at 
#previous time steps, it is called an autocorrelation.
#It is called serial correlation because of the sequenced structure of time
#series data.

#The correlation statistics can also help to choose which lag varibles will be
#useful in a model and which will not.

#If all lag variables show low or no correlation with the output variable, then
#it suggests that the time series problem may not be predictable.

#MINIMUM DAILY TEMPERATURES DATASET:
#------------------------------------

import pandas as pd
import matplotlib as pyplot

series = pd.read_csv(r"D:\Data Science and Artificial Inteligence\Semister- ll\Machine Learning Sujit Deokar Sir\AutoRegression\melbourn_temp_dataset.csv",header=0,index_col=0)

print(series.head())

series.plot()
pyplot.show()

#Quick Check for Autocorrelation:
#---------------------------------

#There is q quick, visual check that we can do to see if there is an autocorrelation
#in our time series dataset.

#We can plot the obsercation at the previous time step (t-1) with the observation
#at the next time step (t+1) as a scatter plot.

#This could be done manually by first creating a lag version of the time series
#dataset and using a built- in scatter plot function in the pandas library.

#Pandas provides a built-in plot to do exactly this, called the lag_plot() function.

from pandas.plotting import lag_plot

lag_plot(series)
pyplot.show()

#We can see a large ball of observations along a diagonal line of the plot. If
#clearly shows a relationship or some correlation.

#Another quick check that we can do is to directly calculate the correlation
#between the observation and the lag variable.

#We can use a statistical test like the Pearson correlation coefficient.
#This produces a number to summarize how two variables are between -1(negatively
#correlated)and +1(positively correlated) with small values close to zero indicating
#low correlation and high values above 0.5 or below -05 showing high correlation.

from pandas import concat
from pandas import DataFrame
values = DataFrame(series.values)
df = concat([values.shift(1),values],axis=1)
df.columns=['t-1','t+1']
result =df.corr()
print(result)

#It shows a strong positive correlation (0.77) between the observation and the
#lag=1 value.

#Autocorrelation Plots:
#-------------------------

#We can plot the correlation coefficient for each lag variable.

#This can very quickly give an idea of which lag variables may be good candidates 
#for use in a predictive model and how the relationship between the observation 
#and its historic values changes over time.

#The plot provides the lag number along the x-axis and the correlation coefficient 
#value between -1 and 1 on the y-axis.The plot also includes solid and dashed lines 
#that indicate the 95% and 99% confidence interval for the correlation values.
#Correlation values above these lines are more significant than those below the line, 
#providing a threshold or cutoff for selecting more relevant lag values.

from pandas.plotting import autocorrelation_plot

autocorrelation_plot(series)
pyplot.show()

#The statsmodels library also provides a version of the plot in the plot_act() function as a line plot.

from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf

plot_acf(series, lags=31)
pyplot.show()

#In this example, we limit the lag variables evaluated to 31 for readablility.

#Persistence Model:
#-------------------

#Let's say that we want to develop a model to predict the last 7 days of minimum 
#temperatures in the dataset given all prior observations.

#The simplest model that we could use to make predictions would be to presist the 
#last observation. We can call this a persistence model and it provides a baseline
#of performance for the problem that we can use for comparision with an autoregression model.

#We can develop a test harness for the problem by splitting the observations into
#training and test sets, with only the last 7 observations in the dataset assigned
#to the test set as "unseen" data that we wish to predict.


from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
#Create lagged dataset

values = DataFrame(series.values)
dataframe = concat([values.shift(1), values],axis=1)
dataframe.columns = ['t-1','t+1']

#Split into train and test sets
X = dataframe.values
train, test = X[1:len(X)-7], X[len(X)-7:]
train_X,train_y = train[:,0],train[:,1]
test_X,test_y = test[:,0],test[:,1]


#Persistence model:
def model_persistence(x):
    return x

#walk-forward validation:

predictions = list()
for x in test_X:
    yhat = model_persistence(x)
    predictions.append(yhat)
test_score = mean_squared_error(test_y, predictions)
print('Test MSE: %.3f' % test_score)

#plot predictions vs expected
pyplot.plot(test_y)
pyplot.plot(predictions, color='red')
pyplot.show()

#Running the example prints the mean squared error (MSE)
#The value provides a baseline performance for the problem.
#The expected values for the next 7 days are plotted (blue) compared to the 
#predictions from the model(red).

#Autoregression Model:

#An autoregression model is a linear regression model that uses lagged variables 
#as input variables.We could calculate the linear regression model manually using
#the Linear Regression class in scikit-learn and manually specify the lag input
#variables to use. Alternately, the statsmodels library provides an autoregression
#model where you must specify an appropriate lag value and trains a linear regression model.
#It is provided in the AutoReg class.

#Create and evaluate a static autoregressive model.

from matplotlib import pyplot
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
from math import sqrt

#split dataset
X = series.values
train, test = X[1:len(X)-7], X[len(X)-7:]

#Train autoregression
model = AutoReg(train, lags=29)
model_fit = model.fit()
print('Coefficients: %s' % model_fit.params)

#make predictions:
predictions = model_fit.predict(start=len(train),end=len(train)+len(test)-1, dynamic=False)
for i in range(len(predictions)):
    print('predicted=%f, expected=%f' % (predictions[i], test[i]))
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)

#plot results:
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()

#A plot of the expected (blue) vs the predicted values (red) is made.
#The forecast does look pretty good (about 1 degree Celsius out each day), with 
#big devation on day 5.The statsmodels API does not  make it easy to update the
#model as new observatoins became available.

#One way would be to retrain the AutoReg model each day as new observations became 
#available, and that may be a valid approach, if not computationally expensive.

#An alternative would be to use the learned coeffcients and manually make predictions.
#This requires that the history of 29 prior observations be kept and that the 
#coefficients be retrieved from the model and used in the regression equation to 
#come up with new forecasts.

#The coefficients are provided in an arrya with the intercept term followed by the 
#coefficients for lag variable starting at t-1 to t-n. We simply need to use them 
#in the right order on the history of observations, as follows:

#yhat = b0 + b1*X1 + b2*X2 ... bn*Xn

from matplotlib import pyplot
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
from math import sqrt

#split dataset
X = series.values
train, test = X[1:len(X)-7], X[len(X)-7:]

#train autoregression
window = 29
model = AutoReg(train, lags=29)
model_fit = model.fit()
coef = model_fit.params

#walk forward over time steps in test
history = train[len(train)-window:]
history = [history[i] for i in range(len(history))]
predictions = list()
for t in range(len(test)):
    length = len(history)
    lag = [history[i] for i in range(length-window, length)]
    yhat = coef[0]
    for d in range(window):
        yhat += coef[d+1] * lag[window-d-1]
    obs = test[t]
    predictions.append(yhat)
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)

#plot
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()

#Specifically, you learned:

#About autocorrelation and autoregression and how they can be used to better 
#understand time series data.How to explore the autocorrelation in a time series 
#using plots and statistical tests.How to train an autoregression model in Python 
#and use it to make short-term and rolling forecasts.



































































