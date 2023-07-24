# CS229

## grad_desc.py
### This is example code of a gradient descent function. It creates one feature of random numbers and based on those numbers creates an output by multiplying the x value by 5 and adding noise. The gradient descent function has two weights because the equation is y(x)=theta0+theta1*x1. Theta0 should be the calculated bias and theta1 should be a number close to or at 5. 

## real_estate.py
### This is a real world example of a gradient descent function that takes in 6 features (listed below). Using University of California, Irvine's data, the function predicts the house price of unit area (10000 New Taiwan Dollar/Ping, where Ping is a local unit, 1 Ping = 3.3 meters squared). 
### The inputs are as follows
### X1=the transaction date (for example, 2013.250=2013 March, 2013.500=2013 June, etc.)
### X2=the house age (unit: year)
### X3=the distance to the nearest MRT station (unit: meter)
### X4=the number of convenience stores in the living circle on foot (integer)
### X5=the geographic coordinate, latitude. (unit: degree)
### X6=the geographic coordinate, longitude. (unit: degree)

## stoch_descent.py
### Similiar to the real_estate.py file this function takes 6 features from University of California, Irvine's data and predicts the house price of unit are (10000 New Taiwan Dollar/Ping, where Ping is a local unit, 1 Ping = 3.3 meters squared)
### The main difference between stochastic gradient descent and batch gradient descent is that it takes less time to run because you are not iterating through the entire dataset for each descent. This results in a faster runtime but also a slightly less accurate model.