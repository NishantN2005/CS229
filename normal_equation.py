import numpy as np
import pandas as pd

data=pd.read_excel('Real estate valuation data set.xlsx')
data_np=np.array(data)[:,1:]

data_train=data_np[:int(data_np.shape[0]/2),:]
data_test=data_np[int(data_np.shape[0]/2):,:]



X=data_np[:,:-1]
Y=data_np[:,-1:]
#adding theta0 which is set to 1 by default
ones=np.ones((X.shape[0], 1))

X=np.append(ones,X,axis=1)
#normal equation: theta=(Xtranspose*X)^-1 * Xtranspose*y
W=np.dot(np.linalg.pinv(np.dot(X.T,X)), np.dot(X.T, Y))

y_actual=data_test[:,6:]
x_actual_1=data_test[:,:1]
x_actual_2=data_test[:,1:2]
x_actual_3=data_test[:,2:3]
x_actual_4=data_test[:,3:4]
x_actual_5=data_test[:,4:5]
x_actual_6=data_test[:,5:6]

y_pred=W[0]+W[1]*x_actual_1+W[2]*x_actual_2+W[3]*x_actual_3+W[4]*x_actual_4+W[5]*x_actual_5+W[6]*x_actual_6

sum=0
for y_act,y_p in zip(y_actual,y_pred):
    sum+=(y_p-y_act)**2
print((1/data_test.shape[0])*sum)
    #print(f'Actual Value: {y_act}, Predicted value: {y_p}, Difference: {y_act-y_p}')

