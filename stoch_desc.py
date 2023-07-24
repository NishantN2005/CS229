import numpy as np
import pandas as pd


data=pd.read_excel('Real estate valuation data set.xlsx')
data_np=np.array(data)[:,1:]

data_train=data_np[:int(data_np.shape[0]/2),:]
data_test=data_np[int(data_np.shape[0]/2):,:]

#yhat=t0+t1x1+t2x2+t3x3+t4x4+t5x5+t6x6
#initialize parameters
x1_data=data_train[:,:1]
x2_data=data_train[:,1:2]
x3_data=data_train[:,2:3]
x4_data=data_train[:,3:4]
x5_data=data_train[:,4:5]
x6_data=data_train[:,5:6]
y_data=data_train[:,6:]

learning_rate=0.0000016

t0=0.0
t1=0.0
t2=0.0
t3=0.0
t4=0.0
t5=0.0
t6=0.0

#create stochastic gradient descent function
def stoc_desc(x1_data,x2_data,x3_data,x4_data,x5_data,x6_data,y_data,learning_rate,t0,t1,t2,t3,t4,t5,t6):
    dldt0=0.0
    dldt1=0.0
    dldt2=0.0
    dldt3=0.0
    dldt4=0.0
    dldt5=0.0
    dldt6=0.0
    
    m=x1_data.shape[0]
    for i in range(1,7):
        ri=np.random.randint(i)
        delta_j_theta=(t0+t1*x1_data[ri:ri+1]+t2*x2_data[ri:ri+1]+t3*x3_data[ri:ri+1]+t4*x4_data[ri:ri+1]+t5*x5_data[ri:ri+1]+t6*x6_data[ri:ri+1])
        y_ri=y_data[ri:ri+1]
        dldt0=2*(delta_j_theta-y_data[ri:ri+1])
        dldt1=2*x1_data[ri:ri+1]*(delta_j_theta-y_ri)
        dldt2=2*x2_data[ri:ri+1]*(delta_j_theta-y_ri)
        dldt3=2*x3_data[ri:ri+1]*(delta_j_theta-y_ri)
        dldt4=2*x4_data[ri:ri+1]*(delta_j_theta-y_ri)
        dldt5=2*x5_data[ri:ri+1]*(delta_j_theta-y_ri)
        dldt6=2*x6_data[ri:ri+1]*(delta_j_theta-y_ri)
    
    t0=t0-learning_rate*1/m*dldt0
    t1=t1-learning_rate*1/m*dldt1
    t2=t2-learning_rate*1/m*dldt2
    t3=t3-learning_rate*1/m*dldt3
    t4=t4-learning_rate*1/m*dldt4
    t5=t5-learning_rate*1/m*dldt5
    t6=t6-learning_rate*1/m*dldt6

    return t0,t1,t2,t3,t4,t5,t6


for epoch in range(4000):
    t0,t1,t2,t3,t4,t5,t6=stoc_desc(x1_data,x2_data,x3_data,x4_data,x5_data,x6_data,y_data,learning_rate,t0,t1,t2,t3,t4,t5,t6)

    yhat=t0+t1*x1_data+t2*x2_data+t3*x3_data+t4*x4_data+t5*x5_data+t6*x6_data

    loss=np.sum((yhat-y_data)**2)/x1_data.shape[0]



y_actual=data_test[:,6:]
x_actual_1=data_test[:,:1]
x_actual_2=data_test[:,1:2]
x_actual_3=data_test[:,2:3]
x_actual_4=data_test[:,3:4]
x_actual_5=data_test[:,4:5]
x_actual_6=data_test[:,5:6]

y_pred=t0+t1*x1_data+t2*x2_data+t3*x3_data+t4*x4_data+t5*x5_data+t6*x6_data

for y_act,y_p in zip(y_data,y_pred):
    print(f'Actual Value: {y_act}, Predicted value: {y_p}, Difference: {y_act-y_p}')