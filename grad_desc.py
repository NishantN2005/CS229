import numpy as np
#yhat=theta0+theta1x1
# initilialize params
x=np.random.randn(10,1)
y=5*x+np.random.random(1)

theta0=0.0
theta1=0.0

learning_rate=0.1

#create gradient descent function
#loss=(yhat-y)**2
#yhat=theta0+theta1x1
#loss=(t0+t1x1-y)**2
def grad_desc(theta0,theta1,learning_rate,x,y):
    dldtheta0=0.0
    dldtheta1=0.0
    m=x.shape[0]

    for xi,yi in zip(x,y):
        dldtheta0+=2*(theta0+theta1*xi-yi)
        dldtheta1+=2*(theta0+theta1*xi-yi)*xi

    theta0=theta0-learning_rate*1/m*dldtheta0
    theta1=theta1-learning_rate*1/m*dldtheta1

    return theta0, theta1

#iterate until convergence
for epoch in range(100):
    theta0,theta1=grad_desc(theta0,theta1,learning_rate,x,y)
    yhat=theta0+theta1*x
    loss=np.sum((yhat-y)**2)/x.shape[0]
    print(f'epoch:{epoch+1}, loss:{loss}, theta0:{theta0},theta1{theta1}')