import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 


# Calculate root mean squared error
def msee(actual, predicted):
    sum_error = 0.0
    for i in range(len(actual)):
        prediction_error = predicted[i] - actual[i]
        sum_error += (prediction_error ** 2)
        mean_error = sum_error / float(len(actual))
    return mean_error



def Train(X,Y):
    ''' With this function we are calculate the weights   '''
    X.astype(float)
    first=np.dot(X.T,X)
    first.astype(np.float16)
    inverse=np.linalg.inv(first)
    second=np.dot(X.T,Y)
    
    b=np.dot(inverse,second)
    return b


def add_bias(x):
    if (len(x.shape)==1):
        x=x[:,np.newaxis]
    b=np.ones((x.shape[0],1))
    nx=np.concatenate((b,x), axis=1)
    return nx

def prepare_y(x):
    if (len(x.shape)==1):
        x=x[:,np.newaxis]
    
    return x   

def predict(X,b):
    return (np.dot(X,b))




df=pd.read_csv('MpgData_with_Cateogeries.csv')
col=df.columns
we=df.to_numpy()
we=we[:,0:8]
we=we.astype(np.float64)
df.head()



xtrain=we[:292,1:8]
ytrain=we[:292,0]
xtest=we[292:,1:8]
ytest=we[292:,0] 




for i in range(2,8):
    x_train=add_bias(xtrain[:,0:i])
    x_test=add_bias(xtest[:,0:i])
    
    
    b=Train(x_train,ytrain)
    
    train_predict=predict(x_train,b)
    train_error=msee(ytrain,train_predict)
    print('Training  Error for Multivariable regression using  {} variables is   {}  '.format(i,train_error))
    
    
    
    
    
for i in range(2,8):
    x_train=add_bias(xtrain[:,0:i])
    x_test=add_bias(xtest[:,0:i])
    b=Train(x_train,ytrain)
    test_predict=predict(x_test,b)
    test_error=msee(ytest,test_predict)
    print('Testing Error for Multivariable regression using  {} variables is   {}  '.format(i,test_error))
    
    


x_train=add_bias(xtrain)
x_test=add_bias(xtest)
b=Train(x_train,ytrain)
test_predict=predict(x_test,b)
plt.figure(figsize=(10,5))
plt.title('Multivariate linear regression for Test data',fontsize=16)
plt.grid(True)
plt.plot(ytest , color='purple')
plt.plot(test_predict , color='red'  )
plt.show()






