import pandas as pd
import numpy as np
from Actlayer import ActivationLayer
from FClayer import FCLayer
from network import Network
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

def relu(x):
    y=[]
    for i in range(0,len(x[0])):
        y.append(max(0,x[0][i]))
    y = np.array(y)
    y = y.reshape(x.shape)
    return y

def relu_prime(x):
    y=[]
    for i in range(0,len(x[0])):
        if x[0][i]>0 :
            y.append(1)
        else :
            y.append(0)
    y = np.array(y)
    y = y.reshape(x.shape)
    return y

def meanSqError(y_pred,y_true):
    return np.mean(np.power(y_true-y_pred,2))

def meanSqError_prime(y_pred,y_true):
    return 2*(y_pred-y_true)/y_true.size

df = pd.read_csv("data1.csv")
X = df.iloc[:,3:13].values
y = df.iloc[:,13].values

LabelEn1 = LabelEncoder()
X[:,1]=LabelEn1.fit_transform(X[:,1])
LabelEn2 = LabelEncoder()
X[:,2] = LabelEn2.fit_transform(X[:,2])

y = y.reshape(X.shape[0],1,1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_train = X_train.reshape(X_train.shape[0],1,10)
X_test = X_test.reshape(X_test.shape[0],1,10)


model = Network()
model.add(FCLayer(10,6))
model.add(ActivationLayer(relu,relu_prime))
# model.add(FCLayer(6,3))
model.add(FCLayer(6,1))
# model.add(ActivationLayer(relu,relu_prime))
model.use(meanSqError,meanSqError_prime)

model.fit(X_train,y_train,20,0.01)
y_pred = model.predict(X_test)
err = 0
for i in range(0,len(y_pred)):
    print(y_pred[i][0]>0.5,y_test[i][0])
    err += meanSqError(y_pred[i][0],y_test[i][0])
print(err/len(y_pred))


































