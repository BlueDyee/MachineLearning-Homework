# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 09:57:49 2022

@author: user
"""

import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd

X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
tmp=np.array([range(100)])
tmp=np.append(tmp,[range(100)])
print(tmp)
print(len(tmp))
a = np.array([[1,1,1],[2,2,2]])
b = np.append(a, [[3],[4]],1)
print(b)
y = np.dot(X, np.array([1, 2])) + 3
reg = LinearRegression().fit(X, y)

def predict(W,x):
    return np.matmul(W,x)
def Error(W,X,Y):
    count=0
    N=Y.size
    for i in range(N):
        tmp=predict(W,X[i])-Y[i]
        count+=tmp*tmp
    return count/(2*N)

X_trans=np.transpose(X)
tmp=np.linalg.inv(np.matmul(X_trans,X))
W=np.matmul(np.matmul(tmp,X_trans),y)

#print(type(W))
#print(type(reg.coef_))
#print(Error(reg.coef_,X,y))
"""
import pandas as pd
df = pd.DataFrame(
	[[21, 72, 67],
	[23, 78, 69],
	[32, 74, 56],
	[52, 54, 76]],
	columns=['a', 'b', 'c'])

print('DataFrame\n----------\n', df)

#convert dataframe to numpy array
arr = df.to_numpy()
print(type(arr))
print('\nNumpy Array\n----------\n', arr)
"""
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.linear_model import LinearRegression
wine_data=pd.read_csv("C:\\Users\\user\\jupyter_notebook\\ML2022_HW1\\X.csv")
wine_target=pd.read_csv("C:\\Users\\user\\jupyter_notebook\\ML2022_HW1\\T.csv")

wine_features=["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]
X=wine_data[wine_features]
Y=wine_target["quality"]
print(X)
#print(Y)
dt=list(range(1599))
#print(dt)
pdt=pd.DataFrame(dt)
print(pdt)
X_pred=X*5+3
tmp=X+pdt
print(tmp)
#%%
def split_DataFrame(X,N):
    df_1=X.iloc[:N,:]
    df_2=X.iloc[N:,:]
    return df_1, df_2
def predict(W0,W1,X):
    X=X.reshape(len(X),1)
    return np.matmul(W1,X)+W0
def GD_M1(X,Y):
    (N,D)=X.shape
    features=X.columns
    L=0.0001
    epochs=1
    W0=1.
    dict_tmp={}
    for f in features:
        dict_tmp[f]=[0.]
    W=pd.DataFrame(dict_tmp)

    for i in range(epochs):
        new_W0=0.
        new_W=pd.DataFrame(dict_tmp)
        print("start")
        for j in range(N):# range(N)
            cur_W_sum=(X.iloc[j]*W).sum(1)+W0 # X[j]跟W dot +截距
            new_W0+=(cur_W_sum.at[0]-Y[j])

            for f in features:
                new_W[f]+=(cur_W_sum.at[0]-Y[j])*X[f][j]
                
        W0=W0-L*(new_W0)/N      
        for f in features:
            W[f]=W[f]-L*new_W[f]/N
        print("W:",W0,W)
    return W0,W
def regression_M2(X,Y):
    X_numpy=X.to_numpy()
    H=X_numpy
    for i in range(D):
        for j in range(D):
            tmp=X_numpy[:,i]*X_numpy[:,j]
            tmp=tmp.reshape(N,1)
            H=np.append(H,tmp,1)
            #print(H.shape)
    ones=np.array([[1]]*N)
    H=np.append(H,ones,1)
    Y_numpy=Y.to_numpy()
    H_trans=np.transpose(H)
    mul=np.matmul(H_trans,H)
    tmp=np.linalg.pinv(mul)
    W=np.matmul(np.matmul(tmp,H_trans),Y_numpy)
    intercept=W[-1]
    W=np.delete(W,-1)
    H=np.delete(H,-1,1)
    return intercept, W ,H
def Error_numpy(W0,W,X,Y):
    count=0
    N=Y.size
    
    for i in range(N):
        tmp=predict(W0,W,X[i])-Y[i]
        count+=tmp*tmp
    return count/(2*N)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
wine_data=pd.read_csv("C:\\Users\\user\\jupyter_notebook\\ML2022_HW1\\X.csv")
wine_target=pd.read_csv("C:\\Users\\user\\jupyter_notebook\\ML2022_HW1\\T.csv")

#wine_features=["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]
wine_features=wine_data.columns
X=wine_data[wine_features]
Y=wine_target["quality"]
N,D=X.shape
dict_tmp={}
for f in wine_features:
    dict_tmp[f]=[1.]
W=pd.DataFrame(dict_tmp)
dict_count={}
for i in range(10):
    dict_count[i]=0
for num in Y:
    dict_count[num]+=1
print("target distribution:\n",dict_count)
tmp_list=[]
for i in range(10):
    tmp_list.append(dict_count[i])
Y_count_data=pd.DataFrame(tmp_list)
plt.figure(figsize=(20, 15))
plt.xlabel("quality", fontweight = "bold")                  #設定x座標標題及粗體
plt.ylabel("count", fontweight = "bold")   #設定y座標標題及粗體
plt.title("distribution of Y",
              fontsize = 15, fontweight = "bold") 
plt.plot(Y_count_data)
plt.savefig("D:\上課\機器學習\ML2022_HW1\plots\Distribution of Y.jpg")
#plt.plot(Y)
(W0,W,H)=regression_M2(X, Y)

print(Error_numpy(W0,W,H,Y))

#print(X.iloc[0])
#print(W)
#cur_W_sum=(X.iloc[0]*W).sum(1)
#print(cur_W_sum-Y[0])

#print(GD_M1(X,Y))

"""
i=1
plt.figure(figsize=(20, 15))
plt.xlabel("index of Y", fontweight = "bold")                  #設定x座標標題及粗體
plt.ylabel("quality", fontweight = "bold")   #設定y座標標題及粗體
plt.title("Scatter of Y and "+ "quality",
              fontsize = 15, fontweight = "bold") 
plt.scatter(range(Y.size),Y,     # y軸資料
            c = "m",                                  # 點顏色
            s = 80,                                   # 點大小
            alpha = .2,                               # 透明度
            )
plt.savefig("D:\上課\機器學習\ML2022_HW1\plots\Scatter of Y.jpg")  
for f in wine_features:
    plt.figure(figsize=(20, 15))
    plt.xlabel(f, fontweight = "bold")                  #設定x座標標題及粗體
    plt.ylabel("quality", fontweight = "bold")   #設定y座標標題及粗體
    plt.title("Scatter of "+f+" and "+ "quality",
              fontsize = 15, fontweight = "bold") 
    plt.scatter(X[f],                   # x軸資料
            Y,     # y軸資料
            c = "m",                                  # 點顏色
            s = 80,                                   # 點大小
            alpha = .2,                               # 透明度
            )                             # 點樣式
    i+=1
    plt.savefig("D:\上課\機器學習\ML2022_HW1\plots\Scatter of "+f+".jpg")   #儲存圖檔
    plt.close()      # 關閉圖表
    
"""
"""
#panda access

for f in wine_features:
    #W[f]+=1
    #print(W[f])
    
N,D=X.shape
print(N)
print(D)
features=X.columns
print(features)
print(type(features))
print("xx",Y[0])
for f in features:
    print(X[f][0])
print(features[2])
"""
"""
#delete rows by index i

for i in range(1,N):
    W=W.drop(i)
"""
#print(X)
#print(W)
"""
#sum vector dot product

tmp=(X.iloc[1]*W)
sum=tmp.sum(1)

tmp=tmp.dropna()
print(tmp)
for i in range(1,N):
    sum+=(X.iloc[i]*W).sum(1)
print(sum)
tmp=sum.at[0]
print(type(tmp))
"""

#GD for M=1

"""
def GD_M1(X,Y):
    (N,D)=X.shape
    features=X.columns
    L=0.001
    epochs=100
    W0=1
    dict_tmp={}
    for f in features:
        dict_tmp[f]=[1]
    W=pd.DataFrame(dict_tmp)
    
    for i in range(epochs):
        new_W0=0
        new_W=[0]*D
        for j in range(N):
            cur_W_sum=(X.iloc[j]*W).sum(1)
            new_W0+=(cur_W_sum.at[0]-Y[j])
            for k in range(D):
                new_W[k]+=(cur_W_sum.at[0]-Y[j])*X[features[k]][j]
            
       # W_sum=W_sum.at[0]
        #new_w.append(W_sum-)
 """       

#print(X_numpy.shape)
#X_numpy=np.append(X_numpy,X_numpy,1) #沒有axis會直接攤平
#print(X_numpy[:,0].shape)
#print(X_numpy[:,0]*X_numpy[:,0])
"""
def regression_M2(X,Y):
X_numpy=X.to_numpy()
H=X_numpy
for i in range(D):
    for j in range(D):
        tmp=X_numpy[:,i]*X_numpy[:,j]
        tmp=tmp.reshape(N,1)
        H=np.append(H,tmp,1)
#print(H.shape)
ones=np.array([[1]]*N)
H=np.append(H,ones,1)
Y_numpy=Y.to_numpy()
H_trans=np.transpose(H)
mul=np.matmul(H_trans,H)
tmp=np.linalg.pinv(mul)
W=np.matmul(np.matmul(tmp,H_trans),Y_numpy)
intercept=W[-1]
W=np.delete(W,-1)
return intercept, W
print(W.shape)
print(W)
"""
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def variance_numpy(W0,W,X,Y):
    N,D=X.shape
    count=0
    for i in range(N):
        count+=(predict(W0, W, X[i])-Y[i])**2
    count/=N
    return count
def gaussian(x,u,var):
    deviation=np.sqrt(var)
    tmp=np.sqrt(2*np.pi)
    probability=np.exp(-(x-u)**2/(2*var))/(deviation*tmp)
    return probability
def predict(W0,W1,X):
    X=X.reshape(len(X),1)
    return np.matmul(W1,X)+W0
def regression_M1(X,Y):
    X_numpy=X.to_numpy()  #for pandas.dataframe to numpy.ndarray
    Y_numpy=Y.to_numpy()
    N=len(X_numpy)

    ones=np.array([[1]]*N) #for intercept
    X_numpy=np.append(X_numpy,ones,1) 
#--------Calculating W-------
    X_trans=np.transpose(X_numpy)
    tmp=np.linalg.inv(np.matmul(X_trans,X_numpy))
    W=np.matmul(np.matmul(tmp,X_trans),Y_numpy)

    intercept=W[-1]
    W=np.delete(W,-1)
    X_numpy=np.delete(X_numpy,-1,1)
    return (intercept, W)

def regression_M2(X,Y):
    X_numpy=X.to_numpy()
    H=X_numpy
    N, D=X_numpy.shape
    for i in range(D):
        for j in range(D):
            tmp=X_numpy[:,i]*X_numpy[:,j]
            tmp=tmp.reshape(N,1)
            H=np.append(H,tmp,1)
            #print(H.shape)
    ones=np.array([[1]]*N)
    H=np.append(H,ones,1)
    Y_numpy=Y.to_numpy()
    H_trans=np.transpose(H)
    mul=np.matmul(H_trans,H)
    tmp=np.linalg.pinv(mul)
    W=np.matmul(np.matmul(tmp,H_trans),Y_numpy)
    intercept=W[-1]
    W=np.delete(W,-1)
    H=np.delete(H,-1,1)
    return intercept, W
def X_of_M2(X):
  X_numpy=X.to_numpy()
  H=X_numpy
  N, D=X_numpy.shape
  for i in range(D):
      for j in range(D):
          tmp=X_numpy[:,i]*X_numpy[:,j]
          tmp=tmp.reshape(N,1)
          H=np.append(H,tmp,1)
  return H

wine_data=pd.read_csv("C:\\Users\\user\\jupyter_notebook\\ML2022_HW1\\X.csv")
wine_target=pd.read_csv("C:\\Users\\user\\jupyter_notebook\\ML2022_HW1\\T.csv")

wine_features=wine_data.columns
X=wine_data[wine_features]
Y=wine_target
X_numpy=X.to_numpy()
Y_numpy=Y.to_numpy()
N,D=X.shape

W0, W=regression_M1(X,Y)
var=variance_numpy(W0, W, X_numpy, Y_numpy)
#print(var)
log_likelihood=np.float64(0)
for i in range(N):
    p=gaussian(Y_numpy[i],predict(W0,W,X_numpy[i]),var)
    log_likelihood+=np.log(p)
print("M=1,log_likelihood",log_likelihood)
############################################################
"""
(W02, W2)=regression_M2(X,Y)
H=X_of_M2(X)
var=variance_numpy(W02, W2, H, Y_numpy)
log_likelihood=np.float64(0)
for i in range(N):
    p=gaussian(Y_numpy[i],predict(W02,W2,H[i]),var)
    log_likelihood+=np.log(p)
print("M=2,log_likelihood",log_likelihood)
"""
X_train, X_test=split_DataFrame(X,1500)
Y_train, Y_test=split_DataFrame(Y,1500)
X_train_numpy=X_train.to_numpy()
Y_train_numpy=Y_train.to_numpy()
X_test_numpy=X_test.to_numpy()
Y_test_numpy=Y_test.to_numpy()


for m in range(1,4):
  W0, W, H_train=regression_M(X_train,Y_train,m)
  print(W0)
  print(W.shape)
  print(H_train.shape)
  print(Y_train.shape)
  var=variance_numpy(W0, W, H_train, Y_train)
  log_likelihood=np.float64(0)
  
#--------Testing stage------------ 
  H_test=polynomial_form_M(X_test,m)
  N,D=H_test.shape
  for i in range(N):
      p=gaussian(Y_test_numpy[i],predict(W0,W,H_test[i]),var)
      log_likelihood+=np.log(p)
  print("M=",m, "log_likelihood",log_likelihood)
        
    