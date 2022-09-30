
import numpy as np
import pandas as pd
df=pd.read_csv('Admission_Predict.csv')
result=df.copy()
for feature in df.columns:
    max_value=df[feature].max()
    min_value=df[feature].min()
    result[feature]=(df[feature]-min_value)/(max_value-min_value)*0.8+0.1
array=result.to_numpy()
i=int(input("No of inputs = "))
o=int(input("No of outputs = "))
train=int(input("No of training patterns = "))
test=int(input("No of testing patterns = ")) 
T=[]
Input=[]
for row in array[:train]:
    Input.append(row[:i])
    T.append(row[i:i+o])
I=np.insert(Input,0,1,axis=1)
h=int(input("No of hidden neurons = "))

v=np.random.rand(i+1,h)
w=np.random.rand(h+1,o)
from sklearn.metrics import mean_squared_error
count=0
dw=np.zeros([h+1,o],dtype=float)
dv=np.zeros([i+1,h],dtype=float)
alpha=0.5
eta=0.25
#error=mean_squared_error(T,O)
error_values=[]
iterations=[]
minerr=1
#error_values.append(error)
while(count<10000):
    IH=np.dot(I,v)
    O_H=[[1/(1+np.exp(-x)) for x in IH[i]]for i in range(len(IH))]
    OH=np.insert(O_H,0,1,axis=1)
    IO=np.dot(OH,w)
    O=[[1/(1+np.exp(-x)) for x in IO[i]]for i in range(len(IO))]
    error=mean_squared_error(T,O)
    if(error<minerr):
        minerr=error
        it=count 
    error_values.append(error)
    for j in range(h+1):
        for k in range(o):
            sum=0
            for p in range(train):
                sum+=(T[p][k]-O[p][k])*O[p][k]*(1-O[p][k])*OH[p][j]
            w[j][k]=w[j][k]+eta*sum/train+alpha*dw[j][k]
            dw[j][k]=eta*sum/train
    
    for m in range(i+1):
        for j in range(h):
            sum=0
            for p in range(train):
                for k in range(o):
      
                  sum+=(T[p][k]-O[p][k])*O[p][k]*(1-O[p][k])*w[j][k]*OH[p][j]*(1-OH[p][j])*I[p][m]
            v[m][j]=v[m][j]+sum*eta/(o*train)+alpha*dv[m][j]
            dv[m][j]=sum*eta/(o*train)
    
    count+=1
    #if (error_values[count]-error_values[count-1])/error_values[count-1]<0.000000000001:
        #break
    iterations.append(count)
    print(count,end=",")
    print(error)
print('mean squared error = ',error)
print('no of iterations = ',count)

print('updated v and w matrices are :')
print('v = ',v)
print('w = ',w)

#testing

T_test=[]
Input_test=[]
for row in array[train:train+test]:
    Input_test.append(row[:i])
    T_test.append(row[i:i+o])
I_test=np.insert(Input_test,0,1,axis=1)
IH_test=np.dot(I_test,v)
O_H_test=[[1/(1+np.exp(-x)) for x in IH_test[i]]for i in range(len(IH_test))]
OH_test=np.insert(O_H_test,0,1,axis=1)
IO_test=np.dot(OH_test,w)
O_test=[[1/(1+np.exp(-x)) for x in IO_test[i]]for i in range(len(IO_test))]
error=mean_squared_error(T_test,O_test)
print('mean squared error of testing pattern = ',error)

    
print(minerr) 
print(it)  
import matplotlib.pyplot as plt
plt.plot(iterations,error_values,c='g')
plt.xlim(iterations[1],iterations[9999])
plt.ylim(0.009,0.2)
plt.xlabel('no of iterations')
plt.ylabel('error')
plt.title('overfitting plot in range 0 to 10000 iteration')
plt.show()
