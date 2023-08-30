# Iris Data Set (Mini Project)
## The dataset contains 4 distinct attribute information. 
### Species mentioned are Iris-setosa,Iris-versicolor and Iris-virginica.




## Import Modules


```python
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
```

## Loading Dataset


```python
df = pd.read_csv('Iris (1).csv')
df.head()
df=df.drop(columns=['Id'])
```


```python
#Overview statistic display
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SepalLengthCm</th>
      <th>SepalWidthCm</th>
      <th>PetalLengthCm</th>
      <th>PetalWidthCm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>150.000000</td>
      <td>150.000000</td>
      <td>150.000000</td>
      <td>150.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5.843333</td>
      <td>3.054000</td>
      <td>3.758667</td>
      <td>1.198667</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.828066</td>
      <td>0.433594</td>
      <td>1.764420</td>
      <td>0.763161</td>
    </tr>
    <tr>
      <th>min</th>
      <td>4.300000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>0.100000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>5.100000</td>
      <td>2.800000</td>
      <td>1.600000</td>
      <td>0.300000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>5.800000</td>
      <td>3.000000</td>
      <td>4.350000</td>
      <td>1.300000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>6.400000</td>
      <td>3.300000</td>
      <td>5.100000</td>
      <td>1.800000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>7.900000</td>
      <td>4.400000</td>
      <td>6.900000</td>
      <td>2.500000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Display data types present in dataset
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 150 entries, 0 to 149
    Data columns (total 5 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   SepalLengthCm  150 non-null    float64
     1   SepalWidthCm   150 non-null    float64
     2   PetalLengthCm  150 non-null    float64
     3   PetalWidthCm   150 non-null    float64
     4   Species        150 non-null    object 
    dtypes: float64(4), object(1)
    memory usage: 6.0+ KB
    


```python
df['Species'].value_counts()
```




    Iris-virginica     50
    Iris-versicolor    50
    Iris-setosa        50
    Name: Species, dtype: int64



## Preprocessing of Dataset


```python
#Check for Null values
df.isnull().sum()
```




    SepalLengthCm    0
    SepalWidthCm     0
    PetalLengthCm    0
    PetalWidthCm     0
    Species          0
    dtype: int64



## Exploratory Data Analysis



```python
#Histogram Rep.
df['SepalLengthCm'].hist()
```




    <AxesSubplot:>




    
![png](output_11_1.png)
    



```python
df['SepalWidthCm'].hist()
```




    <AxesSubplot:>




    
![png](output_12_1.png)
    



```python
df['PetalLengthCm'].hist()
```




    <AxesSubplot:>




    
![png](output_13_1.png)
    



```python
df['PetalWidthCm'].hist()
```




    <AxesSubplot:>




    
![png](output_14_1.png)
    



```python
#Scatter Plot
colors =['red','green','blue']
species=['Iris-setosa','Iris-versicolor','Iris-virginica']
```


```python
for i in range(3):
    x=df[df['Species']== species[i]]
    plt.scatter(x['SepalLengthCm'],x['SepalWidthCm'],c=colors[i],label=species[i])
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Widht')
plt.legend()
```




    <matplotlib.legend.Legend at 0x1b3bd17ddc0>




    
![png](output_16_1.png)
    



```python
for i in range(3):
    x=df[df['Species']== species[i]]
    plt.scatter(x['PetalLengthCm'],x['PetalWidthCm'],c=colors[i],label=species[i])
plt.xlabel('Petal Length')
plt.ylabel('Petal Widht')
plt.legend()
```




    <matplotlib.legend.Legend at 0x1b3bd1feb20>




    
![png](output_17_1.png)
    



```python
for i in range(3):
    x=df[df['Species']== species[i]]
    plt.scatter(x['SepalLengthCm'],x['PetalLengthCm'],c=colors[i],label=species[i])
plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')
plt.legend()
```




    <matplotlib.legend.Legend at 0x1b3bd279e80>




    
![png](output_18_1.png)
    



```python
for i in range(3):
    x=df[df['Species']== species[i]]
    plt.scatter(x['SepalWidthCm'],x['PetalWidthCm'],c=colors[i],label=species[i])
plt.xlabel('Sepal Width')
plt.ylabel('Petal Width')
plt.legend()
```




    <matplotlib.legend.Legend at 0x1b3bd2d0b80>




    
![png](output_19_1.png)
    


### After visualizing through scatter plot we can select the attributes which can help distinguish between two species based on the distribution of data


```python
#Petal Widht vs Petal Length appears to give the best distinction between different species.
```

## Correaltion Check


```python
df.corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SepalLengthCm</th>
      <th>SepalWidthCm</th>
      <th>PetalLengthCm</th>
      <th>PetalWidthCm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>SepalLengthCm</th>
      <td>1.000000</td>
      <td>-0.109369</td>
      <td>0.871754</td>
      <td>0.817954</td>
    </tr>
    <tr>
      <th>SepalWidthCm</th>
      <td>-0.109369</td>
      <td>1.000000</td>
      <td>-0.420516</td>
      <td>-0.356544</td>
    </tr>
    <tr>
      <th>PetalLengthCm</th>
      <td>0.871754</td>
      <td>-0.420516</td>
      <td>1.000000</td>
      <td>0.962757</td>
    </tr>
    <tr>
      <th>PetalWidthCm</th>
      <td>0.817954</td>
      <td>-0.356544</td>
      <td>0.962757</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Displaying correaltion in form of heat map
corr=df.corr()
fig,ax=plt.subplots(figsize=(6,6))
sns.heatmap(corr,annot=True,ax=ax,cmap='coolwarm')
```




    <AxesSubplot:>




    
![png](output_24_1.png)
    



```python
from sklearn.preprocessing import LabelEncoder
le =LabelEncoder()
```


```python
df['Species']=le.fit_transform(df['Species'])
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SepalLengthCm</th>
      <th>SepalWidthCm</th>
      <th>PetalLengthCm</th>
      <th>PetalWidthCm</th>
      <th>Species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## Training Model


```python
from sklearn.model_selection import train_test_split
X=df.drop(columns=['Species'])
Y=df['Species']
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.30)
```


```python
#logistic Regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
```


```python
#model training
model.fit(x_train,y_train)
```

    C:\Users\arune\anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py:763: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    




    LogisticRegression()




```python
#print metric to get performance
print("Accuracy: ",model.score(x_test,y_test)*100)
```

    Accuracy:  95.55555555555556
    


```python
#K-Nearest Neighbours
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
```


```python
#model training
model.fit(x_train,y_train)
```




    KNeighborsClassifier()




```python
#print metric to get performance
print("Accuracy: ",model.score(x_test,y_test)*100)
```

    Accuracy:  97.77777777777777
    


```python
#Decision Tree
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
```


```python
model.fit(x_train,y_train)
```




    DecisionTreeClassifier()




```python
#print metric to get performance
print("Accuracy: ",model.score(x_test,y_test)*100)
```

    Accuracy:  91.11111111111111
    

## Model Training Train Test Split 80:20


```python
from sklearn.model_selection import train_test_split
X=df.drop(columns=['Species'])
Y=df['Species']
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.20)
```


```python
#logistic Regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
```


```python
#model training
model.fit(x_train,y_train)
```

    C:\Users\arune\anaconda3\lib\site-packages\sklearn\linear_model\_logistic.py:763: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    




    LogisticRegression()




```python
#print metric to get performance
print("Accuracy: ",model.score(x_test,y_test)*100)
```

    Accuracy:  100.0
    


```python
#K-Nearest Neighbours
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
```


```python
#model training
model.fit(x_train,y_train)
```




    KNeighborsClassifier()




```python
#print metric to get performance
print("Accuracy: ",model.score(x_test,y_test)*100)
```

    Accuracy:  100.0
    


```python
#Decision Tree
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
```


```python
model.fit(x_train,y_train)
```




    DecisionTreeClassifier()




```python
#print metric to get performance
print("Accuracy: ",model.score(x_test,y_test)*100)
```

    Accuracy:  96.66666666666667
    

## Logistic Regression:

### While the method’s name includes the term regression, it is really a classification technique designed to predict binary outcomes. True/False, Yes/No, Pass/Fail, or 1/0, for example.

### Independent Variables are assessed to predict the binary result where it can either be 1(True) or 0(False). The independent variable can be of either numeric type or have a category, but dependent variable is always categorical.

## K-Nearest Neighbor: 
### It is pattern recognition algorithm that is used to train data set to form clusters by identifying its closest data points by measuring the distance between points in a plot.

## Decision Tree Algorithm:
### It is the most appropriate algorithm which can be used to classify dataset. As it works like a flow chart which keeps differentiating from top to deepest root, where even similar items can be categorized based on difference in attribute making them fall under different categories that can separate them the tree trunk to branch to leaves, where they become more finitely similar. And when you visualize it, the top row acts as root of the tree hence the name Decision Tree. 




```python

```
