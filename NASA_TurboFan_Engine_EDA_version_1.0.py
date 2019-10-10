#!/usr/bin/env python
# coding: utf-8

# # Dataset name : Turbofan Engine Degradation Simulation Data Set

# This dataset comprises of data that gives us information of when the turbofan engine fails and for that this dataset
# provides various columns/features :-

#     1.Unit                 : This tells us about the unit number of turbofan engine.
#     2.Operational settings : These features tells us about different operational setting have been applied on the engine.
#                              Their are total three operational settings for each cycle(Time).
#     3.Sensor measurement   : These features gives the result of the effect of those operational 
#                              setting's combination applied upon the engine. Their are total 26 measurements for 
#                              each settings combinations.
# 

# The datasets which were provided to us are comprise 4 training dataset,4 testing dataset and 4 Y-label/dependent dataset for testing dataset. In theses dataset we have to do Exploratory Data Analysis(EDA) to visualize,analysis to know which of the features are important and more importantly to know what is EDA.

# In[988]:


#In this cell all the libraries required for the EDA are imported down below.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[989]:


#Here we have loaded our Train Dataset(train_FD001) 
df = pd.read_csv("Desktop/train_FD001.txt",sep=" ",header=None)
#There were two columns filled with NaN values so dropping them.
print(df.columns)
for names in df.columns:
    
    if (df[names].isnull().sum() >1):
         df.drop(names,axis = 1,inplace = True)
#df.drop([26,27],axis = 1,inplace = True)
#The names of the columns were missing, proving headers to the column 
df.columns = ["unit","time(in cycles)","os1","os2","os3","sensor1","sensor2","sensor3","sensor4","sensor5","sensor6","sensor7","sensor8","sensor9","sensor10","sensor11","sensor12","sensor13","sensor14","sensor15","sensor16","sensor17","sensor18","sensor19","sensor20","sensor21"] 
df.head()
print(df.columns)


# In[990]:


df.info()
#Here we can see all the columns are numerical typeand having non NULL values.
#Total rows = 20631 (0 to 20630)
#Total columns = 26


# In[991]:


df.head()


# In[992]:


df.describe().T
#here we can see among all the variables some of the variables are having constant values


# In[993]:


#but first in order to remove those columns we need to see what effect they are putting on the label column or, the dependent variables
#for that we need dependent variable
#from the article and readme provided we found out that their is a RUL file that contains a column which is actually a label column and it seems that it is a label/dependent column of test dataset
#there is no label variable for Train dataset so we have to build one i.e. RUL column


# RUL means remaining useful life of component that means for how long that specific component will survive and our training dataset is run-to-failure dataset that means it is answering us that at this point of time/cycles the component will fail but is it dependent upon os or not that we will have to check

# In[994]:


#First we have to find out what is the correlation of all the os and sensors
df1 = df.drop('unit',axis=1)
df1.corr()


# In[995]:


#As we can see the colinearity techinque is not so helpful right now but we can see that os3,sensor18,19 are totally not 
#showing any value and previously we saw they were constant


# In[996]:


#Now in order to check colinearity of these columns/feature with the label variable we need to create one Y-label variable 
df['RUL'] = df.groupby(['unit'])['time(in cycles)'].transform(max)-df['time(in cycles)']


# In[997]:


df.head()


# In[998]:


#Checking which all sensors are getting effected by the RUL column(Y-label) using pair plot
g = sns.pairplot(data=df[df['unit']<10] ,
                 y_vars=["os1","os2","os3","sensor1","sensor2","sensor3","sensor4","sensor5","sensor6","sensor7","sensor8","sensor9","sensor10","sensor11","sensor12","sensor13","sensor14","sensor15","sensor16","sensor17","sensor18","sensor19","sensor20","sensor21"] ,
                 x_vars=['RUL'],
                 hue="unit", size=5, aspect=2.5)
g = g.map(plt.scatter, alpha=0.5)
g = g.set(xlim=(300,-100))
g = g.add_legend()


# In[999]:


df.head()


# In[1000]:


#dropping the below sensor columns as they are having constant value
for names in df.columns:
    if (df[names].nunique()==1):
        df.drop(names,axis = 1,inplace = True)


# In[1001]:


dropped_columns = ["sensor1","sensor5","sensor10","sensor16","sensor18","sensor19","os3"]
df.head()


# Since OS1 and OS2 separately were having no effect so checking the effect with every available sensors.

# In[1002]:


#The checking has been done using pairplot.
g = sns.pairplot(data=df[df['unit']<10] ,
                 y_vars=["sensor2","sensor3","sensor4","sensor6","sensor7","sensor8","sensor9","sensor11","sensor12","sensor13","sensor14","sensor15","sensor17","sensor20","sensor21"] ,
                 x_vars=["os1","os2"],
                 hue="unit", size=5, aspect=2.5)


# In[1003]:


df.tail()


# In[1004]:


#We saw that pairplot shows that there were similarity between some of the features so in order to check how much similar they
#were, also the features having less importance on y label
corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap='coolwarm')


# In[1005]:


df.columns


# In[1006]:


#here after observing the heatmap we discard all the columns which are not showing co-relation with the RUL column
for name in df.columns:
    for names in df.columns:
        matrix = df[[name,names]].corr()
        if(matrix.iloc[1,0]<=0.55): 
            if((matrix.iloc[1,0]>=-0.55) & ((name != 'RUL') & (name != 'unit') & (name != 'os1') & (name != 'os2')) & (names == 'RUL')):
                print(name)
                print(matrix.iloc[1,0])
                drop_to = name
                df.drop(name,axis=1,inplace=True)
                break
drop_to
df.head()


# Here we are checking whether columns are having any outliers or not using Boxplot for every available sensor.

# In[1007]:


dropped_columns = ["sensor1","sensor5","sensor10","sensor16","sensor18","sensor19","os3","sensor6","sensor9","sensor14"]
df.columns


# In[1008]:


sns.boxplot(x=df['sensor2'])


# In[1009]:


sns.boxplot(x=df['sensor3'])


# In[1023]:


sns.boxplot(df['sensor4'])


# In[1024]:


sns.boxplot(df['sensor7'])


# In[1025]:


sns.boxplot(df['sensor8'])


# In[1010]:


sns.boxplot(x=df['sensor11'])


# In[1021]:


sns.boxplot(df['sensor8'])


# In[1011]:


sns.boxplot(x=df['sensor12'])


# In[1012]:


sns.boxplot(x=df['sensor13'])


# In[1013]:


sns.boxplot(x=df['sensor15'])


# In[1014]:


sns.boxplot(x=df['sensor17'])


# In[1015]:


sns.boxplot(x=df['sensor20'])


# In[1022]:


sns.boxplot(df['sensor21'])


# In[977]:


#To remove those outliers we have used Z-Score and stored it in a numpy array
from scipy import stats
z = np.abs(stats.zscore(df))


# In[971]:


#Printing the numpy array where the Z-score is greater than 3
threshold = 3
print(np.where(z>3))


# In[972]:


df.shape


# In[973]:


x = pd.DataFrame(z)


# In[974]:


#here with the help of z-score we are deleting some rows which will act like an outliers 
df = df[(x<3).all(axis=1)]


# In[1016]:


df.shape


# In[1017]:


df.head()


# In[1018]:


#this is the method which we are thinking to apply in order to know the number of outliers in each column 
#and if those numbers crosses the threshold point then we will discard those features


# In[1020]:


for column in df.columns :
    if ( column !='RUL'):
        q25,q75 = np.percentile(df[column],25),np.percentile(df[column],75)
        Iqr = q75-q25
        cutoff = Iqr*1.5
        lower = q25 - cutoff
        upper = q75 + cutoff
        outliers = [x for x in df[column] if x<lower or x>upper]
        print(column)
        print(len(outliers))


# In[987]:


sns.boxplot(x=df['os1'])


# In[986]:


sns.boxplot(x=df['os2'])


# In[1026]:


df.columns


# # Conclusion: 

# After doing this much EDA we have found that above features are important as of now. We have done only for one DataSet having CONDITION: ONE (Sea Level), there are basically two pairs of dataset having same conditions (ONE and SIX).
# with the help EDA visualize and found out that this much features are important for CONDITION: ONE(Sea Level) dataset.
# We are also looking forward to perform EDA on CONDITION: SIX and after that we are going to build model which will be able to predict the RUL from the test dataset.
