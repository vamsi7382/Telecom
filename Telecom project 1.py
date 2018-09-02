
# coding: utf-8

# In[3]:


# IMPORTING THE REQUIRED LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


# IMPORTING THE DATASET 
churn=pd.read_csv("E:\\DATA SCIENCE\\PROJECT\\project telecom\\TRAIN.csv")


# In[5]:


churn


# In[6]:


churn.shape


# In[7]:


# REMOVING THE DUPLICATES FROM DATASET
churn.duplicated().sum()


# In[8]:


# REMOVING THE ID TYPE ATTRIBUTES FROM OUR DATASET
churn1=churn.drop(["Customer ID"],axis=1)


# In[9]:


churn1.shape


# In[10]:


churn1.dtypes


# In[11]:


# CHECKING THE MISSING VALUES IN THE DATASET (churn1)
churn1.isnull().sum()


# In[12]:


# FINDING THE MISSING VALUES PERCENTAGE FOR EACH VARIABLE
x=((churn1.isnull().sum())/len(churn1))*100


# In[13]:


x


# In[14]:


# IMPUTE MISSING VALUES 
churn1.fillna(method="bfill")


# In[15]:


x=((churn1.isnull().sum())/len(churn1))*100


# In[16]:


x


# In[17]:


churn2=churn1.fillna(method="bfill")


# In[18]:


churn2.isnull().sum()


# In[19]:


churn2=churn1.fillna(method="ffill")


# In[20]:


churn2.isnull().sum()


# In[21]:


# CHECKING THE OUTLIEARS IN THE DATASET
# OUTLIERS CAN BE DONE ONLY FOR THE NUMERIC ATTRIBUTES..
# 1. OUTLIERS FOR "network_age" attribute
churn2.boxplot("network_age")


# In[22]:


# 2. For "Customer tenure in month" 
churn2.boxplot("Customer tenure in month")


# In[23]:


# Cross checking the outliers
churn2["Customer tenure in month"].quantile([0,0.25,0.5,0.75,0.95,0.997,1])


# In[24]:


# 3. For "Total Spend in Months 1 and 2 of 2017" variable
churn2.boxplot("Total Spend in Months 1 and 2 of 2017")


# In[25]:


churn2["Total Spend in Months 1 and 2 of 2017"].quantile([0,0.25,0.5,0.75,0.95,0.997,1])


# In[26]:


churn2["Total Spend in Months 1 and 2 of 2017"]=np.where(churn2["Total Spend in Months 1 and 2 of 2017"]>2232.41,2232.41,churn2["Total Spend in Months 1 and 2 of 2017"])


# In[27]:


churn2["Total Spend in Months 1 and 2 of 2017"].quantile([0,0.25,0.5,0.75,0.95,0.997,1])


# In[28]:


churn2["Total Spend in Months 1 and 2 of 2017"].max()


# In[29]:


churn2.boxplot("Total Spend in Months 1 and 2 of 2017")


# In[30]:


# 4. For "Total SMS Spend" variable
churn2.boxplot("Total SMS Spend")


# In[31]:


churn2["Total SMS Spend"].quantile([0,0.25,0.5,0.75,0.95,0.997,1])


# In[32]:


churn2["Total SMS Spend"]=np.where(churn2["Total SMS Spend"]>121.26,121.26,churn2["Total SMS Spend"])


# In[33]:


churn2.boxplot("Total SMS Spend")


# In[34]:


# 4. For "Total Data Spend" variable
churn2.boxplot("Total Data Spend")


# In[35]:


churn2["Total Data Spend"].quantile([0,0.25,0.5,0.75,0.95,0.997,1])


# In[36]:


churn2["Total Data Spend"]=np.where(churn2["Total Data Spend"]>196.25,196.25,churn2["Total Data Spend"])


# In[37]:


# 5. For "Total Data Consumption"
churn2.boxplot("Total Data Consumption")


# In[38]:


churn2["Total Data Consumption"].quantile([0,0.25,0.5,0.75,0.95,0.997,1])


# In[39]:


churn2["Total Data Consumption"]=np.where(churn2["Total Data Consumption"]>1.109908e+07,1.109908e+07,churn2["Total Data Consumption"])


# In[40]:


churn2["Total Data Consumption"].quantile([0,0.25,0.5,0.75,0.95,0.997,1])


# In[41]:


churn2.boxplot("Total Data Consumption")


# In[42]:


# 6. For "Total Unique Calls" variable
churn2.boxplot("Total Unique Calls")


# In[43]:


churn2["Total Unique Calls"].quantile([0,0.25,0.5,0.75,0.95,0.997,1])


# In[44]:


churn2["Total Unique Calls"]=np.where(churn2["Total Unique Calls"]>1853,1853,churn2["Total Unique Calls"])


# In[45]:


churn2.boxplot("Total Unique Calls")


# In[46]:


list(churn2.columns)


# In[47]:


# 7. For "Total Onnet spend" variable
churn2.boxplot("Total Onnet spend ")


# In[48]:


churn2["Total Onnet spend "].quantile([0,0.25,0.5,0.75,0.95,0.997,1])


# In[49]:


churn2["Total Onnet spend "]=np.where(churn2["Total Onnet spend "]>29382,29382,churn2["Total Onnet spend "])


# In[50]:


churn2["Total Onnet spend "].quantile([0,0.25,0.5,0.75,0.95,0.997,1])


# In[51]:


churn2.boxplot("Total Onnet spend ")


# In[52]:


# 8. For "Total Offnet spend" variable
churn2.boxplot("Total Offnet spend")


# In[53]:


churn2["Total Offnet spend"].quantile([0,0.25,0.5,0.75,0.95,0.997,1])


# In[54]:


churn2["Total Offnet spend"]=np.where(churn2["Total Offnet spend"]>68051,68051,churn2["Total Offnet spend"])


# In[55]:


churn2.boxplot("Total Offnet spend")


# In[56]:


# We had treated OUTLIERS in our dataset..


# In[57]:


# EDA (EXPLORATORY DATA ANALYSIS)
# UNI-VARIATE ANALYSIS FOR DATASET
# FOR ALL NUMERIC VARIABLES, WE HAD A SINGLE LINE CODE..
churn2.hist(figsize=(15,20),xlabelsize=12,ylabelsize=12)


# In[58]:


churn2


# In[59]:


# For categorial variables, we need to write a code for each variable..
sns.countplot(churn2["Most Loved Competitor network in in Month 1"])


# In[60]:


# EDA is not completed yet, but going for furthur process.
# AFTER EDA, WE HAVE TO DO CORRELATION, Dummies creation & VARIABLE REDUCTION TECHNIQUES..
# Finding the correlation between variables.
import seaborn as sns
corr = churn2.corr()
sns.heatmap(corr, xticklabels=corr.columns,yticklabels=corr.columns, annot=True)


# In[66]:


churn3=pd.get_dummies(churn2)  # Dummies will be created for entire dataset..(For categorical variable)


# In[67]:


churn3.shape


# In[68]:


# Variable reduction techniques..
# 1. I.V method
import pandas as pd
import numpy as np
import pandas.core.algorithms as algos
from pandas import Series
import scipy.stats.stats as stats
import re
import traceback
import string

max_bin = 20
force_bin = 5

# define a binning function
def mono_bin(Y, X, n = max_bin):
    
    df1 = pd.DataFrame({"X": X, "Y": Y})
    justmiss = df1[['X','Y']][df1.X.isnull()]
    notmiss = df1[['X','Y']][df1.X.notnull()]
    r = 0
    while np.abs(r) < 1:
        try:
            d1 = pd.DataFrame({"X": notmiss.X, "Y": notmiss.Y, "Bucket": pd.qcut(notmiss.X, n)})
            d2 = d1.groupby('Bucket', as_index=True)
            r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)
            n = n - 1 
        except Exception as e:
            n = n - 1

    if len(d2) == 1:
        n = force_bin         
        bins = algos.quantile(notmiss.X, np.linspace(0, 1, n))
        if len(np.unique(bins)) == 2:
            bins = np.insert(bins, 0, 1)
            bins[1] = bins[1]-(bins[1]/2)
        d1 = pd.DataFrame({"X": notmiss.X, "Y": notmiss.Y, "Bucket": pd.cut(notmiss.X, np.unique(bins),include_lowest=True)}) 
        d2 = d1.groupby('Bucket', as_index=True)
    
    d3 = pd.DataFrame({},index=[])
    d3["MIN_VALUE"] = d2.min().X
    d3["MAX_VALUE"] = d2.max().X
    d3["COUNT"] = d2.count().Y
    d3["EVENT"] = d2.sum().Y
    d3["NONEVENT"] = d2.count().Y - d2.sum().Y
    d3=d3.reset_index(drop=True)
    
    if len(justmiss.index) > 0:
        d4 = pd.DataFrame({'MIN_VALUE':np.nan},index=[0])
        d4["MAX_VALUE"] = np.nan
        d4["COUNT"] = justmiss.count().Y
        d4["EVENT"] = justmiss.sum().Y
        d4["NONEVENT"] = justmiss.count().Y - justmiss.sum().Y
        d3 = d3.append(d4,ignore_index=True)
    
    d3["EVENT_RATE"] = d3.EVENT/d3.sum().EVENT
    d3["NON_EVENT_RATE"] = d3.NONEVENT/d3.sum().NONEVENT
    d3["WOE"] = np.log(d3.EVENT_RATE/d3.NON_EVENT_RATE)
    d3["IV"] = (d3.EVENT_RATE-d3.NON_EVENT_RATE)*np.log(d3.EVENT_RATE/d3.NON_EVENT_RATE)
    d3["VAR_NAME"] = "VAR"
    d3 = d3[['VAR_NAME','MIN_VALUE', 'MAX_VALUE', 'COUNT', 'EVENT', 'EVENT_RATE', 'NONEVENT', 'NON_EVENT_RATE', 'WOE', 'IV']]       
    d3 = d3.replace([np.inf, -np.inf], 0)
    d3.IV = d3.IV.sum()
    
    return(d3)

def char_bin(Y, X):
        
    df1 = pd.DataFrame({"X": X, "Y": Y})
    df2 = df1.groupby('X',as_index=True)
    
    d3 = pd.DataFrame({},index=[])
    d3["COUNT"] = df2.count().Y
    d3["MIN_VALUE"] = df2.groups
    d3["MAX_VALUE"] = df2.groups
    d3["EVENT"] = df2.sum().Y
    d3["NONEVENT"] = df2.count().Y - df2.sum().Y     
    d3["EVENT_RATE"] = d3.EVENT/d3.sum().EVENT
    d3["NON_EVENT_RATE"] = d3.NONEVENT/d3.sum().NONEVENT
    d3["WOE"] = np.log(d3.EVENT_RATE/d3.NON_EVENT_RATE)
    d3["IV"] = (d3.EVENT_RATE-d3.NON_EVENT_RATE)*np.log(d3.EVENT_RATE/d3.NON_EVENT_RATE)
    d3["VAR_NAME"] = "VAR"
    d3 = d3[['VAR_NAME','MIN_VALUE', 'MAX_VALUE', 'COUNT', 'EVENT', 'EVENT_RATE', 'NONEVENT', 'NON_EVENT_RATE', 'WOE', 'IV']]       
    d3 = d3.replace([np.inf, -np.inf], 0)
    d3.IV = d3.IV.sum()
    d3 = d3.reset_index(drop=True)
    
    return(d3)

def data_vars(df1, target):
    
    stack = traceback.extract_stack()
    filename, lineno, function_name, code = stack[-2]
    vars_name = re.compile(r'\((.*?)\).*$').search(code).groups()[0]
    final = (re.findall(r"[\w']+", vars_name))[-1]
    
    x = df1.dtypes.index
    count = -1
    
    for i in x:
        if i.upper() not in (final.upper()):
            if np.issubdtype(df1[i], np.number) and len(Series.unique(df1[i])) > 2:
                conv = mono_bin(target, df1[i])
                conv["VAR_NAME"] = i
                count = count + 1
            else:
                conv = char_bin(target, df1[i])
                conv["VAR_NAME"] = i            
                count = count + 1
                
            if count == 0:
                iv_df = conv
            else:
                iv_df = iv_df.append(conv,ignore_index=True)
    
    iv = pd.DataFrame({'IV':iv_df.groupby('VAR_NAME').IV.max()})
    iv = iv.reset_index()
    return(iv_df,iv)


# In[69]:


IV = data_vars(churn3,churn3["Churn Status"])


# In[70]:


IV


# In[83]:


IV.shape


# In[72]:


# After Filtering the i.v values, we got 16 attributes in our dataset
type(IV)


# In[87]:


iv1=list(IV)   # IV is in tuple format, so we converting tuple to list format
type(iv1)


# In[88]:


iv1


# In[86]:


iv2=pd.DataFrame(iv1)     # converting the iv1 into dataframe
iv2.shape


# In[89]:


iv2    # Can't display whole iv2 values


# In[90]:


iv2=iv1[1]    # So selecting only iv values in iv1 and storing them into iv2


# In[91]:


iv2


# In[92]:


iv3=pd.DataFrame(iv2)   # Converting the iv2 values into data frame..


# In[93]:


iv3


# In[94]:


iv3.shape


# In[80]:


iv3.to_csv("E:\\Jupiter notebook\\iv3.csv")   # Storing iv3 in csv format


# In[81]:


iv3.shape


# In[95]:


iv3.head()


# In[97]:


churn3.shape


# In[104]:


# Selecting the variables whose i.v values are b/w 0.02 - 0.5 which are done in excel..
variables=["Customer tenure in month",
           "Most Loved Competitor network in in Month 1_PQza",
           "Most Loved Competitor network in in Month 2_Mango",
           "Most Loved Competitor network in in Month 2_PQza",
           "Most Loved Competitor network in in Month 2_ToCall",
           "Most Loved Competitor network in in Month 2_Uxaa",
           "Most Loved Competitor network in in Month 2_Weematel",
           "Network type subscription in Month 2_3G",
           "Network type subscription in Month 2_Other",
           "Total Call centre complaint calls",
           "Total Data Consumption",
           "Total Offnet spend",
           "Total Onnet spend ",
           "Total SMS Spend",
           "Total Unique Calls",
           "network_age",
           "Churn Status"]


# In[105]:


churn4=churn3[variables]    # so, we replace all the variables in churn3 with the i.v modified variables with all observations and store in churn4..


# In[106]:


churn4.shape      # This is our final dataset..


# In[111]:


churn4.head()
churn4


# In[107]:


# Now divide the dataset into train & test datas..
from sklearn.cross_validation import train_test_split
train,test=train_test_split(churn4,train_size=0.8)


# In[108]:


train.shape


# In[109]:


test.shape


# In[112]:


# Cross checking the variavles with vif method.., is there any chance to remove any extra variables..

import statsmodels.formula.api as sm
def vif_cal(input_data, dependent_col):
    x_vars=input_data.drop([dependent_col], axis=1)
    xvar_names=x_vars.columns
    for i in range(0,xvar_names.shape[0]):
        y=x_vars[xvar_names[i]] 
        x=x_vars[xvar_names.drop(xvar_names[i])]
        rsq=sm.ols(formula="y~x", data=x_vars).fit().rsquared  
        vif=round(1/(1-rsq),2)
        print (xvar_names[i], " VIF = " , vif)


# In[113]:


vif_cal(input_data=train, dependent_col="Churn Status")


# In[115]:


# So the first variable has very high vif value, so remove that and again check vif for remaining variables
train1=train.drop(["Customer tenure in month"],axis=1)


# In[116]:


train1.shape


# In[117]:


# So we again check the vif values for the remaining variables in train 1 data
vif_cal(input_data=train1, dependent_col="Churn Status")


# In[118]:


train1=train.drop(["Customer tenure in month","Most Loved Competitor network in in Month 2_Uxaa"],axis=1)


# In[119]:


vif_cal(input_data=train1, dependent_col="Churn Status")


# In[120]:


# So above all variables have vif values < 4, and these are our final variavles to build the model..


# In[121]:


# I want to store train1 data into train2, so train1 data can't be modified..
train2=train1


# In[122]:


# Now we have to divide train2 data into x_train & y_train..
# x_train contains all independent variables and y_train contains only dependent variable..


# In[123]:


x_train=train2.drop(["Churn Status"],axis=1)


# In[124]:


y_train=train2["Churn Status"]


# In[126]:


# So our training dataset is prepared
# Now create the testing dataset data
# The variables that are removed in training data by vif should be removed in testing data also


# In[127]:


# Removing the variables that are removed by vif in training data are removing from testing dataset
test1=test.drop(["Customer tenure in month","Most Loved Competitor network in in Month 2_Uxaa"],axis=1)


# In[128]:


# Checking the vif for test dataset
vif_cal(input_data=test1, dependent_col="Churn Status")


# In[129]:


# So our final testing dataset is prepared
# Now we have to divide tedting data into x_test & y_test..


# In[131]:


x_test=test1.drop(["Churn Status"],axis=1)
x_test.shape


# In[133]:


y_test=test1["Churn Status"]
y_test.shape


# In[134]:


# Finally TRAINING & TESTING datasets are prepared
# On top of them, we have to build our final models..


# In[135]:


# Building the "LOGISTIC REGRESSION" model
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,y_train)    # fit is only used for training data only, by using that we apply the results on testing data..


# In[136]:


lr.score(x_train,y_train)


# In[137]:


lr.score(x_test,y_test)


# In[138]:


# Building the "DECISION TREE" model
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)


# In[139]:


dt.score(x_train,y_train)


# In[140]:


dt.score(x_test,y_test)


# In[142]:


# Building the "NAVIE BAYES" model
from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb.fit(x_train,y_train)


# In[143]:


nb.score(x_train,y_train)


# In[144]:


nb.score(x_test,y_test)


# In[145]:


# Building the "RANDOM FOREST" model
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(x_train,y_train)


# In[146]:


rf.score(x_train,y_train)


# In[147]:


rf.score(x_test,y_test)


# In[148]:


# Checking the observations and variables are distributed correctly..
x_train.shape


# In[149]:


y_train.shape


# In[150]:


x_test.shape


# In[151]:


y_test.shape

