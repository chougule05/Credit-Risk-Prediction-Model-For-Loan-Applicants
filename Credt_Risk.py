#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import libraries
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.style as style
import pylab
import seaborn as sns
import itertools
import os


# In[2]:


import warnings
warnings.simplefilter('ignore')


# In[3]:


os.chdir(r'D:\CDAC\Statistics & R\MODULE_END') #Change the directory


# In[4]:


# Import Data
df_app = pd.read_csv('application_data.csv')
df_prev = pd.read_csv('previous_application.csv')


# In[5]:


pd.set_option('display.max_columns', 500)
# will display max 500 columns
pd.set_option('display.width', 1000) 
# does not exceed 1000 characters in width
pd.set_option('display.expand_frame_repr', False)
# it restricts the DataFrame from wrapping


# In[6]:


df_prev.head()


# In[7]:


df_app.head()


# In[8]:


df_app.shape # to check number of rows and columns in data


# In[9]:


df_prev.shape # to check number rows and columns in data


# In[10]:


df_prev.info() # to check column infomation


# In[11]:


df_app.info(verbose=True) 
# information of data (verbose= true gives all columns details)


# In[12]:


df_app.describe()


# In[13]:


df_prev.describe()


# In[14]:


# Data Cleaning 
df_app.isnull().sum() # null values count in each row


# In[15]:


df_prev.isnull().sum() # null values count in each row


# In[16]:


df_app.dtypes


# In[17]:


# DATA CLEANING


# In[18]:


round(df_app.isnull().sum() / df_app.shape[0] * 100.00,2) 
#null counts/ num rows


# In[19]:


# There are many columns in applicationDF dataframe where missing value is more than 40%. Let's plot the columns vs missing value % with 40% being the cut-off marks


# In[20]:


null_df_app = pd.DataFrame((df_app.isnull().sum())*100/df_app.shape[0]).reset_index()
null_df_app.columns = ['Column Name', 'Null Values Percentage']
plt.bar(null_df_app['Column Name'], null_df_app['Null Values Percentage'])
plt.xticks(rotation=90,fontsize=2)
plt.axhline(40, ls='--',color='red')
plt.xlabel('Column Name')
plt.ylabel('Percentage')
plt.title('Percentage of Null Counts of Application Data')
plt.show()


# In[21]:


# From the  Bar plot we can see the columns in which percentage of null values 
# more than 40% are marked above the red line and the columns which have 
# less than 40 % null values below the red line


# In[22]:


# more than or equal to 40% empty rows in  columns
null_40_df_app =null_df_app.iloc[np.where(null_df_app['Null Values Percentage'] >= 40)]
null_40_df_app


# In[23]:


print('Number of Columns having more 40% of null values:',len(null_40_df_app))


# In[24]:


# Total of 49 columns are there which have more than 40% null values.Seems like most of the columns with high missing values are related to different area sizes on apartment owned/rented by the loan applicant


# In[25]:


round(df_prev.isnull().sum() / df_prev.shape[0] * 100.00,2) 
#null counts/ num rows


# In[26]:


# There are many columns in previousDF dataframe where missing value is more than 40%. Let's plot the columns vs missing value % with 40% being the cut-off marks


# In[27]:


null_df_prev = pd.DataFrame((df_prev.isnull().sum())*100/df_prev.shape[0]).reset_index()
null_df_prev.columns = ['Column Name', 'Null Values Percentage']
plt.bar(null_df_prev['Column Name'], null_df_prev['Null Values Percentage'])
plt.xticks(rotation=90,fontsize=6.5)
plt.axhline(40, ls='--',color='red')
plt.xlabel('Column Name')
plt.ylabel('Percentage')
plt.title('Percentage of Null Counts Previous Data')
plt.show()


# In[28]:


# From the  Bar plot we can see the columns in which percentage of null values 
# more than 40% are marked above the red line and the columns which have 
# less than 40 % null values below the red line


# In[29]:


# more than or equal to 40% empty rows in  columns
null_40_df_prev =null_df_prev.iloc[np.where(null_df_prev['Null Values Percentage'] >= 40)]
null_40_df_prev


# In[30]:


print('Number of Columns having more 40% of null values:',len(null_40_df_prev))


# In[31]:


# Total of 11 columns are there which have more than 40% null values. These columns can be deleted. Before deleting these columns, let's review if there are more columns which can be dropped or not


# In[32]:


# Analyze & Delete Unnecessary Columns in applicationDF


# In[33]:


df_app_check = df_app[["EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3","TARGET"]]
df_app_corr = df_app_check.corr()
ax = sns.heatmap(df_app_corr,
            xticklabels=df_app_corr.columns,
            yticklabels=df_app_corr.columns,
            annot = True,
            cmap ="RdYlGn")


# In[34]:


# Based on above heat map, it is clearly seen that EXT_SOURCE columns are not 
# corelated to TARGET column thus we can drop this columns.EXT_SOURCE_1 has 56% null values, 
# where as EXT_SOURCE_3 has close to 20% null values


# In[35]:


# create a list of columns that needs to be dropped including the 
# columns with > 50% null values
unwanted_app=null_40_df_app['Column Name'].tolist()+['EXT_SOURCE_2','EXT_SOURCE_3']
len(unwanted_app)


# In[36]:


# Checking the relevance of Flag_Document and 
# whether it has any relation with loan repayment status
col_Doc = [ 'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3','FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6','FLAG_DOCUMENT_7', 
           'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9','FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12','FLAG_DOCUMENT_13',
           'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15','FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18',
           'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21']
df_app_flag = df_app[col_Doc+['TARGET']]

length = len(col_Doc)

df_app_flag["TARGET"] = df_app_flag["TARGET"].replace({1:"Defaulter",0:"Repayer"})

fig = plt.figure(figsize=(21,24))
for i,j in itertools.zip_longest(col_Doc,range(length)):
    
    plt.subplot(5,4,j+1)
    ax = sns.countplot(df_app_flag[i],hue=df_app_flag["TARGET"],palette=["r","g"])
    plt.yticks(fontsize=8)
    plt.xlabel("")
    plt.ylabel("")
    plt.title(i)


# In[37]:


# The above graph shows that in most of the loan application cases, 
# clients who applied for loans has not submitted FLAG_DOCUMENT_X 
# except FLAG_DOCUMENT_3. Thus, Except for FLAG_DOCUMENT_3, 
# we can delete rest of the columns. Data shows if borrower has submitted
# FLAG_DOCUMENT_3 then there is a less chance of defaulting the loan.


# In[38]:


# Including the flag documents for dropping the Document columns
col_Doc.remove('FLAG_DOCUMENT_3') 
unwanted_app = unwanted_app + col_Doc
len(unwanted_app)


# In[39]:


# checking is there is any correlation between mobile phone, work phone etc, email, Family members and Region rating
contact_col = ['FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE',
       'FLAG_PHONE', 'FLAG_EMAIL','TARGET']
Contact_corr = df_app[contact_col].corr()
fig = plt.figure(figsize=(8,8))
ax = sns.heatmap(Contact_corr,
            xticklabels=Contact_corr.columns,
            yticklabels=Contact_corr.columns,
            annot = True,
            cmap ="RdYlGn",
            linewidth=1)


# In[40]:


# There is no correlation between flags of mobile phone, email etc 
#with loan repayment; thus these columns can be deleted


# In[41]:


# including the 6 FLAG columns to be deleted
contact_col.remove('TARGET') 
unwanted_app = unwanted_app + contact_col
len(unwanted_app)


# In[42]:


# Total 76 columns can be deleted from applicationDF


# In[43]:


# Dropping the unnecessary columns from application data
df_app.drop(labels=unwanted_app,axis=1,inplace=True)


# In[44]:


# Inspecting the dataframe after removal of unnecessary columns
df_app.shape


# In[45]:


# inspecting the column types after removal of unnecessary columns
df_app.info()


# In[46]:


# After deleting unnecessary columns, there are 46 columns remaining in applicationDF


# In[47]:


# Getting the 4 columns which has more than 50% unknown
unwanted_prev = null_40_df_prev["Column Name"].tolist()
unwanted_prev


# In[48]:


# Listing down columns which are not needed
Unnecessary_previous = ['WEEKDAY_APPR_PROCESS_START','HOUR_APPR_PROCESS_START',
                        'FLAG_LAST_APPL_PER_CONTRACT','NFLAG_LAST_APPL_IN_DAY']


# In[49]:


unwanted_prev = unwanted_prev + Unnecessary_previous
len(unwanted_prev)


# In[50]:


# Total 15 columns can be deleted from previous data


# In[51]:


# Dropping the unnecessary columns from previous
df_prev.drop(labels=unwanted_prev,axis=1,inplace=True)
# Inspecting the dataframe after removal of unnecessary columns
df_prev.shape


# In[52]:


# inspecting the column types after after removal of unnecessary columns
df_prev.info()


# In[53]:


# After deleting unnecessary columns, 
# there are 22 columns remaining in previous data


# In[54]:


# Standardize Values

# Strategy for applicationDF:
# Convert DAYS_DECISION,DAYS_EMPLOYED, DAYS_REGISTRATION,DAYS_ID_PUBLISH from negative to positive as days cannot be negative.
# Convert DAYS_BIRTH from negative to positive values and calculate age and create categorical bins columns
# Categorize the amount variables into bins
# Convert region rating column and few other columns to categorical


# In[55]:


# Converting Negative days to positive days

date_col = ['DAYS_BIRTH','DAYS_EMPLOYED','DAYS_REGISTRATION','DAYS_ID_PUBLISH']

for col in date_col:
    df_app[col] = abs(df_app[col])


# In[56]:


# Binning Numerical Columns to create a categorical column

# Creating bins for income amount
df_app['AMT_INCOME_TOTAL']=df_app['AMT_INCOME_TOTAL']/100000

bins = [0,1,2,3,4,5,6,7,8,9,10,11]
slot = ['0-100K','100K-200K', '200k-300k','300k-400k','400k-500k','500k-600k','600k-700k','700k-800k','800k-900k','900k-1M', '1M Above']

df_app['AMT_INCOME_RANGE']=pd.cut(df_app['AMT_INCOME_TOTAL'],bins,labels=slot)


# In[57]:


#checking the binning of data and % of data in each category

df_app['AMT_INCOME_RANGE'].value_counts(normalize=True)*100


# In[58]:


# More than 50% loan applicants have income amount in the range of 100K-200K.
# Almost 92% loan applicants have income less than 300K


# In[59]:


# Creating bins for Credit amount
df_app['AMT_CREDIT']=df_app['AMT_CREDIT']/100000

bins = [0,1,2,3,4,5,6,7,8,9,10,100]
slots = ['0-100K','100K-200K', '200k-300k','300k-400k','400k-500k','500k-600k','600k-700k','700k-800k',
       '800k-900k','900k-1M', '1M Above']

df_app['AMT_CREDIT_RANGE']=pd.cut(df_app['AMT_CREDIT'],bins=bins,labels=slots)


# In[60]:


#checking the binning of data and % of data in each category
df_app['AMT_CREDIT_RANGE'].value_counts(normalize=True)*100


# In[61]:


# More Than 16% loan applicants have taken loan which amounts to more than 1M.


# In[62]:


# Creating bins for Age
df_app['AGE'] = df_app['DAYS_BIRTH'] // 365
bins = [0,20,30,40,50,100]
slots = ['0-20','20-30','30-40','40-50','50 above']

df_app['AGE_GROUP']=pd.cut(df_app['AGE'],bins=bins,labels=slots)


# In[63]:


#checking the binning of data and % of data in each category
df_app['AGE_GROUP'].value_counts(normalize=True)*100


# In[64]:


# 31% loan applicants have age above 50 years.
# More than 55% of loan applicants have age over 40 years.


# In[65]:


# Creating bins for Employement Time
df_app['YEARS_EMPLOYED'] = df_app['DAYS_EMPLOYED'] // 365
bins = [0,5,10,20,30,40,50,60,150]
slots = ['0-5','5-10','10-20','20-30','30-40','40-50','50-60','60 above']

df_app['EMPLOYMENT_YEAR']=pd.cut(df_app['YEARS_EMPLOYED'],bins=bins,labels=slots)


# In[66]:


#checking the binning of data and % of data in each category
df_app['EMPLOYMENT_YEAR'].value_counts(normalize=True)*100


# In[67]:


# More than 55% of the loan applicants have work experience within 0-5 years and almost 80% of them have less than 10 years of work experience


# In[68]:


#Checking the number of unique values each column possess to identify categorical columns
df_app.nunique().sort_values()


# In[69]:


# Data Type Conversion


# In[70]:


# inspecting the column types if they are in correct data type using the above result.
df_app.info()


# In[71]:


# Numeric columns are already in int64 and float64 format.
# Hence proceeding with other columns.


# In[72]:


#Conversion of Object and Numerical columns to Categorical Columns
categorical_columns = ['NAME_CONTRACT_TYPE','CODE_GENDER','NAME_TYPE_SUITE','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE',
                       'NAME_FAMILY_STATUS','NAME_HOUSING_TYPE','OCCUPATION_TYPE','WEEKDAY_APPR_PROCESS_START',
                       'ORGANIZATION_TYPE','FLAG_OWN_CAR','FLAG_OWN_REALTY','LIVE_CITY_NOT_WORK_CITY',
                       'REG_CITY_NOT_LIVE_CITY','REG_CITY_NOT_WORK_CITY','REG_REGION_NOT_WORK_REGION',
                       'LIVE_REGION_NOT_WORK_REGION','REGION_RATING_CLIENT','WEEKDAY_APPR_PROCESS_START',
                       'REGION_RATING_CLIENT_W_CITY'
                      ]

for col in categorical_columns:
    df_app[col] =pd.Categorical(df_app[col])


# In[73]:


# inspecting the column types if the above conversion is reflected
df_app.info()


# In[74]:


#  Standardize Values for previousDF


# Strategy for previous Data:

# Convert DAYS_DECISION from negative to positive values and 
# create categorical bins columns.
# Convert loan purpose and few other columns to categorical.


# In[75]:


#Checking the number of unique values each column possess to 
# identify categorical columns
df_prev.nunique().sort_values() 


# In[76]:


# inspecting the column types 
df_prev.info()


# In[77]:


#Converting negative days to positive days 
df_prev['DAYS_DECISION'] = abs(df_prev['DAYS_DECISION'])


# In[78]:


#age group calculation e.g. 388 will be grouped as 300-400
df_prev['DAYS_DECISION_GROUP'] = (df_prev['DAYS_DECISION']-                                    
(df_prev['DAYS_DECISION'] % 400)).astype(str)+'-'+ ((df_prev['DAYS_DECISION'] - (df_prev['DAYS_DECISION'] % 400)) + 
(df_prev['DAYS_DECISION'] % 400) + (400 - (df_prev['DAYS_DECISION'] % 400))).astype(str)


# In[79]:


df_prev['DAYS_DECISION_GROUP'].value_counts(normalize=True)*100


# In[80]:


# Almost 37% loan applicatants have applied for a new loan within 0-400 days
# of previous loan decision


# In[81]:


#Converting Categorical columns from Object to categorical 
Catgorical_col_p = ['NAME_CASH_LOAN_PURPOSE','NAME_CONTRACT_STATUS','NAME_PAYMENT_TYPE',
                    'CODE_REJECT_REASON','NAME_CLIENT_TYPE','NAME_GOODS_CATEGORY','NAME_PORTFOLIO',
                   'NAME_PRODUCT_TYPE','CHANNEL_TYPE','NAME_SELLER_INDUSTRY','NAME_YIELD_GROUP','PRODUCT_COMBINATION',
                    'NAME_CONTRACT_TYPE','DAYS_DECISION_GROUP']
for col in Catgorical_col_p:
    df_prev[col] =pd.Categorical(df_prev[col])


# In[82]:


# inspecting the column types after conversion
df_prev.info()


# In[83]:


# Null Value Data Imputation

# Strategy for application data:
# To impute null values in categorical variables which has lower null percentage, mode() is used to impute the most frequent items.
# To impute null values in categorical variables which has higher null percentage, a new category is created.
# To impute null values in numerical variables which has lower null percentage, median() is used as
# There are no outliers in the columns
# Mean returned decimal values and median returned whole numbers and the columns were number of requests


# In[84]:


# checking the null value % of each column in applicationDF dataframe
round(df_app.isnull().sum() / df_app.shape[0] * 100.00,2)


# In[85]:


# Impute categorical variable 'NAME_TYPE_SUITE' which has lower null percentage
# (0.42%) with the most frequent category using mode()[0]:


# In[86]:


df_app['NAME_TYPE_SUITE'].describe()


# In[87]:


df_app['NAME_TYPE_SUITE'].fillna((df_app['NAME_TYPE_SUITE'].mode()[0]),inplace = True)


# In[88]:


# Impute categorical variable 'OCCUPATION_TYPE' which has higher null 
# percentage(31.35%) with a new category as assigning to any existing 
# category might influence the analysis:


# In[89]:


df_app['OCCUPATION_TYPE'] = df_app['OCCUPATION_TYPE'].cat.add_categories('Unknown')
df_app['OCCUPATION_TYPE'].fillna('Unknown', inplace =True)


# In[90]:


# Impute numerical variables with the median as there are no outliers 
# that can be seen from results of describe() and mean()
# returns decimal values and these columns represent number of
# enquiries made which cannot be decimal:


# In[91]:


df_app[['AMT_REQ_CREDIT_BUREAU_HOUR','AMT_REQ_CREDIT_BUREAU_DAY',
               'AMT_REQ_CREDIT_BUREAU_WEEK','AMT_REQ_CREDIT_BUREAU_MON',
               'AMT_REQ_CREDIT_BUREAU_QRT','AMT_REQ_CREDIT_BUREAU_YEAR']].describe()


# In[92]:


amount = ['AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY','AMT_REQ_CREDIT_BUREAU_WEEK','AMT_REQ_CREDIT_BUREAU_MON',
         'AMT_REQ_CREDIT_BUREAU_QRT','AMT_REQ_CREDIT_BUREAU_YEAR']

for col in amount:
    df_app[col].fillna(df_app[col].median(),inplace = True)


# In[93]:


# checking the null value % of each column in applicationDF dataframe
round(df_app.isnull().sum() / df_app.shape[0] * 100.00,2)


# In[94]:


# checking the null value % of each column in previousDF dataframe
round(df_prev.isnull().sum() / df_prev.shape[0] * 100.00,2)


# In[95]:


plt.figure(figsize=(6,6))
sns.kdeplot(df_prev['AMT_ANNUITY'])
plt.show()


# In[96]:


# There is a single peak at the left side of the distribution and it indicates the presence of outliers and hence imputing with mean would
# not be the right approach and hence imputing with median.


# In[97]:


df_prev['AMT_ANNUITY'].fillna(df_prev['AMT_ANNUITY'].median(),inplace = True)


# In[98]:


plt.figure(figsize=(6,6))
sns.kdeplot(df_prev['AMT_GOODS_PRICE'][pd.notnull(df_prev['AMT_GOODS_PRICE'])])
plt.show()


# In[99]:


# As it has many peaks and skewed to one side so mean and median 
# will not be right approach, hence imputing with mode

df_prev['AMT_GOODS_PRICE'].fillna(df_prev['AMT_GOODS_PRICE'].mode()[0], inplace=True)


# In[100]:


# Impute CNT_PAYMENT with 0 as the NAME_CONTRACT_STATUS for these 
# indicate that most of these loans were not started:

df_prev.loc[df_prev['CNT_PAYMENT'].isnull(),'NAME_CONTRACT_STATUS'].value_counts()


# In[101]:


df_prev['CNT_PAYMENT'].fillna(0,inplace = True)


# In[102]:


# checking the null value % of each column in previousDF dataframe
round(df_prev.isnull().sum() / df_prev.shape[0] * 100.00,2)


# In[103]:


# We still have few null values in the PRODUCT_COMBINATION column.
# We can ignore as this percentage is very less.


# In[104]:


# Identifying the outliers


# Finding outlier information in application Data


# In[105]:


plt.figure(figsize=(22,10))

app_outlier_col_1 = ['AMT_ANNUITY','AMT_INCOME_TOTAL','AMT_CREDIT','AMT_GOODS_PRICE','DAYS_EMPLOYED']
app_outlier_col_2 = ['CNT_CHILDREN','DAYS_BIRTH']
for i in enumerate(app_outlier_col_1):
    plt.subplot(2,4,i[0]+1)
    sns.boxplot(y=df_app[i[1]])
    plt.title(i[1])
    plt.ylabel("")

for i in enumerate(app_outlier_col_2):
    plt.subplot(2,4,i[0]+6)
    sns.boxplot(y=df_app[i[1]])
    plt.title(i[1])
    plt.ylabel("")


# In[106]:


# It can be seen that in current application data

    # AMT_ANNUITY, AMT_CREDIT, AMT_GOODS_PRICE,CNT_CHILDREN have some number of outliers.
    # AMT_INCOME_TOTAL has huge number of outliers which indicate that few of the 
        # loan applicants have high income when compared to the others.
    # DAYS_BIRTH has no outliers which means the data available is reliable.
    # DAYS_EMPLOYED has outlier values around 350000(days) which is around 
        # 958 years which is impossible and hence this has to be incorrect entry.


# In[107]:


# Finding outlier information in previous Data

plt.figure(figsize=(22,8))

prev_outlier_col_1 = ['AMT_ANNUITY','AMT_APPLICATION','AMT_CREDIT','AMT_GOODS_PRICE','SELLERPLACE_AREA']
prev_outlier_col_2 = ['SK_ID_CURR','DAYS_DECISION','CNT_PAYMENT']
for i in enumerate(prev_outlier_col_1):
    plt.subplot(2,4,i[0]+1)
    sns.boxplot(y=df_prev[i[1]])
    plt.title(i[1])
    plt.ylabel("")

for i in enumerate(prev_outlier_col_2):
    plt.subplot(2,4,i[0]+6)
    sns.boxplot(y=df_prev[i[1]])
    plt.title(i[1])
    plt.ylabel("") 


# In[108]:


# It can be seen that in previous application data

    # AMT_ANNUITY, AMT_APPLICATION, AMT_CREDIT, AMT_GOODS_PRICE, SELLERPLACE_AREA have huge number of outliers.
    # CNT_PAYMENT has few outlier values.
    # SK_ID_CURR is an ID column and hence no outliers.
    # DAYS_DECISION has little number of outliers indicating that these previous applications decisions were taken long back.


# In[109]:


# Data Analysis


# Strategy:

# The data analysis flow has been planned in following way :

# Imbalance in Data
# Categorical Data Analysis
    # Categorical segmented Univariate Analysis
    # Categorical Bi/Multivariate analysis
# Numeric Data Analysis
    # Bi-furcation of databased based on TARGET data
    # Correlation Matrix
    # Numerical segmented Univariate Analysis
    # Numerical Bi/Multivariate analysis


# In[110]:


# 5.1 Imbalance Analysis

Imbalance = df_app["TARGET"].value_counts().reset_index()

plt.figure(figsize=(10,4))
x= ['Repayer','Defaulter']
sns.barplot(x,"TARGET",data = Imbalance,palette= ['g','r'])
plt.xlabel("Loan Repayment Status")
plt.ylabel("Count of Repayers & Defaulters")
plt.title("Imbalance Plotting")
plt.show()


# In[111]:


count_0 = Imbalance.iloc[0]["TARGET"]
count_1 = Imbalance.iloc[1]["TARGET"]
count_0_perc = round(count_0/(count_0+count_1)*100,2)
count_1_perc = round(count_1/(count_0+count_1)*100,2)

print('Ratios of imbalance in percentage with respect to Repayer and Defaulter datas are: %.2f and %.2f'%(count_0_perc,count_1_perc))
print('Ratios of imbalance in relative with respect to Repayer and Defaulter datas is %.2f : 1 (approx)'%(count_0/count_1))


# In[112]:


# Plotting Functions

# function for plotting repetitive countplots in univariate categorical analysis on applicationDF
# This function will create two subplots: 
# 1. Count plot of categorical column w.r.t TARGET; 
# 2. Percentage of defaulters within column

def univariate_categorical(feature,ylog=False,label_rotation=False,horizontal_layout=True):
    temp = df_app[feature].value_counts()
    df1 = pd.DataFrame({feature: temp.index,'Number of contracts': temp.values})

    # Calculate the percentage of target=1 per category value
    cat_perc = df_app[[feature, 'TARGET']].groupby([feature],as_index=False).mean()
    cat_perc["TARGET"] = cat_perc["TARGET"]*100
    cat_perc.sort_values(by='TARGET', ascending=False, inplace=True)
    
    if(horizontal_layout):
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,6))
    else:
        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(20,24))
        
    # 1. Subplot 1: Count plot of categorical column
    # sns.set_palette("Set2")
    s = sns.countplot(ax=ax1, 
                    x = feature, 
                    data=df_app,
                    hue ="TARGET",
                    order=cat_perc[feature],
                    palette=['g','r'])
    
    # Define common styling
    ax1.set_title(feature, fontdict={'fontsize' : 10, 'fontweight' : 3, 'color' : 'Blue'}) 
    ax1.legend(['Repayer','Defaulter'])
    
    # If the plot is not readable, use the log scale.
    if ylog:
        ax1.set_yscale('log')
        ax1.set_ylabel("Count (log)",fontdict={'fontsize' : 10, 'fontweight' : 3, 'color' : 'Blue'})   
    
    
    if(label_rotation):
        s.set_xticklabels(s.get_xticklabels(),rotation=90)
    
    # 2. Subplot 2: Percentage of defaulters within the categorical column
    s = sns.barplot(ax=ax2, 
                    x = feature, 
                    y='TARGET', 
                    order=cat_perc[feature], 
                    data=cat_perc,
                    palette='Set2')
    
    if(label_rotation):
        s.set_xticklabels(s.get_xticklabels(),rotation=90)
    plt.ylabel('Percent of Defaulters [%]', fontsize=10)
    plt.tick_params(axis='both', which='major', labelsize=10)
    ax2.set_title(feature + " Defaulter %", fontdict={'fontsize' : 15, 'fontweight' : 5, 'color' : 'Blue'}) 

    plt.show();


# In[113]:


# function for plotting repetitive countplots in bivariate categorical analysis

def bivariate_bar(x,y,df,hue,figsize):
    
    plt.figure(figsize=figsize)
    sns.barplot(x=x,
                  y=y,
                  data=df, 
                  hue=hue, 
                  palette =['g','r'])     
        
    # Defining aesthetics of Labels and Title of the plot using style dictionaries
    plt.xlabel(x,fontdict={'fontsize' : 10, 'fontweight' : 3, 'color' : 'Blue'})    
    plt.ylabel(y,fontdict={'fontsize' : 10, 'fontweight' : 3, 'color' : 'Blue'})    
    plt.title(col, fontdict={'fontsize' : 15, 'fontweight' : 5, 'color' : 'Blue'}) 
    plt.xticks(rotation=90, ha='right')
    plt.legend(labels = ['Repayer','Defaulter'])
    plt.show()


# In[114]:


# function for plotting repetitive rel plots in bivaritae numerical analysis on application data

def bivariate_rel(x,y,data, hue, kind, palette, legend,figsize):
    
    plt.figure(figsize=figsize)
    sns.relplot(x=x, 
                y=y, 
                data=df_app, 
                hue="TARGET",
                kind=kind,
                palette = ['g','r'],
                legend = False)
    plt.legend(['Repayer','Defaulter'])
    plt.xticks(rotation=90, ha='right')
    plt.show()


# In[115]:


#function for plotting repetitive countplots in univariate categorical analysis on the merged df

def univariate_merged(col,df,hue,palette,ylog,figsize):
    plt.figure(figsize=figsize)
    ax=sns.countplot(x=col, 
                  data=df,
                  hue= hue,
                  palette= palette,
                  order=df[col].value_counts().index)
    

    if ylog:
        plt.yscale('log')
        plt.ylabel("Count (log)",fontdict={'fontsize' : 10, 'fontweight' : 3, 'color' : 'Blue'})     
    else:
        plt.ylabel("Count",fontdict={'fontsize' : 10, 'fontweight' : 3, 'color' : 'Blue'})       

    plt.title(col , fontdict={'fontsize' : 15, 'fontweight' : 5, 'color' : 'Blue'}) 
    plt.legend(loc = "upper right")
    plt.xticks(rotation=90, ha='right')
    
    plt.show()


# In[116]:


# Function to plot point plots on merged dataframe

def merged_pointplot(x,y):
    plt.figure(figsize=(8,4))
    sns.pointplot(x=x, 
                  y=y, 
                  hue="TARGET", 
                  data=loan_process_df,
                  palette =['g','r'])
   # plt.legend(['Repayer','Defaulter'])


# In[117]:


# Categorical Variables Analysis


# In[118]:


# Segmented Univariate Analysis


# Checking the contract type based on loan repayment status


univariate_categorical('NAME_CONTRACT_TYPE',True)


# In[119]:


# Contract type: Revolving loans are just a small fraction (10%) from 
# the total number of loans; in the same time, a larger amount of Revolving loans, 
# comparing with their frequency, are not repaid.


# In[120]:


# Checking the type of Gender on loan repayment status
univariate_categorical('CODE_GENDER')


# In[121]:


# The number of female clients is almost double the number of male clients.
# Based on the percentage of defaulted credits, males have a higher chance of not returning their loans (~10%),
# comparing with women (~7%)


# In[122]:


# Checking if owning a car is related to loan repayment status
univariate_categorical('FLAG_OWN_CAR')


# In[123]:


# Clients who own a car are half in number of the clients who dont own a car.
# But based on the percentage of deault, there is no correlation between owning a car and loan repayment
# as in both cases the default percentage is almost same.


# In[124]:


# Checking if owning a realty is related to loan repayment status
univariate_categorical('FLAG_OWN_REALTY')


# In[125]:


# The clients who own real estate are more than double of the ones that don't own.
# But the defaulting rate of both categories are around the same (~8%). 
# Thus there is no correlation between owning a reality and defaulting the loan.


# In[126]:


# Analyzing Housing Type based on loan repayment status
univariate_categorical("NAME_HOUSING_TYPE",True,True,True)


# In[127]:


# Majority of people live in House/apartment
# People living in office apartments have lowest default rate
# People living with parents (~11.5%) and living in rented apartments(>12%) have higher probability of defaulting


# In[128]:


# Analyzing Family status based on loan repayment status
univariate_categorical("NAME_FAMILY_STATUS",False,True,True)


# In[129]:


# Most of the people who have taken loan are married, followed by Single/not married and civil marriage
# In terms of percentage of not repayment of loan, Civil marriage has the highest percent of not repayment (10%), with Widow the lowest (exception being Unknown).


# In[130]:


# Analyzing Education Type based on loan repayment status
univariate_categorical("NAME_EDUCATION_TYPE",True,True,True)


# In[131]:


# Majority of the clients have Secondary / secondary special education, followed by clients with Higher education.
# Only a very small number having an academic degree
# The Lower secondary category, although rare, have the largest rate of not returning the loan (11%). 
# The people with Academic degree have less than 2% defaulting rate.


# In[132]:


# Analyzing Income Type based on loan repayment status
univariate_categorical("NAME_INCOME_TYPE",True,True,False)


# In[133]:


# Most of applicants for loans have income type as Working, followed by Commercial associate, Pensioner and State servant.
# The applicants with the type of income Maternity leave have almost 40% ratio of not returning loans, followed by Unemployed (37%). The rest of types of incomes are under the average of 10% for not returning loans.
# Student and Businessmen, though less in numbers do not have any default record. Thus these two category are safest for providing loan.


# In[134]:


# Analyzing Region rating where applicant lives based on loan repayment status
univariate_categorical("REGION_RATING_CLIENT",False,False,True)


# In[135]:


# Most of the applicants are living in Region_Rating 2 place.
# Region Rating 3 has the highest default rate (11%)
# Applicant living in Region_Rating 1 has the lowest probability of defaulting, thus safer for approving loans


# In[136]:


# Analyzing Occupation Type where applicant lives based on loan repayment status
univariate_categorical("OCCUPATION_TYPE",False,True,False)


# In[137]:


# Most of the loans are taken by Laborers, followed by Sales staff. IT staff take the lowest amount of loans.
# The category with highest percent of not repaid loans are Low-skill Laborers (above 17%), followed by Drivers and Waiters/barmen staff, Security staff, Laborers and Cooking staff.


# In[138]:


# Checking Loan repayment status based on Organization type
univariate_categorical("ORGANIZATION_TYPE",True,True,False)


# In[139]:


# Organizations with highest percent of loans not repaid are Transport: type 3 (16%), Industry: type 13 (13.5%), Industry: type 8 (12.5%) and Restaurant (less than 12%). Self employed people have relative high defaulting rate, and thus should be avoided to be approved for loan or provide loan with higher interest rate to mitigate the risk of defaulting.
# Most of the people application for loan are from Business Entity Type 3
# For a very high number of applications, Organization type information is unavailable(XNA)
# It can be seen that following category of organization type has lesser defaulters thus safer for providing loans:
# Trade Type 4 and 5
# Industry type 8


# In[140]:


# Analyzing Flag_Doc_3 submission status based on loan repayment status
univariate_categorical("FLAG_DOCUMENT_3",False,False,True)


# In[141]:


# There is no significant correlation between repayers and defaulters in terms of submitting document 3 as we see even if applicants have submitted the document, they have defaulted a slightly more (~9%) than who have not submitted the document (6%)


# In[142]:


# Analyzing Age Group based on loan repayment status
univariate_categorical("AGE_GROUP",False,False,True)


# In[143]:


# People in the age group range 20-40 have higher probability of defaulting
# People above age of 50 have low probability of defailting


# In[144]:


# Analyzing Employment_Year based on loan repayment status
univariate_categorical("EMPLOYMENT_YEAR",False,False,True)


# In[145]:


# Majority of the applicants have been employeed in between 0-5 years. The defaulting rating of this group is also the highest which is 10%
# With increase of employment year, defaulting rate is gradually decreasing with people having 40+ year experience having less than 1% default rate


# In[146]:


# Analyzing Amount_Credit based on loan repayment status
univariate_categorical("AMT_CREDIT_RANGE",False,False,False)


# In[147]:


# More than 80% of the loan provided are for amount less than 900,000
# People who get loan for 300-600k tend to default more than others.


# In[148]:


# Analyzing Amount_Income Range based on loan repayment status
univariate_categorical("AMT_INCOME_RANGE",False,False,False)


# In[149]:


# 90% of the applications have Income total less than 300,000
# Application with Income less than 300,000 has high probability of defaulting
# Applicant with Income more than 700,000 are less likely to default


# In[150]:


# Analyzing Number of children based on loan repayment status
univariate_categorical("CNT_CHILDREN",True)


# In[151]:


# Most of the applicants do not have children
# Very few clients have more than 3 children.
# Client who have more than 4 children has a very high default rate with child count 9 and 11 showing 100% default rate


# In[152]:


# Analyzing Number of family members based on loan repayment status
univariate_categorical("CNT_FAM_MEMBERS",True, False, False)


# In[153]:


# Family member follows the same trend as children where having more family members increases the risk of defaulting


# In[154]:


#  Categorical Bi/Multivariate Analysis


# In[155]:


df_app.groupby('NAME_INCOME_TYPE')['AMT_INCOME_TOTAL'].describe()


# In[156]:


# Income type vs Income Amount Range
bivariate_bar("NAME_INCOME_TYPE","AMT_INCOME_TOTAL",df_app,"TARGET",(18,10))


# In[157]:


# It can be seen that business man's income is the highest and the estimated range with default 95% confidence level seem to indicate that the income of a business man could be in the range of slightly close to 4 lakhs and slightly above 10 lakhs


# In[158]:


# Numeric Variables Analysis


# Bifurcating the application data dataframe based on Target value 0 and 1 for correlation and other analysis


# In[159]:


df_app.columns


# In[160]:


# Bifurcating the applicationDF dataframe based on Target value 0 and 1 for correlation and other analysis
cols_for_correlation = ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 
                        'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 
                        'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS',
                        'NAME_HOUSING_TYPE', 'REGION_POPULATION_RELATIVE', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 
                        'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'OCCUPATION_TYPE', 'CNT_FAM_MEMBERS', 'REGION_RATING_CLIENT',
                        'REGION_RATING_CLIENT_W_CITY', 'WEEKDAY_APPR_PROCESS_START', 'HOUR_APPR_PROCESS_START',
                        'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION', 'LIVE_REGION_NOT_WORK_REGION', 
                        'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY', 'ORGANIZATION_TYPE',
                        'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE', 'DAYS_LAST_PHONE_CHANGE', 'FLAG_DOCUMENT_3', 
                        'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY', 'AMT_REQ_CREDIT_BUREAU_WEEK',
                        'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR']

Repayer_df = df_app.loc[df_app['TARGET']==0, cols_for_correlation] # Repayers
Defaulter_df = df_app.loc[df_app['TARGET']==1, cols_for_correlation] # Defaulters


# In[161]:


# Correlation between numeric variable


# In[162]:


# Getting the top 10 correlation for the Repayers data
corr_repayer = Repayer_df.corr()
corr_repayer = corr_repayer.where(np.triu(np.ones(corr_repayer.shape),k=1).astype(np.bool))
corr_df_repayer = corr_repayer.unstack().reset_index()
corr_df_repayer.columns =['VAR1','VAR2','Correlation']
corr_df_repayer.dropna(subset = ["Correlation"], inplace = True)
corr_df_repayer["Correlation"]=corr_df_repayer["Correlation"].abs() 
corr_df_repayer.sort_values(by='Correlation', ascending=False, inplace=True) 
corr_df_repayer.head(10)


# In[163]:


fig = plt.figure(figsize=(12,12))
ax = sns.heatmap(Repayer_df.corr(), cmap="RdYlGn",annot=False,linewidth =1)


# In[164]:


# Correlating factors amongst repayers:
# Credit amount is highly correlated with
# amount of goods price
# loan annuity
# total income
# We can also see that repayers have high correlation in number of days employed.


# In[165]:


# Getting the top 10 correlation for the Defaulter data
corr_Defaulter = Defaulter_df.corr()
corr_Defaulter = corr_Defaulter.where(np.triu(np.ones(corr_Defaulter.shape),k=1).astype(np.bool))
corr_df_Defaulter = corr_Defaulter.unstack().reset_index()
corr_df_Defaulter.columns =['VAR1','VAR2','Correlation']
corr_df_Defaulter.dropna(subset = ["Correlation"], inplace = True)
corr_df_Defaulter["Correlation"]=corr_df_Defaulter["Correlation"].abs()
corr_df_Defaulter.sort_values(by='Correlation', ascending=False, inplace=True)
corr_df_Defaulter.head(10)


# In[166]:


fig = plt.figure(figsize=(12,12))
ax = sns.heatmap(Defaulter_df.corr(), cmap="RdYlGn",annot=False,linewidth =1)


# In[167]:


# Credit amount is highly correlated with amount of goods price which is same as repayers.
# But the loan annuity correlation with credit amount has slightly reduced in defaulters(0.75) when compared to repayers(0.77)
# We can also see that repayers have high correlation in number of days employed(0.62) when compared to defaulters(0.58).
# There is a severe drop in the correlation between total income of the client and the credit amount(0.038) amongst defaulters whereas it is 0.342 among repayers.
# Days_birth and number of children correlation has reduced to 0.259 in defaulters when compared to 0.337 in repayers.
# There is a slight increase in defaulted to observed count in social circle among defaulters(0.264) when compared to repayers(0.254)


# In[168]:


# Plotting the numerical columns related to amount as distribution plot to see density
amount = df_app[[ 'AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY', 'AMT_GOODS_PRICE']]

fig = plt.figure(figsize=(16,12))

for i in enumerate(amount):
    plt.subplot(2,2,i[0]+1)
    sns.distplot(Defaulter_df[i[1]], hist=False, color='r',label ="Defaulter")
    sns.distplot(Repayer_df[i[1]], hist=False, color='g', label ="Repayer")
    plt.title(i[1], fontdict={'fontsize' : 15, 'fontweight' : 5, 'color' : 'Blue'}) 
    
plt.legend()

plt.show() 


# In[169]:


# Most no of loans are given for goods price below 10 lakhs
# Most people pay annuity below 50000 for the credit loan
# Credit amount of the loan is mostly less then 10 lakhs
# The repayers and defaulters distribution overlap in all the plots and hence we cannot use any of these variables in isolation to make a decision


# In[170]:


# Numerical Bivariate Analysis

# Checking the relationship between Goods price and credit and comparing with loan repayment staus
bivariate_rel('AMT_GOODS_PRICE','AMT_CREDIT',df_app,"TARGET", "line", ['g','r'], False,(15,6))


# In[171]:


# When the credit amount goes beyond 3M, there is an increase in defaulters.


# In[172]:


# Plotting pairplot between amount variable to draw reference against loan repayment status
amount = df_app[[ 'AMT_INCOME_TOTAL','AMT_CREDIT',
                         'AMT_ANNUITY', 'AMT_GOODS_PRICE','TARGET']]
amount = amount[(amount["AMT_GOODS_PRICE"].notnull()) & (amount["AMT_ANNUITY"].notnull())]
ax= sns.pairplot(amount,hue="TARGET",palette=["g","r"])
ax.fig.legend(labels=['Defaulter','Repayer'])
plt.show()


# In[173]:


# When amt_annuity >15000 amt_goods_price> 3M, there is a lesser chance of defaulters
# AMT_CREDIT and AMT_GOODS_PRICE are highly correlated as based on the scatterplot where most of the data are consolidated in form of a line
# There are very less defaulters for AMT_CREDIT >3M
# Inferences related to distribution plot has been already mentioned in previous distplot graphs inferences section


# In[174]:


# merge both the dataframe on SK_ID_CURR with Inner Joins
loan_process_df = pd.merge(df_app, df_prev, how='inner', on='SK_ID_CURR')
loan_process_df.head()


# In[175]:


#Checking the details of the merged dataframe
loan_process_df.shape


# In[176]:


# Checking the element count of the dataframe
loan_process_df.size


# In[177]:


# checking the columns and column types of the dataframe
loan_process_df.info()


# In[178]:


# Checking merged dataframe numerical columns statistics
loan_process_df.describe()


# In[232]:


loan_process_df.isnull().sum()


# In[179]:


# Bifurcating the applicationDF dataframe based on Target value 0 and 1 for correlation and other analysis

L0 = loan_process_df[loan_process_df['TARGET']==0] # Repayers
L1 = loan_process_df[loan_process_df['TARGET']==1] # Defaulters


# In[180]:


univariate_merged("NAME_CASH_LOAN_PURPOSE",L0,"NAME_CONTRACT_STATUS",["#548235","#FF0000","#0070C0","#FFFF00"],True,(18,7))

univariate_merged("NAME_CASH_LOAN_PURPOSE",L1,"NAME_CONTRACT_STATUS",["#548235","#FF0000","#0070C0","#FFFF00"],True,(18,7))


# In[181]:


# Loan purpose has high number of unknown values (XAP, XNA)
# Loan taken for the purpose of Repairs seems to have highest default rate
# A very high number application have been rejected by bank or refused by client which has purpose as repair or other. This shows that purpose repair is taken as high risk by bank and either they are rejected or bank offers very high loan interest rate which is not feasible by the clients, thus they refuse the loan.


# In[182]:


# Checking the Contract Status based on loan repayment status and whether there is any business loss or financial loss
univariate_merged("NAME_CONTRACT_STATUS",loan_process_df,"TARGET",['g','r'],False,(12,8))
g = loan_process_df.groupby("NAME_CONTRACT_STATUS")["TARGET"]
df1 = pd.concat([g.value_counts(),round(g.value_counts(normalize=True).mul(100),2)],axis=1, keys=('Counts','Percentage'))
df1['Percentage'] = df1['Percentage'].astype(str) +"%" # adding percentage symbol in the results for understanding
print (df1)


# In[183]:


# 90% of the previously cancelled client have actually repayed the loan. Revisiting the interest rates would increase business opoortunity for these clients
# 88% of the clients who have been previously refused a loan has payed back the loan in current case.
# Refual reason should be recorded for further analysis as these clients would turn into potential repaying customer.


# In[184]:


# plotting the relationship between income total and contact status
merged_pointplot("NAME_CONTRACT_STATUS",'AMT_INCOME_TOTAL')


# In[185]:


# The point plot show that the people who have not used offer earlier have defaulted even when there average income is higher than others


# In[186]:


# plotting the relationship between people who defaulted in last 60 days being in client's social circle and contact status
merged_pointplot("NAME_CONTRACT_STATUS",'DEF_60_CNT_SOCIAL_CIRCLE')


# In[187]:


# Clients who have average of 0.13 or higher DEF_60_CNT_SOCIAL_CIRCLE score tend to default more and hence client's social circle has to be analysed before providing the loan.


# In[223]:


import scipy
from scipy import stats
from scipy.stats import t, norm, chi2,  binom, chi2_contingency, chisquare, linregress

import statsmodels
from statsmodels import stats
from statsmodels.stats import weightstats, proportion
from statsmodels.stats.proportion import proportions_ztest

import statsmodels.api as sm
from statsmodels.formula.api import ols

import statsmodels.stats.multicomp

import sklearn
from sklearn.model_selection import train_test_split


# In[206]:


loan_process_df.info()


# In[226]:


modl=loan_process_df[['TARGET','AMT_INCOME_TOTAL','AMT_GOODS_PRICE_x','AMT_CREDIT_x',
                     'AMT_ANNUITY_x']]


# In[230]:


corr_matrix = loan_process_df.corr()
print(corr_matrix["TARGET"].sort_values(ascending=False))


# In[233]:


loan_process_df.corr()


# In[ ]:


# loan_process_df.corr() will help us in calculating the correlation between the columns so as to understand how the columns are related to each other.


# In[240]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

features = ['DAYS_LAST_PHONE_CHANGE', 'DAYS_BIRTH', 'REGION_POPULATION_RELATIVE']
X = loan_process_df[features]
y = loan_process_df['TARGET']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))


# In[248]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, mean_squared_error, accuracy_score
from math import sqrt

features = ["DAYS_DECISION", "REG_CITY_NOT_WORK_CITY", "LIVE_CITY_NOT_WORK_CITY", "REG_CITY_NOT_LIVE_CITY",
"CNT_PAYMENT", "CNT_CHILDREN", "OBS_30_CNT_SOCIAL_CIRCLE", "OBS_60_CNT_SOCIAL_CIRCLE", "CNT_FAM_MEMBERS",
"DEF_30_CNT_SOCIAL_CIRCLE", "DEF_60_CNT_SOCIAL_CIRCLE", "SELLERPLACE_AREA", "AMT_REQ_CREDIT_BUREAU_YEAR",
"AMT_CREDIT_y", "AMT_INCOME_TOTAL", "AMT_REQ_CREDIT_BUREAU_MON", "AMT_CREDIT_x", "AMT_APPLICATION",
"AMT_GOODS_PRICE_x", "HOUR_APPR_PROCESS_START", "HOUR_APPR_PROCESS_START", "AMT_GOODS_PRICE_x",
"AMT_ANNUITY_y", "DAYS_EMPLOYED", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH", "REGION_POPULATION_RELATIVE",
"DAYS_BIRTH", "DAYS_LAST_PHONE_CHANGE"]

X = loan_process_df[features]
y = loan_process_df['TARGET']
model = RandomForestClassifier(random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred)
print(f'Test ROC-AUC score: {roc_auc}')
risk_threshold = 0.5
approved_loans = X_test[y_pred <= risk_threshold]


rmse1 = sqrt(mean_squared_error(y_test, y_pred))
print(f'Test RMSE score: {rmse1}')


# In[252]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

features = ['AMT_CREDIT_x', 'AMT_ANNUITY_x', 'CNT_PAYMENT', 'DAYS_DECISION']
X = loan_process_df[features]
y = loan_process_df['TARGET']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

rmse = sqrt(mean_squared_error(y_test, y_pred))
print(f'Test RMSE score: {rmse}')

roc_auc = roc_auc_score(y_test, y_pred)
print(f'Test ROC-AUC score: {roc_auc}')


# In[249]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Calculate the ROC-AUC score
roc_auc = roc_auc_score(y_test, y_pred)
print(f'Test ROC-AUC score: {roc_auc}')


# In[ ]:




