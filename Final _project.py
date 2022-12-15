#%%
# Loading packages for analysis
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import plotly.graph_objs as go 
import plotly.express as px
import plotly.io as pio
import re
from wordcloud import WordCloud,STOPWORDS
%matplotlib inline


# %%
#Database setup
jobs = pd.read_csv('../dataset/all_jobs.csv') # Importing dataset
jobs.rename(columns={'Job Title': 'job_title', 'Salary Estimate': 'Salary_Estimate','Job Description': 'Job_Description','Company Name': 'Company_Name','Type of ownership':'Type_ownership','Easy Apply':'Easy_apply'}, inplace= True)

# %%
jobs = jobs.drop(labels=['Unnamed: 0'],axis=1) # Removing unnecesary column 
jobs #exploring the data

#%%
jobs.shape
#%%
jobs.head(10)
#%%
jobs.tail(10)
#%%
jobs.info()
# %% Checking the summary statistics of the data
jobs.describe(include="all")

# %% Checking for duplicate rows
duplicate_rows=jobs[jobs.duplicated()]
print(duplicate_rows.shape)


# %%
##Data Cleaning
from mlxtend.frequent_patterns import apriori, association_rules
import copy

# %%
jobs = jobs.drop_duplicates(subset = ['Job_Description','job_title','Location'], keep = 'first') 
# %%
#Removing Capitals
jobs['Job_Description'] = jobs['Job_Description'].str.lower()

#Removing  all non-word charachters

regex = re.compile('[^a-zA-Z\']')

jobs['Job_Description'] = jobs['Job_Description'].apply(lambda x: regex.sub(' ', x))
# %%
#The Equal Opportunity tagline may skew our results, let's remove it
equal_emp = 'Kelly is an equal opportunity employer committed to employing a diverse workforce, including, but not limited to, minorities, females, individuals with disabilities, protected veterans, sexual orientation, gender identity. Equal Employment Opportunity is The Law.'
equal_emp = equal_emp.lower().split(' ')

jobs['Job_Description'] = jobs['Job_Description'].apply(lambda x: [item for item in x.split() if item.lower() not in equal_emp])

#and then re-join our Job Descriptions
jobs['Job_Description'] = jobs['Job_Description'].apply(lambda x: ' '.join(x))

#%% Replacing -1 with NA'S
jobs.replace(to_replace = -1 , value=np.nan,inplace=True)
jobs.replace(to_replace ='-1' , value=np.nan,inplace=True)
jobs.replace(to_replace =-1.0 , value=np.nan,inplace=True)

#%%
#Quantifying missing values
def FindingMissingValues(dataFrame):
    for col in dataFrame.columns:
        print('{0:.2f}% or {1} values are Missing in {2} Column'.format(dataFrame[col].isna().sum()/len(dataFrame)*100,dataFrame[col].isna().sum(),col),end='\n\n')

FindingMissingValues(jobs)

#%%[markdown]
#There are 72.04% of values missing in the competitors columns
#and 96.25% in the Easy Apply, therefore we are going to drop this columns

#%%

jobs.drop(['Easy_apply','Competitors'],1,inplace = True)

#%%
jobs['Salary_Estimate'].replace('', np.nan, inplace=True) # replacing empty cell for NA
#%%
jobs.dropna(subset=['Salary_Estimate'], inplace=True) # Removing empty rows

# %%
#Splitting information from Job domain and role
jobs['Job Domain'] = jobs['job_title'].apply(lambda x: re.search(r',.*',x).group().replace(',','') if(bool(re.search(r',.*',x))) else x )
jobs['Job Role'] = jobs['job_title'].apply(lambda x: re.search(r'.*,',x).group().replace(',','') if(bool(re.search(r',.*',x))) else x )
jobs.rename(columns = {'Job Domain':'Job_Domain'}, inplace = True)

#%%
jobs = jobs.assign(newCol=jobs['Salary_Estimate'].str.extract('(Per Hour)')) # Identifying per hour entries
#%%
jobs= jobs[jobs["newCol"].isnull()]  #dropping per hour rows


#%%
jobs.drop(['newCol'],1,inplace = True) # removing row

#%%
# removing employer est.
jobs['Salary_Estimate'] = jobs['Salary_Estimate'].map(lambda x:  x.rstrip('(Employer est.)')) # Removing employer estimate

#%% Adding min and max salary ranges
jobs['Min_Salary'] = 0
jobs['Max_Salary'] = 0

for x in range(len(jobs)):
    
    if(type(jobs.iloc[x,1])==float):
        jobs.iloc[x,15] = np.nan
        jobs.iloc[x,16] = np.nan
    else:
        cleanSal = jobs.iloc[x,1].replace('(Glassd','').strip().split('-') 

    if('K' in cleanSal[0]):
        jobs.iloc[x,15] = float(cleanSal[0].replace('$','').replace('K',''))  
        
    if('K' in cleanSal[1]):
        jobs.iloc[x,16]= float(cleanSal[1].replace('$','').replace('K',''))

#%% Cleaning for max number of employees
jobs['MaxEmpSize'] = 0

for x in range(len(jobs)):
    emp = jobs.iloc[x,7]
    
    try:
        if(type(emp)==float or emp == 'Unknown'): #type(np.nan)== float
            jobs.iloc[x,17] =  np.nan
        elif('+' in emp):
            jobs.iloc[x,17] = float(emp.replace('+','').replace('employees','').strip())
        elif('employees' in emp):
            jobs.iloc[x,17] = float(emp.replace('employees','').strip().split('to')[1])
    except(Exception)as e:
        print(e,emp)


#%%
skill_types= {}

skill_types['Statistics'] = ['matlab',
 'statistical',
 'models',
 'modeling',
 'statistics',
 'analytics',
 'forecasting',
 'predictive',
 'r',
 'R', 
 'pandas',
 'statistics',
 'statistical',
 'Julia']

skill_types['Machine Learning'] = ['datarobot',
 'tensorflow',
 'knime',
 'rapidminer',
 'mahout',
 'logicalglue',
 'nltk',
 'networkx',
 'rapidminer',
 'scikit',
 'pytorch',
 'keras',
 'caffe',
 'weka',
 'orange',
 'qubole',
 'ai',
 'nlp',
 'ml',
 'neuralnetworks',
 'deeplearning']


skill_types['Data Visualization'] = ['tableau',
 'powerpoint',
 'Qlik',
 'looker',
 'powerbi',
 'matplotlib',
 'tibco',
 'bokeh',
 'd3',
 'octave',
 'shiny',
 'microstrategy']


skill_types['Data Engineering'] = ['etl',
 'mining',
 'warehousing',
 'cloud',
 'sap',
 'salesforce',
 'openrefine',
 'redis',
 'sybase',
 'cassandra',
 'msaccess',
 'databasemanagement',
 'aws',
 'ibmcloud',
 'azure',
 'redshift',
 's3',
 'ec2',
 'rds',
 'bigquery',
 'googlecloudplatform',
 'googlecloudplatform',
 'hadoop',
 'hive',
 'kafka',
 'hbase',
 'mesos',
 'pig',
 'storm',
 'scala',
 'hdfs',
 'mapreduce',
 'kinesis',
 'flink']


skill_types['Software Engineer'] = ['java',
 'javascript',
 'c#',
 'c',
 'docker',
 'ansible',
 'jenkins',
 'nodejs',
 'angularjs',
 'css',
 'html',
 'terraform',
 'kubernetes',
 'lex',
 'perl',
 'cplusplus',
 'Python',
 'python']


skill_types['SQL'] = ['sql',
 'oracle',
 'mysql',
 'oraclenosql',
 'nosql',
 'postgresql',
 'plsql',
 'mongodb']




skill_types['Trait Skills'] = ['Learning',
 'TimeManagement',
 'AttentiontoDetail',
 'ProblemSolving',
 'criticalthinking']



skill_types['Social Skills']= ['teamwork',
 'team'
 'communication',
 'written',
 'verbal',
 'writing',
 'leadership',
 'interpersonal',
 'personalmotivation',
 'storytelling']

skill_types['Business'] = ['excel',
 'bi',
 'reporting',
 'reports',
 'dashboards',
 'dashboard',
 'businessintelligence'
 'business']

for k,v in skill_types.items():
    skill_types[k] = [skill.lower() for skill in skill_types.get(k)]
# %%
def refiner(desc):
    desc = desc.split()
    
    two_word = ''
    
    newskills = []
    
    for word in desc:
        two_word = two_word + word 
        for key,value in skill_types.items():
            if((word in value) or (two_word in value)):
                newskills.append(key)
                
        #check for the two worders, like 'businessintelligence'        
        two_word = word
                
    return list(set(newskills))
# %%
jobs['refined_skills'] = jobs['Job_Description'].apply(refiner)
# %%
#This is what our new column looks like
jobs['refined_skills']

# %%

def apriori_df(series, min_support):
    lisolis =[]
    series.apply(lambda x: lisolis.append(list(x)))
    
    from mlxtend.preprocessing import TransactionEncoder

    te = TransactionEncoder()
    te_ary = te.fit(lisolis).transform(lisolis)
    df = pd.DataFrame(te_ary, columns=te.columns_)


    from mlxtend.frequent_patterns import apriori

    freq_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    
    return freq_itemsets

# %%
frequent_itemsets = apriori_df(jobs['refined_skills'],.1)

# %%
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))


# %% Obtaining States
jobs['State_Location'] = jobs.Location.str[-2:] #Cleaning to add State to the data

#%%
# Separate 'City' & 'State' from job 'Location'
jobs['City'],jobs['State'] = jobs['Location'].str.split(', ',1).str
jobs['HQCity'],jobs['HQState'] = jobs['Headquarters'].str.split(', ',1).str

# Clean up duplicated city names in State's name
jobs['State']=jobs['State'].replace('Arapahoe, CO','CO')
jobs['State']=jobs['State'].replace('Los Angeles, CA','CA')
jobs['HQState']=jobs['HQState'].replace('NY (US), NY','NY')
# %%
jobs['Salary_Estimate'] = jobs['Salary_Estimate'].map(lambda x:  x.rstrip('(Glassdoor est.)')) # Removing glassdoor estimate
#%%
# For regression purpose salary: Est_Salary = (Min_Salary+Max_Salary)/2
jobs['Est_Salary']=(jobs['Min_Salary']+jobs['Max_Salary'])/2

#%% Removing Rate on company column
jobs['Company_Name'] = jobs['Company_Name'].apply(lambda x: re.sub(r'\n.*','',str(x)))

#%%
# Create a variable for how many years a firm has been founded
jobs['Years_Founded'] = 2022 - jobs['Founded']

#%% Revenue
jobs['MaxRevenue'] = 0

for x in range(len(jobs)):
    rev = jobs.iloc[x,12]
    
    if(rev == 'Unknown / Non-Applicable' or type(rev)==float):
        jobs.iloc[x,26] = np.nan
    elif(('million' in rev) and ('billion' not in rev)):
        maxRev = rev.replace('(USD)','').replace("million",'').replace('$','').strip().split('to')
        if('Less than' in maxRev[0]):
            jobs.iloc[x,26] = float(maxRev[0].replace('Less than','').strip())*100000000
        else:
            if(len(maxRev)==2):
                jobs.iloc[x,26] = float(maxRev[1])*100000000
            elif(len(maxRev)<2):
                jobs.iloc[x,26] = float(maxRev[0])*100000000
    elif(('billion'in rev)):
        maxRev = rev.replace('(USD)','').replace("billion",'').replace('$','').strip().split('to')
        if('+' in maxRev[0]):
            jobs.iloc[x,26] = float(maxRev[0].replace('+','').strip())*1000000000
        else:
            if(len(maxRev)==2):
                jobs.iloc[x,26] = float(maxRev[1])*1000000000
            elif(len(maxRev)<2):
                jobs.iloc[x,26] = float(maxRev[0])*1000000000
                
## Extracting Skills from Job Description:
# %%
#python
jobs['python'] = jobs['Job_Description'].apply(lambda x: 1 if 'python' in x.lower() else 0)
jobs.python.value_counts()

#spark 
jobs['spark'] = jobs['Job_Description'].apply(lambda x: 1 if 'spark' in x.lower() else 0)
jobs.spark.value_counts()

#aws 
jobs['aws'] = jobs['Job_Description'].apply(lambda x: 1 if 'aws' in x.lower() else 0)
jobs.aws.value_counts()

#excel
jobs['excel'] = jobs['Job_Description'].apply(lambda x: 1 if 'excel' in x.lower() else 0)
jobs.excel.value_counts()

#sql
jobs['sql'] = jobs['Job_Description'].apply(lambda x: 1 if 'sql' in x.lower() else 0)
jobs.sql.value_counts()

#sas
jobs['sas'] = jobs['Job_Description'].apply(lambda x: 1 if 'sas' in x.lower() else 0)
jobs.sas.value_counts()

#hadoop
jobs['hadoop'] = jobs['Job_Description'].apply(lambda x: 1 if 'hadoop' in x.lower() else 0)
jobs.hadoop.value_counts()

#tableau
jobs['tableau'] = jobs['Job_Description'].apply(lambda x: 1 if 'tableau' in x.lower() else 0)
jobs.tableau.value_counts()

#bi
jobs['bi'] = jobs['Job_Description'].apply(lambda x: 1 if 'power bi' in x.lower() else 0)
jobs.bi.value_counts()

############

#%%
jobs.drop(['Salary_Estimate','Job_Description'],1,inplace = True)

#%%
#Finding Null values 
jobs.isnull().sum()
#%%
sns.distplot(jobs.Rating)
#Replacing Rating null values with mean
jobs.Rating=jobs.Rating.fillna(jobs.Rating.mean())
#%%
#Replacing Headquaters,Industry,Sector null values with mode
jobs.Headquarters=jobs.Headquarters.fillna(jobs.Headquarters.mode()[0])
# %%
jobs.Industry=jobs.Industry.fillna(jobs.Industry.mode()[0])
jobs.Sector=jobs.Sector.fillna(jobs.Sector.mode()[0])
# %%
sns.distplot(jobs.Founded)
# %%
jobs["Founded"] = jobs["Founded"].fillna(jobs["Founded"].median())

# %%
sns.displot(jobs.MaxRevenue)
# %%
jobs["MaxRevenue"] = jobs["MaxRevenue"].fillna(jobs["MaxRevenue"].median())
# %%
sns.distplot(jobs.Years_Founded)
# %%
jobs["Years_Founded"] = jobs["Years_Founded"].fillna(jobs["Years_Founded"].median())

jobs.Type_ownership=jobs.Type_ownership.fillna(jobs.Type_ownership.mode()[0])
# %%
jobs.Size=jobs.Size.fillna(jobs.Size.mode()[0])
# %%
jobs.State=jobs.State.fillna(jobs.State.mode()[0])
# %%
jobs.HQCity=jobs.HQCity.fillna(jobs.HQCity.mode()[0])
# %%
jobs.HQState=jobs.HQState.fillna(jobs.HQState.mode()[0])
sns.distplot(jobs.MaxEmpSize)
# %%
jobs["MaxEmpSize"] = jobs["MaxEmpSize"].fillna(jobs["MaxEmpSize"].median())

#%%
#### Exploring the data with visualizations

#%% Min, max and avg salary distribution for data scientists

plt.figure(figsize=(13,5))
sns.set(style= 'white') #style==background
sns.distplot(jobs['Min_Salary'], color="r")
sns.distplot(jobs['Max_Salary'], color="g")
sns.distplot(jobs['Est_Salary'], color="b")

plt.xlabel("Salary ($'000)")
plt.legend({'Min_Salary':jobs['Min_Salary'],'Max_Salary':jobs['Max_Salary'],'Est_Salary':jobs['Est_Salary']})
plt.title("Distribution of Min, Max and Avg Salary",fontsize=19)
plt.xlim(0,210)
plt.xticks(np.arange(0, 210, step=10))
plt.tight_layout()
plt.show()

plt.savefig('min_max_sal.png', dpi=300)

#Printing the mean of min,max and avg salary
import statistics
mean_min_salary=statistics.mean(jobs['Min_Salary'])
print("Mean of minimum salary:",mean_min_salary)

mean_max_salary=statistics.mean(jobs['Max_Salary'])
print("Mean of maximum salary:",mean_max_salary)

mean_avg_salary=statistics.mean(jobs['Est_Salary'])
print("Mean of average salary:",mean_avg_salary)


#%% Salary/Hires by Companies
df_by_firm=jobs.groupby('Company_Name')['job_title'].count().reset_index().sort_values(
    'job_title',ascending=False).head(20).rename(columns={'job_title':'Jobs'})

Sal_by_firm = df_by_firm.merge(jobs,on='Company_Name',how='left')

sns.set(style="white")
f, (ax_bar, ax_point) = plt.subplots(ncols=2, sharey=True, gridspec_kw= {"width_ratios":(0.6,1)},figsize=(13,7))
sns.barplot(x='Jobs',y='Company_Name',data=Sal_by_firm,ax=ax_bar, palette='Accent').set(ylabel="")
sns.pointplot(x='Est_Salary',y='Company_Name',data=Sal_by_firm, join=False,ax=ax_point, palette='Accent').set(
    ylabel="",xlabel="Salary ($'000)")
plt.subplots_adjust(top=0.9)
plt.suptitle('Jobs and Estimated Salary offered by Companies', fontsize = 16)
plt.tight_layout()


#%% Salary/Hires by Industry
df_by_industry=jobs.groupby('Industry')['job_title'].count().reset_index().sort_values(
    'job_title',ascending=False).head(20).rename(columns={'job_title':'Hires'})

Sal_by_industry = df_by_industry.merge(jobs,on='Industry',how='left')

sns.set(style="white")
f, (ax_bar, ax_point) = plt.subplots(ncols=2, sharey=True, gridspec_kw= {"width_ratios":(0.6,1)},figsize=(13,7))
sns.barplot(x='Hires',y='Industry',data=Sal_by_industry,ax=ax_bar, palette='Accent').set(ylabel="")
sns.pointplot(x='Est_Salary',y='Industry',data=Sal_by_industry, join=False,ax=ax_point, palette='Accent').set(
    ylabel="",xlabel="Salary ($'000)")
plt.subplots_adjust(top=0.9)
plt.suptitle('Hiring and salary by industry', fontsize = 16)
plt.tight_layout()

#%% Salary/Hires by Sector
df_by_sector=jobs.groupby('Sector')['job_title'].count().reset_index().sort_values(
    'job_title',ascending=False).head(20).rename(columns={'job_title':'Hires'})

Sal_by_sector = df_by_sector.merge(jobs,on='Sector',how='left')

sns.set(style="white")
f, (ax_bar, ax_point) = plt.subplots(ncols=2, sharey=True, gridspec_kw= {"width_ratios":(0.6,1)},figsize=(13,7))
sns.barplot(x='Hires',y='Sector',data=Sal_by_sector,ax=ax_bar, palette='Accent').set(ylabel="")
sns.pointplot(x='Est_Salary',y='Sector',data=Sal_by_sector, join=False,ax=ax_point, palette='tab20c').set(
    ylabel="",xlabel="Salary ($'000)")
plt.subplots_adjust(top=0.9)
plt.suptitle('Hiring and salary by sector', fontsize = 16)
plt.tight_layout()

#%%
# Salary/ Hires by City
df_by_city=jobs.groupby('Location')['job_title'].count().reset_index().sort_values(
    'job_title',ascending=False).head(20).rename(columns={'job_title':'Hires'})
Sal_by_city = df_by_city.merge(jobs,on='Location',how='left')

sns.set(style="white")
f, (ax_bar, ax_point) = plt.subplots(ncols=2, sharey=True, gridspec_kw= {"width_ratios":(0.6,1)},figsize=(13,7))
sns.barplot(x='Hires',y='Location',data=Sal_by_city,ax=ax_bar, palette='Accent').set(ylabel="")
sns.pointplot(x='Est_Salary',y='Location',data=Sal_by_city, join=False,ax=ax_point, palette='Accent').set(
    ylabel="",xlabel="Salary ($'000)")
plt.subplots_adjust(top=0.9)
plt.suptitle('Hiring and salary by City', fontsize = 16)
plt.tight_layout()

#%%
# Salary/ Hires by State
df_by_state=jobs.groupby('State_Location')['job_title'].count().reset_index().sort_values(
    'job_title',ascending=False).head(20).rename(columns={'job_title':'Hires'})
Sal_by_state = df_by_state.merge(jobs,on='State_Location',how='left')

sns.set(style="white")
f, (ax_bar, ax_point) = plt.subplots(ncols=2, sharey=True, gridspec_kw= {"width_ratios":(0.6,1)},figsize=(13,7))
sns.barplot(x='Jobs',y='State_Location',data=Sal_by_state,ax=ax_bar, palette='Accent').set(ylabel="")
sns.pointplot(x='Est_Salary',y='State_Location',data=Sal_by_state, join=False,ax=ax_point, palette='Accent')
plt.subplots_adjust(top=0.9)
plt.suptitle('Jobs and salary by State', fontsize = 16)
plt.tight_layout()

#%% SMART Question Analysis

# %%
# Barplot for estimated salary by state
sns.set(rc={'figure.figsize':(14,6)})
state_barplot=sns.barplot(x='State_Location',y='Est_Salary',data=jobs,palette="Accent")
plt.xlabel('States')
plt.ylabel("Salary($'000)")
plt.xticks(rotation=90)
plt.show()

#%%
#Lineplot for Revenue vs Salary
sns.set(rc={'figure.figsize':(6,6)})
lineplot=sns.lineplot(x="Revenue", y="Est_Salary", data=jobs,sort= False)
lineplot.tick_params(axis='x', rotation=90)
plt.show()
jobs.columns
#%%
#Barplot for estimated salary by industry
sns.set(rc={'figure.figsize':(6,6)})
sns.barplot(x='Est_Salary',y='Industry',data=Sal_by_industry,palette="Accent").set(title='Salary Estimate by Industry',xlabel="Salary ($'000)")

#%%
#Barplot for estimated salary by sector
sns.set(rc={'figure.figsize':(6,6)})
sns.barplot(x='Est_Salary',y='Sector',data=Sal_by_sector,palette="Accent").set(title='Salary Estimate by Sector',xlabel="Salary ($'000)")

#%% Firms age
plt.figure(figsize=(13,5))
sns.set(style='white') #style==background
sns.distplot(jobs['Years_Founded'], color="b")
plt.axvline(x=jobs.Years_Founded.mean(),
            color='k', linestyle='--')
plt.xlabel("Yrs founded")
plt.title("Companies ages",fontsize=19)
plt.xlim(0,210)
plt.xticks(np.arange(0, 150, step=10))
plt.tight_layout()
plt.show()


#%% Revenue and type of ownership
sns.boxplot(x=jobs["MaxRevenue"], y =jobs['Type_ownership']).set(ylabel='Ownership Type',xlabel="Max Revenue in billionsof USD")
plt.title("Max revenue per ownership type")

plt.show()


#%% Hires and Salary Estimate Revenue
RevCount = jobs.groupby('Revenue')[['job_title']].count().reset_index().rename(columns={'job_title':'Jobs'}).sort_values(
    'Jobs', ascending=False).reset_index(drop=True)

RevCount["Revenue_USD"]=['Unknown','10+ billion','100-500 million','50-100 million','2-5 billion','10-25 million','25-50 million','1-5 million','5-10 billion','<1 million','1-2 billion','0.5-1 billion','5-10 million']
RevCount2 = RevCount[['Revenue','Revenue_USD']]
RevCount = RevCount.merge(jobs, on='Revenue',how='left')

jobs=jobs.merge(RevCount2,on='Revenue',how='left')

sns.set(style="whitegrid")
f, (ax_bar, ax_point) = plt.subplots(ncols=2, sharey=True, gridspec_kw= {"width_ratios":(0.6,1)},figsize=(13,7))
sns.barplot(x='Jobs',y='Revenue_USD',data=RevCount,ax=ax_bar, palette='Accent').set(ylabel='Revenue in USD',xlabel="Hires")
sns.pointplot(x='Est_Salary',y='Revenue_USD',data=RevCount, join=False,ax=ax_point, palette='Accent').set(ylabel="",xlabel="Salary ($'000)")
plt.subplots_adjust(top=0.9)
plt.suptitle('Jobs and Salary by Revenue', fontsize = 16)
plt.tight_layout()

#%%
## More demanded skills
_ = frequent_itemsets[frequent_itemsets['length'] == 1]
_['itemsets'] = _['itemsets'].astype("unicode").str.replace('[\(\)\'\{\}]|frozenset','', regex = True)
ax = sns.barplot(x="itemsets", y="support", data= _,palette='Accent');
ax.set_xticklabels(ax.get_xticklabels(), rotation=90);
ax.set(ylabel="Frequency",xlabel="Skills", title= ' Main requested skills in data analysis')


##Deep diving in Virginia, Washington DC and Maryland
#%%
# create a separate dataset for Virginia and Washington DC
jobs_VA_DC_MD= jobs[(jobs['State']=='VA')|(jobs['State']=='DC')|(jobs['State']=='MD')]
jobs_VA_DC_MD

#Visual Exploration

#%% Avg salary distribution for data scientists national and regional
plt.figure(figsize=(13,5))
sns.set(style= 'white') #style==background
sns.distplot(jobs_VA_DC_MD['Est_Salary'], color="r")
sns.distplot(jobs['Est_Salary'], color="g")

plt.xlabel("Salary ($'000)")
plt.legend({'Est_Salary VA_MD_DC':jobs_VA_DC_MD['Est_Salary'],'Est_Salary_all':jobs['Est_Salary']})
plt.title("Distribution of Avg Salary in VA,DC,MD and national level",fontsize=19)
plt.xlim(0,210)
plt.xticks(np.arange(0, 210, step=10))
plt.tight_layout()
plt.show()
plt.savefig('avg_sal.png', dpi=300)


#%%
#Comparison heatmap
# Table for heatmap of number of companies with different sizes and revenues
Firm_Size = jobs.pivot_table(columns="Size",index="Revenue_USD",values="Company_Name",aggfunc=pd.Series.nunique).reset_index()
Firm_Size = Firm_Size[['Revenue_USD','1 to 50 employees','51 to 200 employees','201 to 500 employees','501 to 1000 employees','1001 to 5000 employees','5001 to 10000 employees','10000+ employees']]
Firm_Size = Firm_Size.reindex([11,2,9,4,7,10,5,0,1,6,8,3,12])
Firm_Size = Firm_Size.set_index('Revenue_USD').replace(np.nan,0)

# Table for heatmap of number of companies with different sizes and revenues in VA,DC,MA
Firm_Size_VA_DC_MD = jobs_VA_DC_MD.pivot_table(columns="Size",index="Revenue_USD",values="Company_Name",aggfunc=pd.Series.nunique).reset_index()
Firm_Size_VA_DC_MD = Firm_Size_VA_DC_MD[['Revenue_USD','1 to 50 employees','51 to 200 employees','201 to 500 employees','501 to 1000 employees','1001 to 5000 employees','5001 to 10000 employees','10000+ employees']]
Firm_Size_VA_DC_MD = Firm_Size_VA_DC_MD.reindex([11,2,9,4,7,10,5,0,1,6,8,3,12])
Firm_Size_VA_DC_MD = Firm_Size_VA_DC_MD.set_index('Revenue_USD').replace(np.nan,0)

# Table for heatmap of salaries by companies with different sizes and revenues
Firm_Size_Sal = jobs.pivot_table(columns="Size",index="Revenue_USD",values="Est_Salary",aggfunc=np.mean).reset_index()
Firm_Size_Sal = Firm_Size_Sal[['Revenue_USD','1 to 50 employees','51 to 200 employees','201 to 500 employees','501 to 1000 employees','1001 to 5000 employees','5001 to 10000 employees','10000+ employees']]
Firm_Size_Sal = Firm_Size_Sal.reindex([11,2,9,4,7,10,5,0,1,6,8,3,12])
Firm_Size_Sal = Firm_Size_Sal.set_index('Revenue_USD').replace(np.nan,0)

# Table for heatmap of salaries by companies with different sizes and revenues in CA
Firm_Size_VA_DC_MD_Sal = jobs_VA_DC_MD.pivot_table(columns="Size",index="Revenue_USD",values="Est_Salary",aggfunc=np.mean).reset_index()
Firm_Size_VA_DC_MD_Sal = Firm_Size_VA_DC_MD_Sal[['Revenue_USD','1 to 50 employees','51 to 200 employees','201 to 500 employees','501 to 1000 employees','1001 to 5000 employees','5001 to 10000 employees','10000+ employees']]
Firm_Size_VA_DC_MD_Sal = Firm_Size_VA_DC_MD_Sal.reindex([11,2,9,4,7,10,5,0,1,6,8,3,12])
Firm_Size_VA_DC_MD_Sal = Firm_Size_VA_DC_MD_Sal.set_index('Revenue_USD').replace(np.nan,0)
# %%
### Comparison between revenue and salaries and size VA,DC,MD and all 
f, axs = plt.subplots(nrows=2,ncols=2, sharey=True,sharex=True, figsize=(13,9))

fs = sns.heatmap(Firm_Size,annot=True,fmt='.0f',annot_kws={"size": 12},cmap="YlGnBu", ax=axs[0,0]).set(title="Number of Firms offering jobs for Data Scientist roles (US)",xlabel="",ylabel="Revenue USD")
fsc = sns.heatmap(Firm_Size_VA_DC_MD,annot=True,fmt='.0f',annot_kws={"size": 12},cmap="YlGnBu", ax=axs[0,1]).set(title="Number of Firms offering jobs for Data Scientist roles(VA,DC,MD)",xlabel="",ylabel="")
fss = sns.heatmap(Firm_Size_Sal,annot=True,fmt='.0f',annot_kws={"size": 12},cmap="Greens",ax=axs[1,0]).set(title="Avg. Salaries of Data Scientist roles (US)",ylabel="Revenue USD")
fscs = sns.heatmap(Firm_Size_VA_DC_MD_Sal,annot=True,fmt='.0f',annot_kws={"size": 12},cmap="Greens",ax=axs[1,1]).set(title="Avg. Salaries of Data Scientist roles (VA,DC,MD)",ylabel="")
plt.setp([a.get_xticklabels() for a in axs[1,:]],rotation=45,ha='right')
plt.tight_layout()
plt.show()

#%% Salary/Hires by Firm VA,DC,MD
df_by_firm_VA_DC_MD=jobs_VA_DC_MD.groupby('Company_Name')['job_title'].count().reset_index().sort_values(
    'job_title',ascending=False).head(20).rename(columns={'job_title':'Hires'})

Sal_by_firm_VA_DC_MD = df_by_firm.merge(jobs_VA_DC_MD,on='Company_Name',how='left')

sns.set(style="white")
f, (ax_bar, ax_point) = plt.subplots(ncols=2, sharey=True, gridspec_kw= {"width_ratios":(0.6,1)},figsize=(13,7))
sns.barplot(x='Hires',y='Company_Name',data=Sal_by_firm_VA_DC_MD,ax=ax_bar, palette='Accent').set(ylabel="")
sns.pointplot(x='Est_Salary',y='Company_Name',data=Sal_by_firm_VA_DC_MD, join=False,ax=ax_point, palette='Accent').set(
    ylabel="",xlabel="Salary ($'000)")
plt.subplots_adjust(top=0.9)
plt.suptitle('Hiring and salary by firms in VA,DC,MD', fontsize = 16)
plt.tight_layout()

#%%

###Anova Analysis to check for correlation between numerical and categorical variables
import statsmodels.api as sm
from statsmodels.formula.api import ols

model = ols('Est_Salary ~ State', data=jobs).fit()
aov_table = sm.stats.anova_lm(model, typ=2)
aov_table

#Sector
from scipy.stats import f_oneway
CategoryGroupLists=jobs.groupby('Sector')['Est_Salary'].apply(list)
AnovaResults = f_oneway(*CategoryGroupLists)
print('P-Value for Anova between Sector and Est_Salary is: ', AnovaResults[1])

#%%
#Industry
CategoryGroupLists2=jobs.groupby('Industry')['Est_Salary'].apply(list)
AnovaResults = f_oneway(*CategoryGroupLists2)
print('P-Value for Anova between Industry and Est_Salary is: ', AnovaResults[1])

#%%
#State
CategoryGroupLists2=jobs.groupby('State')['Est_Salary'].apply(list)
AnovaResults = f_oneway(*CategoryGroupLists2)
print('P-Value for Anova between State and Est_Salary is: ', AnovaResults[1])

#%%
#HQState
CategoryGroupLists2=jobs.groupby('HQState')['Est_Salary'].apply(list)
AnovaResults = f_oneway(*CategoryGroupLists2)
print('P-Value for Anova between HQ and Est_Salary is: ', AnovaResults[1])

# %%
#Company
CategoryGroupLists4=jobs.groupby('Company_Name')['Est_Salary'].apply(list)
AnovaResults = f_oneway(*CategoryGroupLists4)
print('P-Value for Anova between Company and Est_Salary is: ', AnovaResults[1])

# %%
# Job Domain
CategoryGroupLists6=jobs.groupby('Job_Domain')['Est_Salary'].apply(list)
AnovaResults = f_oneway(*CategoryGroupLists6)
print('P-Value for Anova between Job_Domain and Est_Salary is: ', AnovaResults[1])


# %%
#Revenue
CategoryGroupLists3=jobs.groupby('Revenue')['Est_Salary'].apply(list)
AnovaResults = f_oneway(*CategoryGroupLists3)
print('P-Value for Anova is: ', AnovaResults[1])


# %%
#Rating
CategoryGroupLists5=jobs.groupby('Rating')['Est_Salary'].apply(list)
AnovaResults = f_oneway(*CategoryGroupLists5)
print('P-Value for Anova is: ', AnovaResults[1])


# %%
import statsmodels.api as sm
from statsmodels.formula.api import ols



#%%
#Converting float data type variables to int for ease of modelling.

jobs['Rating']=jobs['Rating'].astype(int)
jobs['Founded']=jobs['Founded'].astype(int)
jobs['MaxEmpSize']=jobs['MaxEmpSize'].astype(int)
jobs['Est_Salary']=jobs['Est_Salary'].astype(int)
jobs['Years_Founded']=jobs['Years_Founded'].astype(int)
jobs['MaxRevenue']=jobs['MaxRevenue'].astype(int)


#%%
# create a new dataset from original data for job title
jobs_lm = jobs[['job_title','Est_Salary','Max_Salary','Min_Salary','State','City','MaxRevenue','Rating','MaxEmpSize','Industry','Sector','Type_ownership','Years_Founded','Company_Name','HQState']]
# remove special characters and unify some word use
jobs_lm['job_title2']= jobs_lm['job_title'].str.upper().replace(
    [',','Â','/','\t','\n','-','AND ','&','\(','\)','WITH ','SYSTEMS','OPERATIONS','ANALYTICS','SERVICES','\[','\]','ENGINEERS','NETWORKS','GAMES','MUSICS','INSIGHTS','SOLUTIONS','JR.','MARKETS','STANDARDS','FINANCE','ENGINEERING','PRODUCTS','DEVELOPERS','SR. ','SR ','JR. ','JR '],
    ['','',' ',' ',' ',' ','',' ',' ',' ','','SYSTEM','OPERATION','ANALYTIC','SERVICE','','','ENGINEER','NETWORK','GAME','MUSIC','INSIGHT','SOLUTION','JUNIOR','MARKET','STANDARD','FINANCIAL','ENGINEER','PRODUCT','DEVELOPER','SENIOR ','SENIOR ','JUNIOR ','JUNIOR '],regex=True)
#%%
jobs_lm['job_title2']= jobs_lm['job_title2'].str.upper().replace(['  ','   ','    '], [' ',' ',' '],regex=True)

#Unifying words
jobs_lm['job_title2']= jobs_lm['job_title2'].str.upper().replace(
    ['BUSINESS INTELLIGENCE','INFORMATION TECHNOLOGY','QUALITY ASSURANCE','USER EXPERIENCE','USER INTERFACE','DATA WAREHOUSE','DATA ANALYST','DATA BASE','DATA QUALITY','DATA GOVERNANCE','BUSINESS ANALYST','DATA MANAGEMENT','REPORTING ANALYST','BUSINESS DATA','SYSTEM ANALYST','DATA REPORTING','QUALITY ANALYST'],
    ['BI','IT','QA','UX','UI','DATA_WAREHOUSE','DATA_ANALYST','DATABASE','DATA_QUALITY','DATA_GOVERNANCE','BUSINESS_ANALYST','DATA_MANAGEMENT','REPORTING_ANALYST','BUSINESS_DATA','SYSTEM_ANALYST','DATA_REPORTING','QUALITY_ANALYST'],regex=True)

#more unifying
jobs_lm['job_title2']= jobs_lm['job_title2'].str.upper().replace(
    ['DATA_ANALYST JUNIOR','DATA_ANALYST SENIOR','DATA  REPORTING_ANALYST'],
    ['JUNIOR DATA_ANALYST','SENIOR DATA_ANALYST','DATA_REPORTING_ANALYST'],regex=True)
#%%
jobCount=jobs_lm.groupby('job_title2')[['job_title']].count().reset_index().rename(
    columns={'job_title':'Count'}).sort_values('Count',ascending=False)
jobSalary = jobs_lm.groupby('job_title2')[['Max_Salary','Est_Salary','Min_Salary']].mean().sort_values(
    ['Max_Salary','Est_Salary','Min_Salary'],ascending=False)
jobSalary['Spread']=jobSalary['Max_Salary']-jobSalary['Est_Salary']
jobSalary=jobSalary.merge(jobCount,on='job_title2',how='left').sort_values('Count',ascending=False).head(20)

# %%
jobs["Revenue_USD"] = jobs["Revenue_USD"].fillna(jobs["Revenue_USD"].mode()[0])

#%%
#Checking for any null values
jobs.isnull().sum()



#%%
#Identyfying top words
ds = jobs_lm['job_title2'].str.split(expand=True).stack().value_counts().reset_index().rename(columns={'index':'TW',0:'Count'})
DS = ds[ds['Count']>1000]
DS
# %%

## loop for top words
def get_keyword(x):
   x_ = x.split(" ")
   keywords = []
   try:
      for word in x_:
         if word in np.asarray(DS['TW']):
            keywords.append(word)
   except:
      return -1

   return keywords


#%%
#keywords from each row
jobs_lm['TW'] = jobs_lm['job_title2'].apply(lambda x: get_keyword(x))

#%%
# dummy columns by top words

twdummy = pd.get_dummies(jobs_lm['TW'].apply(pd.Series).stack()).sum(level=0).replace(2,1)
jobs_lm = jobs_lm.merge(twdummy,left_index=True,right_index=True).replace(np.nan,0)

# %%
jobs_lm.to_csv('jobs_lm.csv')
# %%
from scipy import stats
# running a  t-test for top words to check for correlation with salaries
topwords = list(jobs_lm.columns)
ttests=[]
for word in topwords:
    if word in set(DS['TW']):
        ttest = stats.ttest_ind(jobs_lm[jobs_lm[word]==1]['Est_Salary'],
                                     jobs_lm[jobs_lm[word]==0]['Est_Salary'])
        ttests.append([word,ttest])


ttests = pd.DataFrame(ttests,columns=['TW','R'])
ttests['R']=ttests['R'].astype(str).replace(['Ttest_indResult\(statistic=','pvalue=','\)'],['','',''],regex=True)
ttests['Statistic'],ttests['P-value']=ttests['R'].str.split(', ',1).str
ttests=ttests.drop(['R'],axis=1).sort_values('P-value',ascending=True)
ttests
# %%
# Selecting top words with p-value <0.1 into multiple regression model.
ttest_pass = list(ttests[ttests['P-value'].astype(float)<0.1]['TW'])
print(*ttest_pass,sep=' + ')

# %%
####FEATURE IMPORTANCE

### Converting variables into dummies 
#%%
jobs2= jobs

##Changing states to dummies
jobs2.State_Location.replace({'NY': 0,'NJ': 1, 'CA':2, 'IL':3, 'TX':4,
'AZ':5, 'PA': 6, 'DE':7,'FL':8,'IN':9,'OH':10,'NC':11,'SC':12,'UT':13,
'VA':14,'WA':15,'GA':16,'KS':17,'CO':18,'DC':19,'MD':20,'MA':21,'TN':22,
'MI':23,'OK':24,'OR':25,'NV':26,'KY':27,'WI':28,'NM':29,'MO':30,'NE':31,
'MN':32,'LA':33,'AK':34,'VT':35,'MS':36,'CT':37,'PR':38, 'HI':39}, inplace=True)

#Changing sectors to dummies
jobs2.Sector.replace({'Health Care':0, 'Finance': 1, 'Biotech & Pharmaceuticals':2,
'Manufacturing':3, 'Information Technology':4, 'Insurance':5,
'Business Services':6, 'Education':7, 'Media':8, 'Consumer Services':9,
'Restaurants, Bars & Food Services':10, 'Retail':11, 'Accounting & Legal':12,
'Non-Profit':13, 'Oil, Gas, Energy & Utilities':14, 'Agriculture & Forestry':15,
'Transportation & Logistics':16, 'Aerospace & Defense':17, 'Travel & Tourism':18,
'Construction, Repair & Maintenance':19, 'Government':20, 'Real Estate':21,
'Telecommunications':22, 'Arts, Entertainment & Recreation':23, 'Mining & Metals':24}, inplace=True)


jobs2.Type_ownership.replace({'Nonprofit Organization': 0, 'Company - Private':1, 'Company - Public':2, 'Subsidiary or Business Segment':3,
'College / University':4, 'Contract':5, 'Self-employed':6, 'Unknown':7,
'Hospital':8, 'Government':9, 'Other Organization':10, 'School / School District':11,
'Franchise':12, 'Private Practice / Firm':13}, inplace=True)
#%%
# Adding random features
X = jobs[['Rating','Sector','MaxEmpSize','State_Location', 'MaxRevenue', 'Years_Founded']]
y = jobs[['Est_Salary']]

#%%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,
                                                    random_state=42)
#%%
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train_scaled = ss.fit_transform(X_train)
X_test_scaled = ss.transform(X_test)

#%%
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train_scaled, y_train)
importances = pd.DataFrame(data={
    'Attribute': X_train.columns,
    'Importance': model.coef_[0]
})
importances = importances.sort_values(by='Importance', ascending=False)

#%%
plt.bar(x=importances['Attribute'], height=importances['Importance'], color='#087E8B')
plt.title('Feature importances obtained from coefficients', size=20)
plt.xticks(rotation='vertical')
plt.show()



# %%
## Modeling:


#%%

### eda

#%%
def title_simplifier(title):
    if 'business analyst' in title.lower():
        return 'business analyst'
    elif 'data scientist' in title.lower():
        return 'data scientist'
    elif 'data engineer' in title.lower():
        return 'data engineer'
    elif 'data analyst' in title.lower():
        return 'data analyst'
    elif 'analyst' in title.lower():
        return 'analyst'
    elif 'machine learning' in title.lower():
        return 'mle'
    elif 'consultant' in title.lower():
        return 'consultant'
    elif 'engineer' in title.lower():
        return 'engineer'    
    elif 'manager' in title.lower() or 'executive' in title.lower() or 'principal' in title.lower():
        return 'manager'
    elif 'director' in title.lower():
        return 'director'
    elif 'scientist' in title.lower():
        return 'Other Scientist'
    else:
        return 'other' #'na'
    
def seniority(title):
    if 'sr' in title.lower() or 'senior' in title.lower() or 'sr' in title.lower() or 'lead' in title.lower() or 'principal' in title.lower() or 'manager' in title.lower() or 'manager' in title.lower() or 'executive' in title.lower() or 'director' in title.lower():
        return 'senior'
    elif 'junior' in title.lower() or 'jr' in title.lower() or 'jr.' in title.lower():
        return 'jr'
    else:
        return 'na'

#%%
jobs['job_simp'] = jobs['job_title'].apply(title_simplifier)

#%%
print(jobs.job_simp.value_counts())

#%%
jobs['seniority'] = jobs['job_title'].apply(seniority)
#%%
jobs.seniority.value_counts()

#%%
jobs.isnull().sum()

# %%
df_numeric = jobs.select_dtypes(include=np.number)
df_numeric.head()

#%%
corr = df_numeric.corr()
corr.style.background_gradient(cmap='coolwarm')

# %%
df_numeric = df_numeric.drop(labels=['tableau', 'bi', 'Min_Salary', 'Max_Salary', 'Founded', 'MaxEmpSize', 'MaxRevenue'],axis=1) # Removing unnecesary column 
df_numeric.head()

#%%
df_numeric.isnull().sum()

#%%
df_numeric.shape

#%%
df_categoric = jobs.select_dtypes(include = object)
df_categoric.head()

#%%
df_categoric.Size=df_categoric.Size.fillna(df_categoric.Size.mode()[0])
df_categoric.Revenue=df_categoric.Revenue.fillna(df_categoric.Revenue.mode()[0])

# %%
df_categoric = df_categoric.drop(labels=['Industry', 'job_title', 'Company_Name','Location', 'Headquarters', 'Job_Domain', 'Job Role', 'refined_skills', 'City', 'State', 'HQCity', 'HQState', 'Revenue_USD'],axis=1) # Removing unnecesary column 

#%%
df_categoric.isnull().sum()
# %%
df_categoric.head()

# %%
df_categoric.shape
# %%
dummy_encoded_variables = pd.get_dummies(df_categoric, drop_first = True)
dummy_encoded_variables.head()


# %%
dummy_encoded_variables.shape

#%%
# concatenate the numerical and dummy encoded categorical variables column-wise
df_dummy = pd.concat([df_numeric, dummy_encoded_variables], axis=1)

# display data with dummy variables
df_dummy.head()

# Model Codes:

#%%
# add the intercept column using 'add_constant()'
df_dummy = sm.add_constant(df_dummy)

# separate the independent and dependent variables
# drop(): drops the specified columns
X = df_dummy.drop(["Est_Salary"], axis = 1)

# extract the target variable from the data set
y = df_dummy[["Est_Salary"]]

#%%
X.shape
#%%
y.shape

# %%
#importing sklearn for split data
from sklearn.model_selection import train_test_split

# split data into train data and test data 
# what proportion of data should be included in test data is passed using 'test_size'
# set 'random_state' to get the same data each time the code is executed 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10)

# check the dimensions of the train & test subset for 
# print dimension of predictors train set
print("The shape of X_train is:",X_train.shape)

# print dimension of predictors test set
print("The shape of X_test is:",X_test.shape)

# print dimension of target train set
print("The shape of y_train is:",y_train.shape)

# print dimension of target test set
print("The shape of y_test is:",y_test.shape)


#%%

## OLS

# %%
# build a full model using OLS()
# consider the log of sales price as the target variable
# use fit() to fit the model on train data
linreg_full = sm.OLS(y_train, X_train).fit()

# print the summary output
print(linreg_full.summary())

#%%
linreg_full_predictions = linreg_full.predict(X_test)
linreg_full_predictions


#%%
actual_salary = y_test["Est_Salary"]
actual_salary
#%%
#importing stats model to add intercept and to implement OLS
import statsmodels.api as sm
#importing sklearn for split data
from sklearn.model_selection import train_test_split
from statsmodels.tools.eval_measures import rmse
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# importing random lib for replace age values
import random

#%%
# calculate rmse using rmse()
linreg_full_rmse = rmse(actual_salary,linreg_full_predictions )

# calculate R-squared using rsquared
linreg_full_rsquared = linreg_full.rsquared

# calculate Adjusted R-Squared using rsquared_adj
linreg_full_rsquared_adj = linreg_full.rsquared_adj 

#%%
# create the result table for all accuracy scores
# accuracy measures considered for model comparision are RMSE, R-squared value and Adjusted R-squared value
# create a list of column names
cols = ['Model', 'RMSE', 'R-Squared', 'Adj. R-Squared']

# create a empty dataframe of the colums
# columns: specifies the columns to be selected
result_tabulation = pd.DataFrame(columns = cols)

# compile the required information
linreg_full_metrics = pd.Series({'Model': "Linreg full model ",
                     'RMSE':linreg_full_rmse,
                     'R-Squared': linreg_full_rsquared,
                     'Adj. R-Squared': linreg_full_rsquared_adj     
                   })

# append our result table using append()
# ignore_index=True: does not use the index labels
# python can only append a Series if ignore_index=True or if the Series has a name
result_tabulation = result_tabulation.append(linreg_full_metrics, ignore_index = True)

# print the result table
result_tabulation


# %%
# 2. Decision Tree
from sklearn import tree
from sklearn.tree import export_graphviz
from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

# %%
# instantiate the 'DecisionTreeRegressor' object using 'mse' criterion
# pass the 'random_state' to obtain the same samples for each time you run the code
decision_tree = DecisionTreeRegressor(criterion = 'mse', random_state = 10) #Max depth D.Tree gets formed

# fit the model using fit() on train data
decision_tree_model = decision_tree.fit(X_train, y_train) #fit() method is defined inside the class 'DecisionTreeClassifier'

# %%
y_pred_DT=decision_tree_model.predict(X_test)

#%%

r_squared_DT=decision_tree_model.score(X_test,y_test)
# Number of observation or sample size
n = len(X_train)

# No of independent variables
p = len(X_train.columns)

#Compute Adj-R-Squared
Adj_r_squared_DT = 1 - (1-r_squared_DT)*(n-1)/(n-p-1)
Adj_r_squared_DT

#%%
# Compute RMSE
from math import sqrt
rmse_DT = sqrt(mean_squared_error(y_test, y_pred_DT))
# compile the required information
linreg_full_metrics = pd.Series({'Model': "Decision Tree Model ",
                     'RMSE':rmse_DT,
                     'R-Squared': r_squared_DT,
                     'Adj. R-Squared': Adj_r_squared_DT     
                   })

# append our result table using append()
# ignore_index=True: does not use the index labels
# python can only append a Series if ignore_index=True or if the Series has a name
result_tabulation = result_tabulation.append(linreg_full_metrics, ignore_index = True)

# print the result table
result_tabulation

# 3. Random Forest

#%%
# import library for random forest regressor
from sklearn.ensemble import RandomForestRegressor
#intantiate the regressor
rf_reg = RandomForestRegressor(n_estimators=100, random_state=10)

# fit the regressor with training dataset
rf_reg.fit(X_train, y_train)

y_pred_RF = rf_reg.predict(X_test)

#%%
# Calculate MAE
rf_reg_MAE = metrics.mean_absolute_error(y_test, y_pred_RF)
print('Mean Absolute Error (MAE):', rf_reg_MAE)

# Calculate MSE
rf_reg_MSE = metrics.mean_squared_error(y_test, y_pred_RF)
print('Mean Squared Error (MSE):', rf_reg_MSE)

# Calculate RMSE
rf_reg_RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pred_RF))
print('Root Mean Squared Error (RMSE):', rf_reg_RMSE)

#%%
r_squared_RF=rf_reg.score(X_test,y_test)
# Number of observation or sample size
n = len(X_train)

# No of independent variables
p = len(X_train.columns) 

#Compute Adj-R-Squared
Adj_r_squared_RF = 1 - (1-r_squared_RF)*(n-1)/(n-p-1)
# Compute RMSE
rmse_RF = sqrt(mean_squared_error(y_test, y_pred_RF))

#%%
# compile the required information
linreg_full_metrics = pd.Series({'Model': "Random Forest ",
                     'RMSE':rf_reg_RMSE,
                     'R-Squared': r_squared_RF,
                     'Adj. R-Squared': Adj_r_squared_RF     
                   })

# append our result table using append()
# ignore_index=True: does not use the index labels
# python can only append a Series if ignore_index=True or if the Series has a name
result_tabulation = result_tabulation.append(linreg_full_metrics, ignore_index = True)

# print the result table
result_tabulation

#%%

# 7. Ensemble Techniques
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression
x2 = X
y2 = y
x2.shape
X.shape
#%%
select_feature = SelectKBest(f_regression, k=42).fit(x2,y2)
X2=select_feature.transform(x2)
X2.shape
#%%
D=select_feature.get_support(indices=True)
D

#%%
features_df_new = X.iloc[:,D]
features_df_new.columns

#%%
#import libraries
from sklearn.ensemble import BaggingRegressor
# split data into train data and test data 
# what proportion of data should be included in test data is passed using 'test_size'
# set 'random_state' to get the same data each time the code is executed 
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size = 0.2, random_state = 1)

# check the dimensions of the train & test subset for 
# print dimension of predictors train set
print("The shape of X2_train is:",X2_train.shape)

# print dimension of predictors test set
print("The shape of X2_test is:",X2_test.shape)

# print dimension of target train set
print("The shape of y2_train is:",y2_train.shape)

# print dimension of target test set
print("The shape of y2_test is:",y2_test.shape)
#%%
# build the model
meta_estimator = BaggingRegressor(tree.DecisionTreeRegressor(random_state=10)) #Similar to a random forest, just that the DT's are having all the features to split on

# fit the model
meta_estimator.fit(X2_train, y2_train) 


#%%
# predict the values
y_pred_ET = meta_estimator.predict(X2_test)
y_pred_ET

#%%
r_squared_ET=meta_estimator.score(X2_test,y2_test)
# Number of observation or sample size
n = len(X2_train) 

# No of independent variables
p = len(X_train.columns)

#Compute Adj-R-Squared
Adj_r_squared_ET = 1 - (1-r_squared_ET)*(n-1)/(n-p-1)
# Compute RMSE
rmse_ET = sqrt(mean_squared_error(y2_test, y_pred_ET))

#%%
# compile the required information
linreg_full_metrics = pd.Series({'Model': "Ensemble Techniques (Bagging Meta Estimator) ",
                     'RMSE':rmse_ET,
                     'R-Squared': r_squared_ET,
                     'Adj. R-Squared': Adj_r_squared_ET     
                   })

# append our result table using append()
# ignore_index=True: does not use the index labels
# python can only append a Series if ignore_index=True or if the Series has a name
result_tabulation = result_tabulation.append(linreg_full_metrics, ignore_index = True)

# print the result table
result_tabulation

#%%

# 8. Ensemble Techniques(Adaboost)
from sklearn.ensemble import AdaBoostRegressor

# build the model
adaboost = AdaBoostRegressor(random_state=10)
# fit the model
adaboost.fit(X2_train, y2_train)

y_pred_adaboost  = adaboost.predict(X2_test)

#%%
r_squared_ADA=meta_estimator.score(X2_test,y2_test)
# Number of observation or sample size
n = len(X2_train) 

# No of independent variables
p = len(X_train.columns)

#Compute Adj-R-Squared
Adj_r_squared_ADA = 1 - (1-r_squared_ADA)*(n-1)/(n-p-1)
# Compute RMSE
rmse_ADA = sqrt(mean_squared_error(y2_test, y_pred_adaboost))

#%%
# compile the required information
linreg_full_metrics = pd.Series({'Model': "Ensemble Techniques (ADA Boost) ",
                     'RMSE':rmse_ADA,
                     'R-Squared': r_squared_ADA,
                     'Adj. R-Squared': Adj_r_squared_ADA     
                   })

# append our result table using append()
# ignore_index=True: does not use the index labels
# python can only append a Series if ignore_index=True or if the Series has a name
result_tabulation = result_tabulation.append(linreg_full_metrics, ignore_index = True)

# print the result table
result_tabulation

#%%
# 6 XGBoost

import xgboost as xgb

#%%
xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 15, alpha = 10, n_estimators = 150)

#%%
xg_reg.fit(X_train,y_train)
xg_preds = xg_reg.predict(X_test)


#%%
# Calculate MAE
xg_reg_MAE = metrics.mean_absolute_error(y_test, xg_preds)
print('Mean Absolute Error (MAE):', xg_reg_MAE)

# Calculate MSE
xg_reg_MSE = metrics.mean_squared_error(y_test, xg_preds)
print('Mean Squared Error (MSE):', xg_reg_MSE)

# Calculate RMSE
xg_reg_RMSE = np.sqrt(metrics.mean_squared_error(y_test, xg_preds))
print('Root Mean Squared Error (RMSE):', xg_reg_RMSE)



#%%
print('The accuracy of the xgboost classifier is {:.2f} out of 1 on the training data'.format(xg_reg.score(X_train, y_train)))
print('The accuracy of the xgboost classifier is {:.2f} out of 1 on the test data'.format(xg_reg.score(X_test, y_test)))

#%%
r_squared_xg=xg_reg.score(X_test,y_test)
# Number of observation or sample size
n = len(X_train)

# No of independent variables
p = len(X_train.columns) 

#Compute Adj-R-Squared
Adj_r_squared_xg = 1 - (1-r_squared_xg)*(n-1)/(n-p-1)
# Compute RMSE
rmse_xg = sqrt(mean_squared_error(y_test, xg_preds))

#%%
# compile the required information
xg_full_metrics = pd.Series({'Model': "XGBoost",
                     'RMSE':xg_reg_RMSE,
                     'R-Squared': r_squared_xg,
                     'Adj. R-Squared': Adj_r_squared_xg     
                   })

# append our result table using append()
# ignore_index=True: does not use the index labels
# python can only append a Series if ignore_index=True or if the Series has a name
result_tabulation = result_tabulation.append(xg_full_metrics, ignore_index = True)

# print the result table
#%%
result_tabulation
