import pandas as pd
from statsmodels.miscmodels.ordinal_model import OrderedModel

# read in data
df2 = pd.read_csv("dataset2.csv")
df3 = pd.read_csv("dataset3.csv")

# generate total screentime to use as explanatory variable
df2['total_screentime'] = df2.iloc[:,1:].sum(axis=1)

rsquareds = {} # to store pseudo r squareds
for i in range(len(df3.columns) - 1): # iterate through possible response variables  
    response_variable = df3.columns[i + 1]
    reg_df = pd.merge(df2[['ID', 'total_screentime']], df3[['ID', response_variable]], on='ID', how='inner') # build regression table with individual stats
    
    # build model 
    x = reg_df.iloc[:,1:-1]
    y = reg_df.iloc[:,-1]
    result = OrderedModel(y, x, distr='logit').fit()
    
    # store pseudo r squared result
    rsquareds[response_variable] = float(result.prsquared)

for key in rsquareds: # display results
    print("Pseudo R Squared value for", key, "is", rsquareds[key])
