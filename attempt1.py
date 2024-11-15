import pandas as pd
import statsmodels.api as sm

# read in data and remove invalid rows
df1 = pd.read_csv("dataset1.csv").dropna()
df2 = pd.read_csv("dataset2.csv").dropna()
df3 = pd.read_csv("dataset3.csv").dropna()

# we will see if there are any particular variables with strong correlation to total screen time we can use as responsive variables
df2['total_screentime'] = df2.iloc[:,1:].sum(axis=1)
reg_df = pd.merge(df2[['ID', 'total_screentime']], df3, on='ID', how='inner') # this df matches corresponding total screen time and wellbeing
correlation_matrix = reg_df.iloc[:,1:].corr()

# we will choose to build a regression model from the wellbeing variable with the highest correlation
highest = 0
for i in range(1, len(correlation_matrix)):
    if abs(correlation_matrix['total_screentime'].iloc[i]) > highest:
        response_variable = correlation_matrix.columns[i]
        highest = abs(correlation_matrix['total_screentime'].iloc[i])

print("The response variable for regression building is: ", response_variable)

# We will build a progressively improving regression model
# We will start with a baseline model, using only the mean of the response variable to predict
x = [1] * len(df3) # constant predictor
y = df3[response_variable]

# Fit baseline model
x = sm.add_constant(x)
baseline_model = sm.OLS(y, x).fit()

# We will improve the model further by using the screentime statistics as explanatory variables and building a multiple regression
reg_df = pd.merge(df2.iloc[:,:-1], df3[['ID', response_variable]], on='ID', how='inner') # build regression table with individual stats
x = reg_df.iloc[:,1:-1]
y = reg_df.iloc[:,-1]

# Fit multiple linear regression model
x = sm.add_constant(x)
multiple_reg_model = sm.OLS(y, x).fit()

print("BASELINE MODEL adjusted R^2 =", baseline_model.rsquared_adj)
print("MULTIPLE REGRESSION MODEL adjusted R^2 =", multiple_reg_model.rsquared_adj)
# The multiple linear regression model has an R^2 value of 0.048 (very weak correlation)

# We will add demographic variables to improve the prediction
reg_df = pd.merge(df1, reg_df, on='ID', how='inner')

x = reg_df.iloc[:,1:-1]
y = reg_df.iloc[:,-1]

# Fit multiple linear regression model with demographics
x = sm.add_constant(x)
multiple_reg_model2 = sm.OLS(y, x).fit()

print("MULTIPLE REGRESSION MODEL WITH DEMOGRAPHICS adjusted R^2 =", multiple_reg_model2.rsquared_adj)