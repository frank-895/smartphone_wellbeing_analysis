import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

# Load CSV files into pandas dataframes
demographics = pd.read_csv('dataset1.csv')
time_of_week = pd.read_csv('dataset2.csv')
outcome_self_report = pd.read_csv('dataset3.csv')

# Merge the dataframes on the 'ID' column
independent_variables = pd.merge(demographics, time_of_week, on='ID', how='inner')
final_df = pd.merge(outcome_self_report, independent_variables, on='ID', how='inner')
print('[-----] This is some info on the final merged data frame [-----]')
print(final_df.info())

print('\n\n[-----] This is the head of the final merged data frame [-----]')
print(final_df)

# Select independent variables and dependent variables
# Assuming the independent variables are in the final DataFrame, excluding 'ID'
X = final_df.drop(columns=['ID'] + [col for col in outcome_self_report.columns])
y = final_df[outcome_self_report.columns].drop(columns=['ID'])

# Some constants and empty dictionaries to append to
X = sm.add_constant(X)
results = {}
coefficients = {}
r_2_values = {}

for i in y.columns:
    model = sm.OLS(y[i], X).fit() # Fit the model
    results[i] = model.summary() # Summary of linear regression model
    r_2_values[i] = float(model.rsquared) # R^2 values
    coefficients[i] = model.params # Coefficients of regressions!.

# Convert, print and output the results
print(results)
coefficients = pd.DataFrame.from_dict(coefficients, orient='index')
print(coefficients)
coefficients.to_csv('coefficients_outcome.csv', index=True)
r_2_values = pd.DataFrame.from_dict(r_2_values, orient='index', columns = ["Adjusted R Squared Values"])
print(r_2_values)