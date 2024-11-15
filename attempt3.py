import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("cleaned_data.csv")

# Build regression model with AVERAGE WELLBEING as response variable
x = df.iloc[:,1:-2]
y = df.iloc[:, -1]

# analyse correlation visually to confirm linear correlation. 

plt.figure(figsize=(10, 6))
sns.scatterplot(x=df['avg_screentime'], y=y)
plt.title(f'Scatterplot of Average Screentime vs Average Wellbeing')
plt.xlabel('Average Screentime (Normalised and Standardised)')
plt.ylabel('Average Wellbeing')
plt.show()

# Fit multiple linear regression model
x_constant = sm.add_constant(x)
avg_wellbeing_model = sm.OLS(y, x_constant).fit()

print(avg_wellbeing_model.summary())

# Build regression model with TOTAL WELLBEING as response variable
y = df.iloc[:, -2]

# analyse correlation visually to confirm linear correlation. 
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df['avg_screentime'], y=y)
plt.title(f'Scatterplot of Average Screentime vs Total Wellbeing')
plt.xlabel('Average Screentime (Normalised and Standardised)')
plt.ylabel('Total Wellbeing')
plt.show()

total_wellbeing_model = sm.OLS(y, x_constant).fit()

print(total_wellbeing_model.summary())