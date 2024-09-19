import pandas as pd
import scipy.stats as sc

# Alternative Hypothesis: The mean self-reported wellbeing scores of high smartphone users is lower than that of low smartphone users.
# Null Hypothesis: The mean self-reported wellbeing scores of high smartphone users is the same as low smartphone users.

df2 = pd.read_csv('dataset2.csv') # read in data
df3 = pd.read_csv('dataset3.csv')
df2 = df2.dropna() # remove invalid rows from df2

s_prop = [] # smartphone propotion values
for i, row in df2.iterrows(): # iterate through df2
    s_tot = row['S_we'] + row['S_wk'] # smartphone use
    tot = row['T_we'] + row['T_wk'] + row['G_we'] + row['G_wk'] + row['C_we'] + row['C_wk'] + row['S_we'] + row['S_wk'] # total tech use        
    if tot != 0:
        s_prop.append([int(row['ID']), float(s_tot/tot)]) # calculate smartphone use proportion
    
temp = pd.DataFrame(s_prop, columns=['ID', 's_prop']) # turn proportion into df

high = temp['s_prop'].quantile(0.8) # upper threshold
low = temp['s_prop'].quantile(0.2) # lower threshold

high_users_df = temp[temp['s_prop'] > high] # df only containing high users
low_users_df = temp[temp['s_prop'] < low] # df only containing low users

high_users_df = pd.merge(high_users_df, df3, on='ID', how='inner') # calculate mean of happiness indicators
high_users_df = high_users_df.drop(columns=['ID', 's_prop'])
avg_happiness_high = pd.DataFrame(high_users_df.mean(axis=0))

low_users_df = pd.merge(low_users_df, df3, on='ID', how='inner')  # calculate mean of happiness indicators
low_users_df = low_users_df.drop(columns=['ID', 's_prop'])
avg_happiness_low = pd.DataFrame(low_users_df.mean(axis=0))

# perform t-test 
p_val = round(sc.ttest_ind(avg_happiness_high, avg_happiness_low, alternative='less').pvalue[0], 3) # calculate p-value

# display results
print("\nThe mean wellbeing score of high users was", round(avg_happiness_high[0].mean(), 3), " and the standard deviation was ", round(avg_happiness_high[0].std(), 3))
print("The mean wellbeing score of low users was", round(avg_happiness_low[0].mean(), 3), " and the standard deviation was ", round(avg_happiness_low[0].std(),3))
print("\nThe calculated p-value from the t-test was", p_val, end="")
if p_val < 0.05:
    print(" which is sufficient to reject the null hypothesis.")
    print("Therefore, the t-test suggests that low proportion smartphone users are happier than high proportion smartphone users.\n")
else:
    print(" which is not sufficient to reject the null hypothesis.")

# there is a possibility the data is ordinal, even after the mean of all wellbeing scores are calculated. 
# A Mann-Whitney u-test will be performed on the data to test the null hypothesis
p_val = round(sc.mannwhitneyu(avg_happiness_high, avg_happiness_low, alternative='less').pvalue[0], 3)

# display results
print("The calculated p-value from the u-test was", p_val, end="")
if p_val < 0.05:
    print(" which is sufficient to reject the null hypothesis.")
    print("Therefore, the Mann-Whitney u-test suggests that low proportion smartphone users are happier than high proportion smartphone users.\n")
else:
    print(" which is not sufficient to reject the null hypothesis.\n")
    
# we can now explore the specific wellbeing indicators to see if there are any areas where high smartphone users perform well.
print("A u-test is performed for each specific metric comparing high and low smartphone users.")
for column in high_users_df.columns:
    # Perform the Mann-Whitney U test
    result = sc.mannwhitneyu(high_users_df[column], low_users_df[column], alternative='less')
    if result.pvalue > 0.05:
        print("The", column, "wellbeing metric obtained a p-score of", result.pvalue)
        print("This indicates inadequate evidence to claim that high smartphone users have lower", column, "metric scores than low smartphone users.\n")