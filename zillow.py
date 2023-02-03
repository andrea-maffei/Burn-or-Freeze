import matplotlib.pyplot as plt
import numpy as np
import pandas as pd#important update

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_error


#LOAD & DESCRIBE
df = pd.read_csv(
    "hw-City_zhvi_bdrmcnt_1_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv")
df_miami = df[df['RegionName'] == 'Miami']

# Calculate mean for each column
mean_miami = df_miami.mean()

# Filter out the non-date columns
date_columns = df_miami.columns[pd.to_datetime(
    df_miami.columns, errors='coerce').notna()]
mean_miami = mean_miami[date_columns]

# Plot the trendline
plt.plot(mean_miami)
plt.xlabel('Date')
plt.ylabel('Mean Value')
plt.title('Mean Value Trendline for Miami')
plt.yticks(np.linspace(min(mean_miami), max(mean_miami), 10))
plt.rcParams["figure.figsize"] = (15, 15)
plt.show()

# Calculate the moving average with a window size of 10
moving_average = mean_miami.rolling(window=10).mean()

# Plot the trendline and moving average
plt.plot(mean_miami, label='Mean Value')
plt.plot(moving_average, label='Moving Average', linewidth=2)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Mean Value', fontsize=14)
plt.title('Mean Value Trendline for Miami', fontsize=16)
plt.legend(loc='best')
plt.rcParams["figure.figsize"] = (15, 15)
plt.show()


# MANAGE
le = LabelEncoder()
df["RegionName"] = le.fit_transform(df["RegionName"])
df["StateName"] = le.fit_transform(df["StateName"])
df["State"] = le.fit_transform(df["State"])

df.head()

df.fillna(df.mean(), inplace=True)

df.describe()

# SPLIT
# Create X and y arrays
X = df[["RegionName", "State", "SizeRank"]]
y = df.drop(["RegionID", "RegionType", "RegionName", "State",
            "Metro", "CountyName", "SizeRank"], axis='columns')

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

label_encoder = preprocessing.LabelEncoder()
X['RegionName'] = label_encoder.fit_transform(X['RegionName'])
X['State'] = label_encoder.fit_transform(X['State'])

# LINEAR REG
# Instantiate the model
reg = LinearRegression()

# Fit the model to the data
reg.fit(X_train, y_train)

# Make predictions
y_pred = reg.predict(X_test)
print("Predictions: {}, Actual Values: {}".format(y_pred[:2], y_test[:2]))

# Compute R-squared
r_squared = reg.score(X_test, y_test)
print("R^2: {}".format(r_squared))
