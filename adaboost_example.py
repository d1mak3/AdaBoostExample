from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pandas as pd

# importing the dataset
house = pd.read_csv("Dushanbe_house.csv")

# head() method
print(house.head())

# total null values
print(house.isnull().sum())

# removing the null values
house = house.dropna(axis=0, )

# null values
print(house.isnull().sum())

# splitting dataset
x_data = house.drop('price', axis=1)
y_data = house.price

# splitting the dataset
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=0)

# Create adaboost regressor with default parameters
Ada_regressor = AdaBoostRegressor()

# Train Adaboost Classifer
AdaBoost_R = Ada_regressor.fit(x_train, y_train)

#Predict price of houses
AdaBoostR_pred = AdaBoost_R.predict(x_test)

# fitting the size of the plot
plt.figure(figsize=(20, 8))

# plotting the graphs
plt.plot([i for i in range(len(y_test))], AdaBoostR_pred, label="Predicted values")
plt.plot([i for i in range(len(y_test))], y_test, label="actual values")
plt.legend()
plt.show()

# Create adaboost regressor with default parameters
Ada_regressorE = AdaBoostRegressor(n_estimators=4)

# Train Adaboost Classifer
AdaBoost_RE = Ada_regressorE.fit(x_train, y_train)

#Predict price of houses
AdaBoostR_predE = AdaBoost_RE.predict(x_test)

plt.close()
plt.plot([i for i in range(len(y_test))], AdaBoostR_predE, label="Predicted values")
plt.plot([i for i in range(len(y_test))], y_test, label="actual values")
plt.legend()
plt.show()