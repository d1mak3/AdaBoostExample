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
#plt.show()

# Evaluating the model
print('R-square score is :', r2_score(y_test, AdaBoostR_pred))

# initializing the model
model = AdaBoostRegressor()

# applying GridSearchCV
grid = GridSearchCV(estimator=model,param_grid={'n_estimators':range(1,50)})

# training the model
grid.fit(x_train,y_train)

# printing the best estimator
print("The best estimator returned by GridSearch CV is:", grid.best_estimator_)

# cutting value from the string
estimator = grid.best_estimator_.__str__()[31:len(grid.best_estimator_.__str__()) - 1]

# Create adaboost regressor with default parameters
Ada_regressor4 = AdaBoostRegressor(n_estimators=2)

# Train Adaboost Classifer
AdaBoost_R4 = Ada_regressor4.fit(x_train, y_train)

#Predict price of houses
AdaBoostR_pred4 = AdaBoost_R4.predict(x_test)

# Evaluating the model
print('R score is :', r2_score(y_test, AdaBoostR_pred4))