import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use("ggplot")  



import plotly.graph_objects as go
import plotly.express as px


df=pd.read_csv('C:/6105/final-project/winemag-data_first150k.csv')
df.head()

print("Shape The DataSet ", df.shape )

# define the size of figures that I will build
plt.figure(figsize=(16,5))

g = sns.countplot(x='points', data=df, color = sns.xkcd_rgb['wine'])
g.set_title("Points Count distribuition ", fontsize=20) 
g.set_xlabel("Points") 
g.set_ylabel("Count")

df['log_price'] = np.log(df['price'])
plt.figure(figsize=(15, 10))
sns.boxenplot(data = df, x="points", y="log_price", k_depth="trustworthy")

corr = df[['log_price', 'points']].corr()
sns.heatmap(corr, annot = True, vmin=-1, vmax=1, center= 0, cmap= 'coolwarm')


import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

"""**READ DATASET**"""

# Read the three datasets into separate dataframes
df = pd.read_csv('/winedataset.csv')

df.shape

"""**CHECK DATASET FOR MISSING VALUES**"""

df.isnull().sum().sort_values(ascending=False)

"""**CHECK DATASET FOR UNIQUE FEATURES**"""

feat_uniq= {}
for col in df.columns:
    feat_uniq[col] =len(df[col].unique())

feat_uniq

retain = []
for col in df.columns:
    if len(df[col].unique()) > 1:
        retain.append(col)

new_df = df.loc[:, retain]
new_df.loc[:, retain]

new_df.isnull().sum().sort_values(ascending=False)

"""**CHECK CORRELATION OF FEATURES WITH CORRELATION MATRIX**"""

import matplotlib.pyplot as plt
import seaborn as sns

# Check the column names
print(new_df.columns)

# Calculate the correlation matrix
corr_matrix = new_df.corr()

# Plot the correlation matrix as a heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

# Identify the feature with the highest correlation to the target variable
target_var = 'points'
if target_var in new_df.columns:
    target_corr = corr_matrix[target_var]
    best_feature = target_corr.abs().idxmax()
    second_max = target_corr.drop(best_feature).abs().idxmax()
    print('The best feature is:', second_max)
else:
    print(f"The target variable '{target_var}' is not in the dataset.")

"""**DROP IRRELEVANT FEATURES**"""

new_df = new_df.drop(['SN', 'country', 'description', 'designation', 'province', 'region_1', 'region_2', 'variety', 'winery'], axis=1)
new_df.head()

new_df.isnull().sum().sort_values(ascending=False)

sns.heatmap(new_df.isnull(),cmap = 'magma', cbar = False)

new_df.shape

"""**DROP ROWS WITH MISSING VALUES**"""

# Drop rows with missing values
new_df = new_df.dropna()

# Print the new shape of the dataset
print(new_df.shape)

new_df.head()

new_df = new_df.sample(n=11000)
new_df.shape

new_df.head()

"""**HISTOGRAM OF THE TARGET COLUMN "POINTS"**"""

new_df.hist(column = 'points')
print()

"""**SPLIT DATA INTO TRAIN AND TEST**"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

# Preprocess the data
X = new_df.drop(['points'], axis=1)
y = new_df['points']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""**RUN XG-BOOST REGRESSION MODEL AND LEARNING CURVE**"""

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_regression
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve


# create an XGBoost regressor object
xgb_model = xgb.XGBRegressor()

# set parameter grid for grid search
params = {
    'max_depth': [3, 5, 7],
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.5]
}

# perform grid search to find best hyperparameters
grid = GridSearchCV(estimator=xgb_model, param_grid=params, cv=5)
grid.fit(X_train, y_train)

# predict on test set using best estimator from grid search
y_pred = grid.predict(X_test)

# calculate accuracy score
acc = accuracy_score(y_test, y_pred.round())
print("Accuracy:", acc)

# plot learning curve
train_sizes, train_scores, test_scores = learning_curve(
    estimator=grid.best_estimator_, X=X_train, y=y_train, cv=5, n_jobs=-1, 
    train_sizes=np.linspace(0.1, 1.0, 10), scoring='neg_mean_squared_error')
train_scores_mean = np.mean(np.sqrt(-train_scores), axis=1)
train_scores_std = np.std(np.sqrt(-train_scores), axis=1)
test_scores_mean = np.mean(np.sqrt(-test_scores), axis=1)
test_scores_std = np.std(np.sqrt(-test_scores), axis=1)

plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross-validation score')
plt.fill_between(train_sizes, train_scores_mean - train_scores_std, 
                 train_scores_mean + train_scores_std, alpha=0.1, color='r')
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, 
                 test_scores_mean + test_scores_std, alpha=0.1, color='g')
plt.xlabel('Training examples')
plt.ylabel('Root Mean Squared Error')
plt.legend(loc='best')
plt.show()

"""**RUN DECISION TREE REGRESSION MODEL AND LEARNING CURVE**"""

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Define decision tree regressor
tree_reg = DecisionTreeRegressor(random_state=42)

# Define hyperparameters to tune
params = {
    'max_depth': [2, 4, 6, 8],
    'min_samples_split': [2, 4, 6, 8],
    'min_samples_leaf': [1, 2, 3, 4, 5]
}

# Perform grid search to find best hyperparameters
grid_search = GridSearchCV(tree_reg, params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Print best hyperparameters
print("Best hyperparameters for Decision Tree Regressor: ", grid_search.best_params_)

# Fit model with best hyperparameters
best_tree_reg = DecisionTreeRegressor(**grid_search.best_params_, random_state=42)
best_tree_reg.fit(X_train, y_train)

# Make predictions on test set
y_pred = best_tree_reg.predict(X_test)

# Calculate accuracy metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Print accuracy metrics
print("RMSE for Decision Tree Regressor: ", rmse)
print("R2 Score for Decision Tree Regressor: ", r2)

# Plot learning curve
train_sizes, train_scores, test_scores = learning_curve(best_tree_reg, X_train, y_train, cv=5)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure(figsize=(8, 6))
plt.title("Decision Tree Regressor Learning Curve")
plt.xlabel("Training Examples")
plt.ylabel("Score")
plt.grid()

plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training Score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-Validation Score")
plt.legend(loc="best")

# Get feature importance
importances = best_tree_reg.feature_importances_
indices = np.argsort(importances)[::-1]
print("Feature ranking for Decision Tree Regressor:")
for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

"""**RUN RANDOM FOREST REGRESSION MODEL AND LEARNING CURVE**"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
import numpy as np
import matplotlib.pyplot as plt

# Load your dataset here and split into X (features) and y (target variable)

# Define the parameter grid for grid search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7]
}

# Initialize the random forest regressor model
rf = RandomForestRegressor(random_state=42)

# Perform grid search to find the best hyperparameters
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5)
grid_search.fit(X, y)

# Get the best hyperparameters and fit the model on the full dataset
best_params = grid_search.best_params_
rf = RandomForestRegressor(n_estimators=best_params['n_estimators'], max_depth=best_params['max_depth'], random_state=42)
rf.fit(X, y)

# Calculate the accuracy of the model
accuracy = rf.score(X, y)
print("Accuracy:", accuracy)

# Plot the learning curve
train_sizes, train_scores, test_scores = learning_curve(rf, X, y, cv=5)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
plt.plot(train_sizes, train_mean, label="Training score")
plt.plot(train_sizes, test_mean, label="Cross-validation score")
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)
plt.xlabel("Training Size")
plt.ylabel("Score")
plt.legend(loc="best")
plt.show()

# Get the feature importances
importances = rf.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)[::-1]
print("Feature Importances:")
for i in range(X.shape[1]):
    print(f"{i+1}. {feature_names[indices[i]]} ({importances[indices[i]]})")




