import os
import tarfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import hashlib
from six.moves import urllib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"


# gets the data from the root and puts it in the dataset directory
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
        tgz_path = os.path.join(housing_path, "housing.tgz")
        urllib.request.urlretrieve(housing_url, tgz_path)
        housing_tgz = tarfile.open(tgz_path)
        housing_tgz.extractall(path=housing_path)
        housing_tgz.close()


# reads the data in dataset/housing
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


housing = load_housing_data()
housing.head()

ocean_counts = housing["ocean_proximity"].value_counts()
housing_description = housing.describe()


# housing.hist(bins=50, figsize=(20, 15))
# plt.show()

# splits the data into a test and train set
# This is good except that over time the test set will be part of the train set and we do not want that
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


# train_set, test_set = split_train_test(housing, 0.2)
# print(len(train_set), "train +", len(test_set), "test")


def test_set_check(identifier, test_ratio, hash_data):
    return hash_data(np.int64(identifier)).digest()[-1] < 256 * test_ratio


# This works in making sure the test set never
# becomes part of the train data and any updated columns will be separated as well
# and is essentially the same as train_test_split from scikit-learn
def split_train_test_by_id(data, test_ratio, id_column, hash_data=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash_data))
    return data.loc[~in_test_set], data.loc[in_test_set]


# housing_with_id = housing.reset_index()  # adds an 'index' column
# train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

# in the case where income is the most important factor
# this splits the train and test set and makes sure that the income category in both sets are relatively similar
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

print("=====================income_category===========================")
print(housing["income_cat"].value_counts() / len(housing))
print("===============================================================")
# deletes income_cat from both the train and test set
for set in (strat_train_set, strat_test_set):
    set.drop(["income_cat"], axis=1, inplace=True)

housing = strat_train_set.copy()

housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
             s=housing["population"] / 100, label="population",
             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
             )
plt.legend()

corr_matrix = housing.corr()
print("=====================correlations===========================")
print(corr_matrix["median_house_value"].sort_values(ascending=False))
print("============================================================")

attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))

housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["populations_per_household"] = housing["population"] / housing["households"]

corr_matrix = housing.corr()
print("=====================correlations_with_added_columns===========================")
print(corr_matrix["median_house_value"].sort_values(ascending=False))
print("===============================================================================")

# clean training set since we added more columns than necessary
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

median = housing["total_bedrooms"].median()
housing["total_bedrooms"].fillna(median)
imputer = SimpleImputer(strategy="median")
housing_num = housing.drop("ocean_proximity", axis=1)
imputer.fit(housing_num)
print("statistics")
print(imputer.statistics_)

# puts all of the median values in each column so that there are no null values
transformed_housing = imputer.transform(housing_num)
housing_tr = pd.DataFrame(transformed_housing, columns=housing_num.columns)

# ALL OF THIS COMMENTED OUT ESSENTIALLY DOES THE SAME THING
# need to convert strings to numbers
# encoder = LabelEncoder()
# housing_cat = housing["ocean_proximity"]
# housing_cat_encoded = encoder.fit_transform(housing_cat)
# print("encoded ocean_proximity")
# print(housing_cat_encoded)
# print("encoded values")
# print(encoder.classes_)

# encoder = OneHotEncoder(categories="auto")
# housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1, 1))
# print("ocean_proximity one hot encoded")
# print(housing_cat_1hot.toarray())

encoder = LabelBinarizer()
housing_cat = housing["ocean_proximity"]
housing_cat_1hot = encoder.fit_transform(housing_cat)

print("ocean_proximity one hot encoded")
print(housing_cat_1hot)

rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):  # no *args for **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, x, y=None):
        return self  # nothing else to do

    def transform(self, x, y=None):
        rooms_per_household = x[:, rooms_ix] / x[:, household_ix]
        population_per_household = x[:, population_ix] / x[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = x[:, bedrooms_ix] / x[:, rooms_ix]
            return np.c_[x, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[x, rooms_per_household, population_per_household]


att_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = att_adder.transform(housing.values)

# This is almost good but we want to make it so you can do the number pipeline and the category one at the same time.

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attrib_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler())
])

housing_num_tr = num_pipeline.fit_transform(housing_num)


def add_extra_features(x, add_bedrooms_per_room=True):
    rooms_per_household = x[:, rooms_ix] / x[:, household_ix]
    population_per_household = x[:, bedrooms_ix] / x[:, household_ix]
    if add_bedrooms_per_room:
        bedrooms_per_room = x[:, bedrooms_ix] / x[:, rooms_ix]
        return np.c_[x, rooms_per_household, population_per_household, bedrooms_per_room]
    else:
        return np.c_[x, rooms_per_household, population_per_household]


num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

att_adder = FunctionTransformer(add_extra_features, validate=False, kw_args={"add_bedrooms_per_room": False})
housing_extra_attribs = att_adder.fit_transform(housing.values)

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', FunctionTransformer(add_extra_features, validate=False)),
    ('std_scaler', StandardScaler()),
])

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])

housing_prepared = full_pipeline.fit_transform(housing)

# Everything is transformed and ready to train a ML model

# train a linear regression model
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)

print("mean squared error: ", lin_rmse)

# the results was not good. got a RMSE of about 68k when median is 120k - 265k
# this is underfitting. Caused by the features not providing enough info or the model is not good enough
# we are going to try using a decisionTreeRegressor to see if that fixes it

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)

housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
print("mean squared error for tree: ", tree_rmse)

# There is no error at all! obviously there must be something wrong. There might be some a lot of over fitting.
# could either train_test_split the training data into a small size and try again or use cross-validation from scikit

scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


#  tree scores
display_scores(tree_rmse_scores)

lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)

# looks like the tree was over fitting so badly that it was worse than linear regression.
# Lets try the RandomForestRegressor to see how well that works
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)
forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)

# save my model
joblib.dump(forest_reg, "my_forest_model.pkl")
# if i ever want it back use
my_model_loaded = joblib.load("my_forest_model.pkl")

# we went over going through models.
# Currently the forest is the best one but should still see if theres something even better.
# for now though will move to fine-tuning my model
#

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]

forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(housing_prepared, housing_labels)

# param_grid will first do 3x4 combinations then 2x3 combinations
# and each one will be trained 5 times making over 90 rounds of training.
# to get the best params use
print("print params")
print(grid_search.best_params_)

# will get the best estimator automatically
grid_search.best_estimator_

# to show feature importances
feature_importances = grid_search.best_estimator_.feature_importances_
extra_attribs = ["room_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_one_hot_attribs = list(encoder.classes_)
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
print("feature importances")
print(sorted(zip(feature_importances, attributes), reverse=True))


# test the system
final_model = grid_search.best_estimator_
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)

final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

