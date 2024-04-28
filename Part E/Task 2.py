import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load your dataset
file_path = 'Data/FuelConsumption.csv'
data = pd.read_csv(file_path)

# Specify columns to drop and the target column
columns_to_drop = ['Year', 'MODEL', 'VEHICLE CLASS', 'TRANSMISSION', 'FUEL']
target_column = 'ENGINE SIZE'

# Drop specified columns
data.drop(columns=[col for col in columns_to_drop if col in data.columns], axis=1, inplace=True)

# Assuming ENGINE SIZE needs to be categorical for classification
data[target_column] = pd.qcut(data[target_column], q=4, labels=False)  # Quantile-based discretization

# Identify categorical columns for encoding
categorical_cols = data.select_dtypes(include=['object']).columns.tolist()

# Preprocessing: Apply OneHotEncoder to categorical data
column_transformer = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), categorical_cols)],remainder='passthrough')

X = column_transformer.fit_transform(data.drop(target_column, axis=1))
y = data[target_column]

# Split the dataset into training and test sets (80% training, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Calculate total number of data objects
total_samples = X.shape[0]
num_train_samples = X_train.shape[0]
num_test_samples = X_test.shape[0]


# Count occurrences of each class in the training and test sets
class_counts_train = y_train.value_counts()
class_counts_test = y_test.value_counts()

# Define function to perform experiments for Logistic Regression
def logistic_regression_experiments(X_train, X_test, y_train, y_test):
    # Experiment 1
    lr_model_1 = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')  
    lr_model_1.fit(X_train, y_train)
    y_pred_lr_1 = lr_model_1.predict(X_test)
    acc_lr_1 = accuracy_score(y_test, y_pred_lr_1)

    # Experiment 2
    lr_model_2 = LogisticRegression(max_iter=1000, C=0.5, solver='lbfgs')  
    lr_model_2.fit(X_train, y_train)
    y_pred_lr_2 = lr_model_2.predict(X_test)
    acc_lr_2 = accuracy_score(y_test, y_pred_lr_2)

    # Experiment 3
    lr_model_3 = LogisticRegression(max_iter=1000, C=0.1, solver='lbfgs') 
    lr_model_3.fit(X_train, y_train)
    y_pred_lr_3 = lr_model_3.predict(X_test)
    acc_lr_3 = accuracy_score(y_test, y_pred_lr_3)

    return [acc_lr_1, acc_lr_2, acc_lr_3]

# Define function to perform experiments for Decision Trees
def decision_tree_experiments(X_train, X_test, y_train, y_test):
    # Experiment 1
    dt_model_1 = DecisionTreeClassifier(max_depth=1)  
    dt_model_1.fit(X_train, y_train)
    y_pred_dt_1 = dt_model_1.predict(X_test)
    acc_dt_1 = accuracy_score(y_test, y_pred_dt_1)

    # Experiment 2
    dt_model_2 = DecisionTreeClassifier(max_depth=2)  
    dt_model_2.fit(X_train, y_train)
    y_pred_dt_2 = dt_model_2.predict(X_test)
    acc_dt_2 = accuracy_score(y_test, y_pred_dt_2)

    # Experiment 3
    dt_model_3 = DecisionTreeClassifier(max_depth=5)  
    dt_model_3.fit(X_train, y_train)
    y_pred_dt_3 = dt_model_3.predict(X_test)
    acc_dt_3 = accuracy_score(y_test, y_pred_dt_3)

    return [acc_dt_1, acc_dt_2, acc_dt_3]


def main():
    print("Total number of data objects:")
    print("Training set: {} samples ({}%)".format(num_train_samples, (num_train_samples / total_samples) * 100))
    print("Test set: {} samples ({}%)".format(num_test_samples, (num_test_samples / total_samples) * 100))

    # Print class distribution in training and test sets
    print("\nClass Distribution:")
    print("Training Set:")
    print(class_counts_train)
    print("Percentage:")
    print((class_counts_train / num_train_samples) * 100)

    print("\nTest Set:")
    print(class_counts_test)
    print("Percentage:")
    print((class_counts_test / num_test_samples) * 100)

    # Logistic Regression experiments
    lr_results = logistic_regression_experiments(X_train, X_test, y_train, y_test)
    print("\nLogistic Regression Results:")
    for i, acc in enumerate(lr_results):
        print(f"Experiment {i+1}: Accuracy = {acc}")

    # Decision Tree experiments
    dt_results = decision_tree_experiments(X_train, X_test, y_train, y_test)
    print("\nDecision Tree Results:")
    for i, acc in enumerate(dt_results):
        print(f"Experiment {i+1}: Accuracy = {acc}")

if __name__ == "__main__":
    main()