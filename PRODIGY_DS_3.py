import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

df=pd.read_csv('G:/Pratik Profile PC/Desktop/PRODIGY INFOTECH INTERNSHIP TASKS/TASK_03/bank-full.csv', sep=';')
df

df.head(10)

df.isna().sum()

df.dropna(inplace=True)
df_1=df.drop_duplicates()
df_1.info

# Preprocess the data
X = df.drop('poutcome', axis=1)
y = df['poutcome']
X = pd.get_dummies(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the classifier
clf = DecisionTreeClassifier()

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Visualize the decision tree
plt.figure(figsize=(10, 8))
plot_tree(clf, feature_names=list(X.columns), class_names=df['education'].unique().tolist(), filled=True, rounded=True)
plt.show()

# Create the classifier with pruning enabled
clf = DecisionTreeClassifier(ccp_alpha=0.01)

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Visualize the pruned decision tree
plt.figure(figsize=(10, 8))
plot_tree(clf, feature_names=list(X.columns), class_names=df['education'].unique().tolist(), filled=True, rounded=True)
plt.show()
