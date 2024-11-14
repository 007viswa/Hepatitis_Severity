# %%
import pandas as pd 
import numpy as np

# %%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# %%
from ucimlrepo import fetch_ucirepo 

# %%
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/hepatitis/hepatitis.data"
column_names = ['Class', 'Age', 'Sex', 'Steroid', 'Antivirals', 'Fatigue', 'Malaise', 'Anorexia',
                'LiverBig', 'LiverFirm', 'SpleenPalpable', 'Spiders', 'Ascites', 'Varices', 
                'Bilirubin', 'AlkPhosphate', 'Sgot', 'Albumin', 'Protime', 'Histology']
# # data (as pandas dataframes) 
# X = hepatitis.data.features 
# y = hepatitis.data.targets 
  
# # metadata 
# print(hepatitis.metadata) 
  
# # variable information 
# print(hepatitis.variables) 

# %%
data = pd.read_csv(url, names=column_names)

# %%
data.replace('?', np.nan, inplace=True)
data.dropna(inplace=True)

# %%
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le


# %%
X = data.drop('Class', axis=1)
y = data['Class']

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# %%
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# %% [markdown]
# RandomForest

# %%
Rn_model = RandomForestClassifier(n_estimators=100, random_state=42)
Rn_model.fit(X_train, y_train)
y_pred_Rn = Rn_model.predict(X_test)
Rn_accuracy = accuracy_score(y_test,y_pred_Rn)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_Rn))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_Rn))
print("\nRandom Forest Accuracy Score:", Rn_accuracy)

# %% [markdown]
# KNN

# %%
knn_model = KNeighborsClassifier(n_neighbors=5)  # You can experiment with different values of k
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)
knn_accuracy = accuracy_score(y_test, y_pred_knn)
print("KNN Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_knn))
print("\nKNN Classification Report:")
print(classification_report(y_test, y_pred_knn))
print("\nKNN Accuracy Score:", knn_accuracy)

# %% [markdown]
# LogisticRegression

# %%
logreg_model = LogisticRegression(random_state=42, max_iter=1000)  # Increase max_iter if convergence issues occur
logreg_model.fit(X_train, y_train)
y_pred_logreg = logreg_model.predict(X_test)
logreg_accuracy = accuracy_score(y_test, y_pred_logreg)
print("\nLogistic Regression Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_logreg))
print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, y_pred_logreg))
print("\nLogistic Regression Accuracy Score:", logreg_accuracy)

# %%
# Check unique values in the target variable before and after encoding
print("Unique values in y before encoding:", data['Class'].unique())



# %%



