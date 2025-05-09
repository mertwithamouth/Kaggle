import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sqlalchemy import column
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


df_=pd.read_csv('GreenHouseProject/Greenhouse Plant Growth Metrics.csv')
df=df_.copy()

# Display basic info
df.info()

#Checking missing values
df.isnull().sum()




#Data Distribution For Numeric Variables
columns=df.columns.drop(["Random","Class"])
fig, axes = plt.subplots(4, 3, figsize=(12, 8))
axes = axes.flatten()
for index,col in enumerate(columns):
    sns.histplot(df,x=col,ax=axes[index],kde=True)
    axes[index].set_title(col)
plt.tight_layout()
plt.show()

# Data Distribution of the 'Class'
plt.figure(figsize=(6,4))
sns.countplot(x='Class', data=df)
plt.title('Distribution of the Class Variable')
plt.xlabel('Class')
plt.ylabel('Count')
plt.tight_layout()
plt.show()


#Heatmap to see Correlation
plt.figure(figsize=(10,8))
sns.heatmap(df[columns].corr(),annot=True, cmap='coolwarm',fmt='.2f')
plt.title('Correlation Heatmap of Numeric Features')
plt.tight_layout()
plt.show()

X=df.drop(columns=['Class', 'Random'])
y=df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#RansomForest Modeli
rf=RandomForestClassifier(n_estimators=100,random_state=42)
rf.fit(X_train,y_train)

y_pred=rf.predict(X_test)

# Plot the confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

print("Doğruluk Oranı:", accuracy_score(y_test, y_pred))


test_data = np.array([
    34.53346785732356,
    54.56698291488631,
    1.1474490163213231,
    1284.2295490809165,
    4.999713080337564,
    16.274917909603804,
    1.7068098312939444,
    18.3999815454843,
    19.739037367484507,2.943137,0.216154,57.633697
])
test_df = pd.DataFrame([test_data], columns=X.columns)

rf.predict(test_df)


#XGBoost Modeli

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)


xgb_model= xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

xgb_model.fit(X_train,y_train)

y_pred=xgb_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Doğruluk: {accuracy:.2f}")

# Plot the confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()


xgb_model.predict(test_df)