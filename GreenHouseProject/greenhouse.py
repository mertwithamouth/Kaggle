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


#Variable type seperating function
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    # print(f"Observations: {dataframe.shape[0]}")
    # print(f"Variables: {dataframe.shape[1]}")
    # print(f'cat_cols: {len(cat_cols)}')
    # print(f'num_cols: {len(num_cols)}')
    # print(f'cat_but_car: {len(cat_but_car)}')
    # print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def check_outlier(dataframe, col_name, q1=0.25, q3=0.75):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False



df_=pd.read_csv('GreenHouseProject/Greenhouse Plant Growth Metrics.csv')
df=df_.copy()

# Display basic info
df.info()

#Checking missing values
df.isnull().sum()

# Seperating Variable Type Columns
cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=5, car_th=20)

#Checking outliers
for col in num_cols:
    print(col, check_outlier(df, col, 0.05, 0.95))

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