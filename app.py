import numpy as np
import streamlit as st
import polars as pl
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import plotly.express as px

import pickle

MODEL_PATH = "data/log_reg.model"
DATASET_PATH = "data/heart_2020_final.csv"

def write_log(msg, prefix="process"):
    f = open("log/data.log", "a+")
    f.write("[" + str(prefix) + "]\t" + str(msg) + "\n")
    f.flush()
    f.close()


# print("hello world! Im learning about you!")
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
data_csv = pd.read_csv(DATASET_PATH)
data_csv.info()

le = LabelEncoder()
col = data_csv[['HeartDisease', 'Smoking', 'AlcoholDrinking', 'AgeCategory', 'Stroke', 'DiffWalking',
                'Race', 'Sex', 'Asthma', 'KidneyDisease', 'SkinCancer', 'Diabetic']]
write_log("\n"+str(data_csv.head()))

""" for i in col:
    data_csv[i] = le.fit_transform(data_csv[i])
write_log("\n"+str(data_csv.head()))

num_cols = ['MentalHealth', "BMI", 'PhysicalHealth', 'SleepTime']
Scaler = StandardScaler()
data_csv[num_cols] = Scaler.fit_transform(data_csv[num_cols]) """

# correlations image
figures = px.imshow(data_csv.corr(), color_continuous_scale="Blues")
figures.update_layout(height=800)
# if you like to see the color corr matrix
#figures.show()

corr_matrix = data_csv.corr()
corr_matrix["HeartDisease"].sort_values(ascending=False)
write_log("\n"+str(corr_matrix), "correlations matrix")


X = data_csv.drop(columns=['HeartDisease'], axis=1)
Y = data_csv['HeartDisease']


X_train, x_test, y_train, y_test = train_test_split(
    X, Y, shuffle=True, test_size=.2, random_state=42)
y_train.value_counts()

ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled, = ros.fit_resample(X_train, y_train)
y_train_resampled.value_counts()


LR = LogisticRegression()
LR.fit(X_train_resampled, y_train_resampled)

y_pred_6 = LR.predict(x_test)

print(classification_report(y_test, y_pred_6))
sns.set_context("poster")
dispo = plot_confusion_matrix(LR, x_test, y_test, colorbar=False)

pickle.dump(LR, open(MODEL_PATH, 'wb'))


def get_dataframe(bmi_cat, smoking, alcohol_drink, stroke, phys_health, ment_health, diff_walk, sex, age_cat, race, diabetic, sleep_time, asthma, kid_dis, skin_canc):
    features = pd.DataFrame({
        "BMI": [bmi_cat],
        "Smoking": [smoking],
        "AlcoholDrinking": [alcohol_drink],
        "Stroke": [stroke],
        "PhysicalHealth": [phys_health],
        "MentalHealth": [ment_health],
        "DiffWalking": [diff_walk],
        "Sex": [sex],
        "AgeCategory": [age_cat],
        "Race": [race],
        "Diabetic": [diabetic],
        "SleepTime": [sleep_time],
        "Asthma": [asthma],
        "KidneyDisease": [kid_dis],
        "SkinCancer": [skin_canc]
    })
    return features


@st.cache(persist=True)
def load_dataset(path: str) -> pd.DataFrame:
    heart_df = pl.read_csv(path)
    heart_df = heart_df.to_pandas()
    heart_df = pd.DataFrame(np.sort(heart_df.values, axis=0),
                            index=heart_df.index,
                            columns=heart_df.columns)
    return heart_df


dataset = load_dataset(DATASET_PATH)

input_dataframe = get_dataframe(
    3.0, 1, 0, 0, 30, 0, 1, 1, 8, 0, 1, 15, 1, 0, 0)
df = pd.concat([input_dataframe, dataset], axis=0)
df = df.drop(columns=["HeartDisease"])
write_log("\n"+str(df.head()))

df = df[:1]
df.fillna(0, inplace=True)
write_log("\n"+str(df.head()))
log_model = pickle.load(open(MODEL_PATH, "rb"))

prediction = log_model.predict(df)
prediction_prob = log_model.predict_proba(df)
print(str(prediction))
if prediction == 0:
    value = round(prediction_prob[0][1] * 100, 2)
else:
    value = round(prediction_prob[0][1] * 100, 2)
print("probability that you will have a heart disease is "+str(value)+" %")
