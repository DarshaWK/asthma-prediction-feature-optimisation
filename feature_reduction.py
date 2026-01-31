#%% Import libraries
import numpy as np
import pandas as pd
from numpy import mean
from scipy.stats import sem
from sklearn.model_selection import cross_val_score,train_test_split,cross_validate, StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay,roc_auc_score,roc_curve,confusion_matrix,classification_report, auc, f1_score
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.inspection import permutation_importance
import seaborn as sns
from scipy import interp
from pylab import rcParams
import time, pickle, os
from pathlib import Path
from joblib import dump,load
from scikitplot.metrics import plot_roc_curve
from matplotlib import pyplot as plt
import re #regular expressions module
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE, RFECV

### Import Data ###
data_path = r"\asthma_attack_risk_prediction\data"
train_data = pd.read_csv(os.path.join(data_path,"DerivationSet_AsthmaPatients_Q5.csv")) 
test_data = pd.read_csv(os.path.join(data_path,"ValidationSet_AsthmaPatients_Q5.csv"))

#Iteration 1
train_data = pd.read_csv(os.path.join(data_path,"DerivationSet_AsthmaPatients_12YearsAbove_Quarter5_NumericFeatures_NoBins.csv"))
test_data = pd.read_csv(os.path.join(data_path,"ValidationSet_AsthmaPatients_12YearsAbove_Quarter5_NumericFeatures_NoBins.csv"))

#Clinically important features
train_data = train_data[['Q5_WeeksDuringWinter', 'Age', 'Gender', 'EthnicGroup', 'DeprivationQuintile','SmokingStatus', 'SABA_ICS_Ratio', 'P12MNoOfAsthAttacks', 'NoOfOPEDVisits', 'NoOfHospitalisations', 'Paracetamol', 'NSAIDs', 'NumberOfMetforminRx', 'CharlsonComorbidityScore_12Max', 'AllergyHistory','Cardiovascular','AsthmaAttack_Q5']]
test_data = test_data[['Q5_WeeksDuringWinter', 'Age', 'Gender', 'EthnicGroup', 'DeprivationQuintile', 'SmokingStatus', 'SABA_ICS_Ratio', 'P12MNoOfAsthAttacks', 'NoOfOPEDVisits', 'NoOfHospitalisations', 'Paracetamol', 'NSAIDs', 'NumberOfMetforminRx', 'CharlsonComorbidityScore_12Max', 'AllergyHistory','Cardiovascular','AsthmaAttack_Q5']]

train_data.dtypes

#Create age bins before data preprocessing (bcz KBinsDiscretizer does not produce meaningful bin names)
bins = [12, 15, 45, 65, 105]
bin_labels = ['12-14', '15-44', '45-64', '65andPlus']
   #Bin the data
train_data['Age_Binned'] = pd.cut(train_data['Age'], bins=bins, labels=bin_labels, right=False)
test_data['Age_Binned'] = pd.cut(test_data['Age'], bins=bins, labels=bin_labels, right=False)
#Create one-hot encoded features for the binned age
one_hot_encoded_age = pd.get_dummies(train_data['Age_Binned'], prefix='Age')
one_hot_encoded_age_test = pd.get_dummies(test_data['Age_Binned'], prefix='Age')
#Concatenate the columns to the main dataframe
train_data = pd.concat([train_data, one_hot_encoded_age], axis=1)
test_data = pd.concat([test_data, one_hot_encoded_age_test], axis=1)
#Drop Age and Age_Binned columns
train_data.drop(columns=['Age', 'Age_Binned'], axis=1, inplace=True)
test_data.drop(columns=['Age', 'Age_Binned'], axis=1, inplace=True)
#Drop target variable
# train_data.drop(columns=['AsthmaAttack_Q5'], axis=1, inplace=True)

# Converting target variable to categorical
# train_data["AsthmaAttack_Q5"] = train_data["AsthmaAttack_Q5"].astype("category")

categorical_features = ['EthnicGroup'] #,'DHB']
#Iteration 1
# numeric_features = ['P12MNoOfAsthAttacks', 'Q5_WeeksDuringWinter','NoOfAsthmaControllerMeds']
#Iteration 2
numeric_features = ['P12MNoOfAsthAttacks', 'SABA_ICS_Ratio', 'Q5_WeeksDuringWinter','NoOfOPEDVisits', 'NoOfHospitalisations','NumberOfMetforminRx','CharlsonComorbidityScore_12Max']

ST_scalar = StandardScaler()
OH_Encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# data_test_normalisation = train_data.copy()

# transformer = Normalizer().fit(X=data_test_normalisation[numeric_features])
# transformer.transform(data_test_normalisation[numeric_features])

preprocessor = ColumnTransformer(
    transformers=[
         ('categorical',OH_Encoder,categorical_features),
        ('numeric',ST_scalar,numeric_features)
        ],
    remainder='passthrough'
    )

#%%  Splitting into Dependent and Independent variables
y_train = train_data["AsthmaAttack_Q5"]
X_train = train_data.drop("AsthmaAttack_Q5", axis=1)
y_test = test_data["AsthmaAttack_Q5"]
X_test = test_data.drop("AsthmaAttack_Q5", axis=1)

X_train.reset_index(drop=True,inplace=True)
y_train.reset_index(drop=True,inplace=True)
X_test.reset_index(drop=True,inplace=True)
y_test.reset_index(drop=True,inplace=True)

#%% Recursive Feature Elimination with RFECV
np.random.seed(42)
# Define base model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier # sklearn API

# model = LogisticRegression(max_iter=1000)
model = RandomForestClassifier()
# Use cross-validation to select optimal number of features
rfecv = RFECV(
    estimator=model,
    step=1,
    cv=StratifiedKFold(5),
    scoring='f1',  # or 'roc_auc', 'f1', etc.
    n_jobs=-1
)

# Fit on transformed data
rfecv.fit(X_transformed, y_train)

# Check results
print("Optimal number of features:", rfecv.n_features_)
print("Selected feature mask:", rfecv.support_)

feature_names_rfecv = get_feature_names_from_column_transformer(preprocessor, X_train.columns)
selected_feature_names = np.array(feature_names_rfecv)[rfecv.support_]

print("Optimal number of features selected:", rfecv.n_features_)
print("Selected feature names:\n", selected_feature_names)

#%% Recursive Elimination of Features RFECV PLOT
# Using Random Forest
from sklearn.ensemble import RandomForestClassifier
model_base = RandomForestClassifier(random_state=93186)

#Using XGB
from xgboost.sklearn import XGBClassifier # sklearn API
model_base = XGBClassifier(random_state=93186,
                                    n_jobs=-1,
                                    verbosity=1)
                                   # n_estimators=100, #1000
                                    #use_label_encoder=False) #for future changes

# Step 1: Preprocess data manually
X_transformed = preprocessor.fit_transform(X_train)
y_resampled = y_train  # Didn't apply RandomUnderSampler here (explained below)

# Optional: if X_transformed becomes a numpy array (e.g., from OneHotEncoder), convert to DataFrame
if not isinstance(X_transformed, pd.DataFrame):
    # Get output feature names (from transformers)
    try:
        feature_names = preprocessor.get_feature_names_out()
    except:
        feature_names = [f"f{i}" for i in range(X_transformed.shape[1])]
    X_transformed = pd.DataFrame(X_transformed, columns=feature_names)

# Step 2: Apply RFECV on just the model
cv = StratifiedKFold(n_splits=5,shuffle=True, random_state=42)
rfecv = RFECV(estimator=model_base, step=1, cv=cv, scoring='roc_auc', n_jobs=-1)

rfecv.fit(X_transformed, y_resampled)

# Step 3: Get selected features
selected_mask = rfecv.support_
selected_features = X_transformed.columns[selected_mask]
# All feature names after preprocessing
all_features = X_transformed.columns
removed_features = all_features[~selected_mask]

print(f"Optimal number of features: {rfecv.n_features_}")
print("Selected features:", selected_features.tolist())
#Optimal number of features: 58 (with the OHE features - but still it's a lot)
#Optimal number of features: 48(with the OHE features - but still it's a lot)
print("\n Removed Features:")
print(removed_features.tolist())

# toc = time.perf_counter()

plt.figure(figsize=(8,6))
plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1),
         rfecv.cv_results_['mean_test_score'], marker='o')
plt.axvline(rfecv.n_features_, color='red', linestyle='--', label=f'Optimal = {rfecv.n_features_}')
plt.xlabel("Number of features")
plt.ylabel("Mean CV score (AUROC)")
plt.title("RFECV Feature Selection with XGB")
plt.legend()
plt.grid()
plt.show()










