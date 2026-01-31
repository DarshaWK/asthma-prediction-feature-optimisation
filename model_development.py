# ---------------------- This file consists the codes used for prediction model development ----------------------- #
# %% Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score,train_test_split,cross_validate, StratifiedKFold
from xgboost.sklearn import XGBClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler, Normalizer, KBinsDiscretizer, FunctionTransformer
from imblearn.over_sampling import SMOTE,SMOTENC,BorderlineSMOTE
from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours,RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from collections import Counter
import time, pickle, os
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import KMeansSMOTE

#Import self-defined modules
import model_efficiency_evaluation as modefficienteval
import model_performance_evaluation as modperformeval
from scipy import stats
import textwrap

### Import Data ###
train_data = pd.read_csv("DerivationSet_AsthmaPatients_12YearsAbove_Quarter5_NumericFeatures_NoBins.csv")
test_data = pd.read_csv("ValidationSet_AsthmaPatients_12YearsAbove_Quarter5_NumericFeatures_NoBins.csv")

# Different PQ datasets - for external validation
train_data = pd.read_csv("AsthmaPatients_12YearsAbove_Quarter5_NumericFeatures_NoBins_withNullSABARatio_FullData.csv")
test_data = pd.read_csv("AsthmaPatients_12YearsAbove_Quarter9_NumericFeatures_NoBins_withNullSABARatio_FullData.csv")

train_data_copy = train_data.copy()
test_data_copy = test_data.copy()

train_data['HistoryOfAllergies'] = ((train_data['Rhinitis']==1) | (train_data['NasalPolyps']==1) | (train_data['AtopicDermatitis']==1) | (train_data['Anaphylaxis']==1) | (train_data['PulmonaryEosinophilia']==1)).astype(int)
test_data['HistoryOfAllergies'] = ((test_data['Rhinitis']==1) | (test_data['NasalPolyps']==1) | (test_data['AtopicDermatitis']==1) | (test_data['Anaphylaxis']==1)| (test_data['PulmonaryEosinophilia']==1)).astype(int)

#Deriving Cardiovascular Group
train_data['CardiovascularDiseases'] = ((train_data['CardiovascularCerebrovascularDisease']==1) | (train_data['IschaemicHeartDisease']==1) | (train_data['HeartFailure']==1)).astype(int)
test_data['CardiovascularDiseases'] = ((test_data['CardiovascularCerebrovascularDisease']==1) | (test_data['IschaemicHeartDisease']==1) | (test_data['HeartFailure']==1)).astype(int)

# #Handling Unknown Ethnic group (1181 records)
train_data.loc[train_data["EthnicGroup"]=='Unknown',"EthnicGroup"] = 'Other'

#Refined FULL SET
train_data = train_data.drop(columns=['Rhinitis','NasalPolyps','AtopicDermatitis','Anaphylaxis','PulmonaryEosinophilia','CardiovascularCerebrovascularDisease','IschaemicHeartDisease','HeartFailure','P6MNoOfAsthAttacks', 'P3MNoOfAsthAttacks','NoOfICSInhalers', 'NoOfSABAInhalers', 'NoOfAsthmaControllerMeds', 'NebulisedSABA', 'AsthmaSeverityStep', 'RheumatologicalDisease', 'DementiaAlzheimers'], axis=1)
test_data = test_data.drop(columns=['Rhinitis','NasalPolyps','AtopicDermatitis','Anaphylaxis','PulmonaryEosinophilia','CardiovascularCerebrovascularDisease','IschaemicHeartDisease','HeartFailure','P6MNoOfAsthAttacks', 'P3MNoOfAsthAttacks','NoOfICSInhalers', 'NoOfSABAInhalers', 'NoOfAsthmaControllerMeds', 'NebulisedSABA', 'AsthmaSeverityStep', 'RheumatologicalDisease', 'DementiaAlzheimers'], axis=1)

#RFECV Selected Features (RFECV with XGB on fullset)
train_data = train_data.drop(columns=['Psoriasis'], axis=1)
test_data = test_data.drop(columns=['Psoriasis'], axis=1)

#RFECV Selected Features (RFECV with RF on fullset) 
train_data = train_data[['SABA_ICS_Ratio', 'P12MNoOfAsthAttacks','AsthmaAttack_Q5']].copy()
test_data = test_data[['SABA_ICS_Ratio', 'P12MNoOfAsthAttacks','AsthmaAttack_Q5']].copy()

#Clinically important features (all clinically important features)
train_data = train_data[['Q5_WeeksDuringWinter', 'Age', 'Gender', 'EthnicGroup', 'DeprivationQuintile','SmokingStatus', 'SABA_ICS_Ratio', 'P12MNoOfAsthAttacks', 'NoOfOPEDVisits', 'NoOfHospitalisations', 'Paracetamol', 'NSAIDs', 'NumberOfMetforminRx', 'CharlsonComorbidityScore_12Max', 'HistoryOfAllergies','CardiovascularDiseases','AsthmaAttack_Q5']]
test_data = test_data[['Q5_WeeksDuringWinter', 'Age', 'Gender', 'EthnicGroup', 'DeprivationQuintile', 'SmokingStatus', 'SABA_ICS_Ratio', 'P12MNoOfAsthAttacks', 'NoOfOPEDVisits', 'NoOfHospitalisations', 'Paracetamol', 'NSAIDs', 'NumberOfMetforminRx', 'CharlsonComorbidityScore_12Max', 'HistoryOfAllergies','CardiovascularDiseases','AsthmaAttack_Q5']]

#Clinically important features (D2 set - RFECV-XGB)
train_data = train_data[['Q5_WeeksDuringWinter', 'Age', 'Gender', 'EthnicGroup', 'DeprivationQuintile','SmokingStatus', 'SABA_ICS_Ratio', 'P12MNoOfAsthAttacks', 'NoOfOPEDVisits', 'Paracetamol','NSAIDs','CharlsonComorbidityScore_12Max','HistoryOfAllergies','CardiovascularDiseases' ,'AsthmaAttack_Q5']]
test_data = test_data[['Q5_WeeksDuringWinter', 'Age', 'Gender', 'EthnicGroup', 'DeprivationQuintile','SmokingStatus', 'SABA_ICS_Ratio', 'P12MNoOfAsthAttacks', 'NoOfOPEDVisits', 'Paracetamol','NSAIDs','CharlsonComorbidityScore_12Max','HistoryOfAllergies','CardiovascularDiseases' ,'AsthmaAttack_Q5']]

#XGB based Important Features
train_data = train_data[['P12MNoOfAsthAttacks','WeeksDuringWinter','NoOfICSInhalers','SABA_ICS_Ratio','P6MNoOfAsthAttacks', 'P3MNoOfAsthAttacks', 'NoOfSABAInhalers', 'Gender','EthnicGroup','DHB', 'AsthmaAttack']]
test_data = test_data[['P12MNoOfAsthAttacks','WeeksDuringWinter','NoOfICSInhalers','SABA_ICS_Ratio','P6MNoOfAsthAttacks', 'P3MNoOfAsthAttacks', 'NoOfSABAInhalers', 'Gender','EthnicGroup','DHB', 'AsthmaAttack']]

#XGB based Important Features - PQ5 Fullset
train_data = train_data[['P12MNoOfAsthAttacks','WeeksDuringWinter','NoOfICSInhalers','SABA_ICS_Ratio','P6MNoOfAsthAttacks', 'P3MNoOfAsthAttacks', 'NoOfSABAInhalers', 'Gender','EthnicGroup','DHB', 'AsthmaAttack']]
test_data = test_data[['P12MNoOfAsthAttacks','WeeksDuringWinter','NoOfICSInhalers','SABA_ICS_Ratio','P6MNoOfAsthAttacks', 'P3MNoOfAsthAttacks', 'NoOfSABAInhalers', 'Gender','EthnicGroup','DHB', 'AsthmaAttack']]

#Define custom age bins
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

categorical_features = ['EthnicGroup','DHB']

numeric_feature_refined_fullset = ['Q5_WeeksDuringWinter', 'NumberOfMetforminRx', 'NoOfHospitalisations', 'NoOfOPEDVisits', 'SABA_ICS_Ratio', 'P12MNoOfAsthAttacks']

numeric_feature_technical_rfecv_xgb = ['Q5_WeeksDuringWinter', 'NumberOfMetforminRx', 'NoOfHospitalisations', 'NoOfOPEDVisits', 'SABA_ICS_Ratio', 'P12MNoOfAsthAttacks']

numeric_feature_technical_rfecv_rf = ['SABA_ICS_Ratio', 'P12MNoOfAsthAttacks']

numeric_feature_clinical_setD2 = ['Q5_WeeksDuringWinter', 'NumberOfMetforminRx', 'NoOfHospitalisations', 'NoOfOPEDVisits', 'SABA_ICS_Ratio', 'P12MNoOfAsthAttacks']

numeric_feature_clinical_setD2_rfecv_xgb = ['Q5_WeeksDuringWinter','NoOfOPEDVisits', 'SABA_ICS_Ratio', 'P12MNoOfAsthAttacks']

numeric_feature_xgb_important = ['P12MNoOfAsthAttacks','WeeksDuringWinter','NoOfICSInhalers','SABA_ICS_Ratio','P6MNoOfAsthAttacks', 'P3MNoOfAsthAttacks', 'NoOfSABAInhalers']

ST_scalar = StandardScaler()
OH_Encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[
          ('categorical',OH_Encoder,categorical_features),
        ('numeric',ST_scalar,numeric_feature_xgb_important)
        ],
    remainder='passthrough'
    )

print(Counter(train_data["AsthmaAttack"]))

#%% Splitting into Dependent and Independent variables
y_train = train_data["AsthmaAttack"]
X_train = train_data.drop("AsthmaAttack", axis=1)
y_test = test_data["AsthmaAttack"]
X_test = test_data.drop("AsthmaAttack", axis=1)

X_train.reset_index(drop=True,inplace=True)
y_train.reset_index(drop=True,inplace=True)
X_test.reset_index(drop=True,inplace=True)
y_test.reset_index(drop=True,inplace=True)

print(Counter(y_train))
print(Counter(y_test))

#%% Model Efficiency Calculation - Train and Predict
np.random.seed(93196)
model_base = LogisticRegression(random_state=93196, max_iter=400)
model_base = RandomForestClassifier(random_state=93196)
model_base = XGBClassifier(random_state=93196, n_jobs=-1,verbosity=1)
# model_base = GaussianNB()
# model_base = SVM()

sampler = RandomUnderSampler(random_state=93196)
sampler = EditedNearestNeighbours()
sampler = SMOTE(random_state=93196)
sampler = KMeansSMOTE(
    sampling_strategy='auto',
    k_neighbors=5,
    cluster_balance_threshold=0.05,  # Optional: min minority ratio to keep a cluster, cluster balance changed from 0.1 to 0.05 for clinical rfecv-xgb feature set
    random_state=42
)

pipe_base = Pipeline(steps=[ ("preprocessor",preprocessor),
                        ("sampler",sampler),
                        ("classifier",model_base)
                ]
            )

#Calling model_efficiency module
efficiency_metrics = modefficienteval.get_efficiency_metrics(pipe_base, X_train, y_train,X_test)
modefficienteval.log_metrics_to_csv(efficiency_metrics,
                           model_name = model_base.__class__.__name__,
                           dataset ='RFECV-RF 2 features',
                           # imbalhandlingtech = 'No-sample')
                           imbalhandlingtech = sampler.__class__.__name__)

# Looping the efficiency calculation -----------------------------------------------
models = [LogisticRegression(random_state=93196, max_iter=400),
          RandomForestClassifier(random_state=93196),
          XGBClassifier(random_state=93196, n_jobs=-1,verbosity=1) ]

samplers = [RandomUnderSampler(random_state=93196),
            EditedNearestNeighbours(),
            SMOTE(random_state=93196),
            KMeansSMOTE(sampling_strategy='auto', k_neighbors=5, cluster_balance_threshold=0.05, random_state=42),
            "No-sample"]

np.random.seed(93196)
for model in models:
   print('Model: ', model)
   for sampler in samplers:
      print('Sampler:', sampler)
      if sampler == "No-sample":
         pipe = Pipeline(steps=[ ("preprocessor",preprocessor),
                           ("classifier",model)])
      else:
         pipe = Pipeline(steps=[ ("preprocessor",preprocessor),
                        ("sampler",sampler),
                        ("classifier",model)])
      for iteration in range(1,11):
         efficiency_metrics = modefficienteval.get_efficiency_metrics(pipe, X_train, y_train, X_test)
         modefficienteval.log_iterative_metrics_to_csv(iteration, efficiency_metrics,
                           model_name = model.__class__.__name__,
                           dataset ='Clinical fullset 16 features',
                           imbalhandlingtech = sampler)

 #------------------------------------------------------------------------------
#%% Model Efficiency & Accuracy Calculation with 5 Fold CV - Train and Predict
models = [LogisticRegression(random_state=93196, max_iter=400),
          RandomForestClassifier(random_state=93196),
          XGBClassifier(random_state=93196, n_jobs=-1,verbosity=1) ]

samplers = [RandomUnderSampler(random_state=93196),
            EditedNearestNeighbours(),
            SMOTE(random_state=93196),
            KMeansSMOTE(sampling_strategy='auto', k_neighbors=5, cluster_balance_threshold=0.05, random_state=42),
            "No-sample"]

np.random.seed(93196)
for model in models:
   print('Model: ', model)
   for sampler in samplers:
             print('Sampler:', sampler)
             if sampler == "No-sample":
                pipe = Pipeline(steps=[ ("preprocessor",preprocessor),
                           ("classifier",model)])
             else:
                pipe = Pipeline(steps=[ ("preprocessor",preprocessor),
                        ("sampler",sampler),
                        ("classifier",model)])
             modefficienteval.run_cv_and_test(X_train, y_train, X_test, y_test, pipe, model, sampler,
                            dataset_name="XGB-based importance 10 features")


#%%Train and Test model - Extracting
np.random.seed(93196)
# model_base = LogisticRegression(random_state=93196, max_iter=400)
# model_base = RandomForestClassifier(random_state=93196)
model_base = XGBClassifier(random_state=93196, n_jobs=-1,verbosity=1)
# model_base = GaussianNB()
# model_base = SVM()

sampler = RandomUnderSampler(random_state=93196)
# sampler = EditedNearestNeighbours()
# sampler = SMOTE(random_state=93196)
# sampler = KMeansSMOTE(
#     sampling_strategy='auto',
#     k_neighbors=5,
#     cluster_balance_threshold=0.05,  # Optional: min minority ratio to keep a cluster, cluster balance changed from 0.1 to 0.05 for clinical rfecv-xgb feature set
#     random_state=42
# )

pipe_base = Pipeline(steps=[ ("preprocessor",preprocessor),
                        ("sampler",sampler),
                        ("classifier",model_base)
                ]
            )

pipe_base.fit(X_train,y_train)
y_pred_base = pipe_base.predict(X_test)
y_pred_proba_base = pipe_base.predict_proba(X_test)
y_pred_proba_class1 = y_pred_proba_base[::,1]
modperformeval.evaluateFinalModel(y_test,y_pred_proba_class1,y_pred_base)

#%% Analysing the model efficiency factors - for single iterations
### Import Data ###
model_efficiency_data = pd.read_csv("model_efficiency_log_analyse.csv")
dataset_order = ['Refined full set 24 features', 'RFECV-XGB 23 features', 'Clinical fullset 16 features', 'Clinical RFECV-XGB 14 features', 'RFECV-RF 2 features']
sampling_order = ['RandomUnderSampler', 'EditedNearestNeighbours', 'SMOTE', 'KMeansSMOTE', 'No-sample']
model_efficiency_data['Dataset'] = pd.Categorical(model_efficiency_data['Dataset'], categories = dataset_order, ordered = True)
model_efficiency_data['Data Imbalance Handling Technique'] = pd.Categorical(model_efficiency_data['Data Imbalance Handling Technique'], categories = sampling_order, ordered = True)
model_efficiency_data = model_efficiency_data.sort_values(['Dataset','Data Imbalance Handling Technique'])


#%% Analysing the model efficiency factors - for multiple iterations
# Import data
model_efficiency_iter_data = pd.read_csv("model_efficiency_log_iterativeResults.csv")
# Define replacement mappings
model_map = {
    'RandomForestClassifier': 'RF',
    'XGBClassifier': 'XGB'
}

imbalance_map = {
   'RandomUnderSampler(random_state=93196)':'RandomUndersample',
   'EditedNearestNeighbours()': 'ENN',
   'SMOTE(random_state=93196)': 'SMOTE',
   'KMeansSMOTE(cluster_balance_threshold=0.05, k_neighbors=5, random_state=42)': 'KMeans-SMOTE'
}

# Apply replacements
model_efficiency_iter_data['model'] = model_efficiency_iter_data['model'].replace(model_map)
model_efficiency_iter_data['imbalance_handling_technique'] = model_efficiency_iter_data['imbalance_handling_technique'].replace(imbalance_map)

#Calculate mean and std
train_time_ci = (
    model_efficiency_iter_data
    .groupby(['dataset', 'model', 'imbalance_handling_technique'])['train_time_sec']
    .agg(mean='mean', std='std', n='count')
    .reset_index()
)
train_time_ci['sem'] = train_time_ci['std'] / np.sqrt(train_time_ci['n']) #standard error of the mean
train_time_ci['t_crit'] = stats.t.ppf(0.975, df=train_time_ci['n'] - 1) #t-distribution Critical Value
train_time_ci['ci_lower'] = train_time_ci['mean'] - train_time_ci['t_crit'] * train_time_ci['sem']
train_time_ci['ci_upper'] = train_time_ci['mean'] + train_time_ci['t_crit'] * train_time_ci['sem']

#Save the mean and std values
train_time_ci.to_csv("train_time_summaryStatistics.csv", index=False)


#%% Analyse model accuracy and efficiency - 5 fold cv
#ANOVA Analysis
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

#import 5fold cv results
resutls_5fcv =  pd.read_csv("fold_level_metrics.csv")

model_f1 = ols('F1 ~ C(model_id) * C(imbalance_handling_technique) * C(dataset)', data=resutls_5fcv).fit()
print(anova_table_f1)
model_aucroc = ols('AUROC ~ C(model_id) * C(imbalance_handling_technique) * C(dataset)', data=resutls_5fcv).fit()
print(anova_table_aucroc)

#%% Calculating Confidence Intervals for model accuracy - 5 fold cv
import pandas as pd
import scipy.stats as stats

# Load the data
cv_metrics_data = pd.read_csv("fold_level_metrics.csv")  # Replace with your actual file

# Function to compute 95% confidence interval
def compute_ci(series):
    mean = series.mean()
    sem = stats.sem(series)
    ci_low, ci_high = stats.t.interval(
        0.95, df=len(series)-1, loc=mean, scale=sem
    )
    return pd.Series({
        'mean': mean,
        'ci_lower': ci_low,
        'ci_upper': ci_high
    })

# Group by model_id, dataset, and imbalance handling technique
grouped_cv_metrics_data = cv_metrics_data.groupby(
    ['model_id', 'dataset', 'imbalance_handling_technique']
)

# Compute confidence intervals
results = []

for (model_id, dataset, imbalance), group in grouped_cv_metrics_data:
    auroc_ci = compute_ci(group['AUROC'])
    f1_ci = compute_ci(group['F1'])

    results.append({
        'model_id': model_id,
        'dataset': dataset,
        'imbalance_handling_technique': imbalance,
        'AUROC_mean': auroc_ci['mean'],
        'AUROC_CI_lower': auroc_ci['ci_lower'],
        'AUROC_CI_upper': auroc_ci['ci_upper'],
        'F1_mean': f1_ci['mean'],
        'F1_CI_lower': f1_ci['ci_lower'],
        'F1_CI_upper': f1_ci['ci_upper'],
    })

# Create final DataFrame and save
ci_cv_metrics_data = pd.DataFrame(results)
ci_cv_metrics_data.to_csv("model_accuracy_confidence_intervals.csv", index=False)


#%% Plot model efficiency 

model_performance = pd.read_csv('efficiency_metrics_modified.csv')
model_efficiency = model_performance[['predict_time_sec','model_size_MB','model_id','dataset_id','imbalance_handling_technique']]

# Long form
df_long = model_efficiency.melt(
    id_vars=['model_id', 'dataset_id', 'imbalance_handling_technique'],
    value_vars=['predict_time_sec', 'model_size_MB'],
    var_name='Efficiency Metric',
    value_name='Value'
)

row_order = ['Prediction Time (sec)', 'Model Size (MB)']
df_long['Efficiency Metric'] = df_long['Efficiency Metric'].replace({
    'predict_time_sec': row_order[0],
    'model_size_MB'   : row_order[1]
})
df_long['Efficiency Metric'] = pd.Categorical(df_long['Efficiency Metric'],
                                              categories=row_order, ordered=True)

# Dataset order
dataset_order = ['FS1','FS2','FS3','FS4','FS5','FS6']
df_long['dataset_id'] = pd.Categorical(df_long['dataset_id'],
                                       categories=dataset_order, ordered=True)

# Plot
g = sns.catplot(
    data=df_long,
    x='model_id',
    y='Value',
    hue='imbalance_handling_technique',
    col='dataset_id',
    row='Efficiency Metric',
    kind='bar',
    height=4,
    aspect=1.3,
    sharey=False,
    row_order=row_order,
    col_order=dataset_order
)

# SubtLitles
for ax in g.axes.flatten():
    ax.set_title("")  # clear to avoid the "|" joiner

# Set subtitles only for the top row
for j, col_name in enumerate(g.col_names):
    g.axes[0, j].set_title(f"Dataset ID - {col_name}", fontsize=14, pad=10)

# Remove titles for the second row (optional, could skip since cleared above)
for j in range(len(g.col_names)):
    g.axes[1, j].set_title("")

for i, row_lab in enumerate(row_order):
    g.axes[i, 0].set_ylabel(row_lab, fontsize=14)
    for j in range(1, len(g.col_names)):
        g.axes[i, j].set_ylabel("")

for ax in g.axes.flatten():
    ax.tick_params(axis='x', labelsize=12, labelrotation=0)
    ax.tick_params(axis='x', which='both', labelbottom=True)  # force visible
    # keep current labels but ensure alignment/font
    labels = [tick.get_text() for tick in ax.get_xticklabels()]
    ax.set_xticklabels(labels, ha='center')

# Axis labels
g.set_xlabels("ML Algorithm", fontsize=14)

leg = g._legend
if leg is not None:
    leg.set_title("Imbalance Handling Technique")
    leg.set_bbox_to_anchor((0.75, 0))  # outside bottom-right
    leg.set_loc('lower right')
    leg.set_frame_on(False)
    for txt in leg.get_texts():
        txt.set_fontsize(12)
    leg.get_title().set_fontsize(12)

# Spacing to accommodate outside legend
g.figure.subplots_adjust(bottom=0.30, right=0.75, top=0.9, hspace=0.35)
plt.show()


















