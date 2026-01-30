import numpy as np
import pandas as pd
import time, pickle, os, pprint
from pathlib import Path
import calendar
import matplotlib.pyplot as plt
# import matplotlib
import seaborn as sns
from pprint import pprint

data_path = r"C:\Users\dbf1941\OneDrive - AUT University\Python-projects\asthma_attack_risk_prediction\data"

data_path = r"C:\Users\dbf1941\OneDrive - AUT University\Python-projects\asthma_attack_risk_prediction\AdjustedDataset-Andy"
# data_withDomCode = pd.read_csv(os.path.join(data_path,"AUT_Darsha_2023_AsthmaPatients_Over6YearsOfAge_Quarter5_WithDomicileCode.txt"),sep='|')
data_withDomCode_fullQ5 = pd.read_csv(os.path.join(data_path,"Asthma_20082017Cohort_PatientQuarter5_NewDataFields_Over6YearsOfAge.txt"),sep='|')
data_withDomCode_fullQ9 = pd.read_csv(os.path.join(data_path,"Asthma_20082017Cohort_PatientQuarter9_NewDataFields_Over6YearsOfAge.txt"),sep='|')

data_withDomCode_fullQ5_original = data_withDomCode_fullQ5.copy()
data_withDomCode_fullQ9_original = data_withDomCode_fullQ9.copy()

# Filter patients with Age >= 12
data_withDomCode_fullQ5_12andAbove = data_withDomCode_fullQ5[data_withDomCode_fullQ5['AgeAtQuarterStartDate']>=12]
data_withDomCode_fullQ9_12andAbove = data_withDomCode_fullQ9[data_withDomCode_fullQ9['AgeAtQuarterStartDate']>=12]

# Distribution of target variable - Full sets - PQ5
plt.rcParams.update({'font.size': 10})
plt.figure(dpi=300)
ax = sns.countplot(data=data_withDomCode_fullQ5_12andAbove, x=data_withDomCode_fullQ5_12andAbove["AsthmaAttack"], palette=["#1f77b4", "#ff7f0e"]) #,hue="ModellingDataset")
for i in ax.containers:
    ax.bar_label(i, fmt='%d')
   # Change axis labels
ax.set_xlabel("Asthma Attack", fontsize=12)
ax.set_ylabel("Number of patient-records", fontsize=12)
plt.title('Asthma Attacks Distribution in Patient-Quarter 5 Dataset', fontsize=13)

# Distribution of target variable - Full sets - PQ9
plt.rcParams.update({'font.size': 10})
plt.figure(dpi=300)
ax = sns.countplot(data=data_withDomCode_fullQ9_12andAbove, x=data_withDomCode_fullQ9_12andAbove["AsthmaAttack"], palette=["#1f77b4", "#ff7f0e"]) #,hue="ModellingDataset")
for i in ax.containers:
    ax.bar_label(i, fmt='%d')
   # Change axis labels
ax.set_xlabel("Asthma Attack", fontsize=12)
ax.set_ylabel("Number of patient-records", fontsize=12)
plt.title('Asthma Attacks Distribution in Patient-Quarter 9 Dataset', fontsize=13)

# Excluding features
data_withDomCode_fullQ5_12andAbove_copy = data_withDomCode_fullQ5_12andAbove.copy()
data_withDomCode_fullQ9_12andAbove_copy = data_withDomCode_fullQ9_12andAbove.copy()

pprint(data_withDomCode_fullQ5_12andAbove_copy.columns)
excluding_features = ["ENHI","IndexDate","ObservationEndDate", "CohortYear", "PatientQuarter",'QuarterStartDate', 'QuarterEndDate', 'AsthmaHospitalisation', 'OralCorticosteroidPrescriptions', "P12MAsthAttack","P6MAsthAttack", "P3MAsthAttack", "DateOfBirth","P12MAsthAttack","P6MAsthAttack", "P3MAsthAttack","EthnicityCode", 'DomicileCode', "DHB_Code", 'DateOfDeath', 'DeathDiagnosis_Asthma', "Metformin"]
data_withDomCode_fullQ5_12andAbove.drop(excluding_features, axis=1, inplace=True)
data_withDomCode_fullQ9_12andAbove.drop(excluding_features, axis=1, inplace=True)
# pprint(data.columns)

# Re-naming columns
data_withDomCode_fullQ5_12andAbove.rename(columns = {"Sex":"Gender", 'AgeAtQuarterStartDate':'Age'}, inplace=True)
data_withDomCode_fullQ9_12andAbove.rename(columns = {"Sex":"Gender", 'AgeAtQuarterStartDate':'Age'}, inplace=True)

# Replacing NaN values
data_withDomCode_fullQ5_12andAbove.isna().sum()
data_withDomCode_fullQ5_12andAbove["SABA_ICS_Ratio"] = data_withDomCode_fullQ5_12andAbove["SABA_ICS_Ratio"].replace(np.nan,0)
data_withDomCode_fullQ5_12andAbove.isna().sum()
data_withDomCode_fullQ9_12andAbove.isna().sum()
data_withDomCode_fullQ9_12andAbove["SABA_ICS_Ratio"] = data_withDomCode_fullQ9_12andAbove["SABA_ICS_Ratio"].replace(np.nan,0)
data_withDomCode_fullQ9_12andAbove.isna().sum()

#Replacing string values
data_withDomCode_fullQ5_12andAbove["Gender"] = data_withDomCode_fullQ5_12andAbove["Gender"].apply(lambda x: 0 if (x=='F') else 1)
data_withDomCode_fullQ9_12andAbove["Gender"] = data_withDomCode_fullQ9_12andAbove["Gender"].apply(lambda x: 0 if (x=='F') else 1)

# Saving train and test sets into two separate files
data_path_save = r"C:\Users\dbf1941\OneDrive - AUT University\Python-projects\asthma_attack_risk_prediction\AdjustedDataset-Andy"
os.chdir(data_path_save)
data_withDomCode_fullQ5_12andAbove.to_csv("AsthmaPatients_12YearsAbove_Quarter5_NumericFeatures_NoBins_withNullSABARatio_FullData.csv", index=False)
data_withDomCode_fullQ9_12andAbove.to_csv("AsthmaPatients_12YearsAbove_Quarter9_NumericFeatures_NoBins_withNullSABARatio_FullData.csv", index=False)
