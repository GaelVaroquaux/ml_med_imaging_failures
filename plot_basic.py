# -*- coding: utf-8 -*-
"""
Only checking if the tables import correctly, actual plots are in plot_papers.py 


Created on Mon Oct 19 08:16:36 2020

@author: vcheplyg
"""

from matplotlib import pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
import numpy as np
import os
import re

np.random.seed(42)

data_path = 'paper_tables_clean'



###############################################################################
#Liu et al, studies in medical imaging (different applications) where ML was competitive with experts. 
# 82 studies with sample size (table 1, loaded here), type of validation (table 2) and other details   

file_name = 'liu2019comparison_table1.html'
df_liu = pd.read_html(os.path.join(data_path,file_name))
df_liu = df_liu[0]
df_liu = df_liu.dropna()
df_liu['origin'] = 'Liu et al, 2019'

#Different applications in medical iaging  (only a few neuro) 
df_liu['task'] = df_liu['Unnamed: 1'] 

#Extract year from reference
df_liu['year'] = df_liu['Unnamed: 0'].apply(lambda s: re.split("[()]",s)[1])
df_liu['year'] = pd.to_numeric(df_liu['year'])

#Number of subjects represented, often 'NR' - not reported, so not too useful
#df_liu['subjects'] = df_liu['Number of participants represented by the training data'] 

#Table 3 reports numbers of training/validation, maybe this has more data 
file_name = 'liu2019comparison_table3.html'
df_liu3 = pd.read_html(os.path.join(data_path,file_name))
df_liu3 = df_liu3[0]
df_liu3 = df_liu3.dropna()
df_liu3['origin'] = 'Liu et al, 2019'

df_liu3['subjects'] = df_liu3['Number of images for training-tuning'] 
#Format: Number/number (often Number/NR), or "Number/number;Number/number"
#ASSUMPTION for simplicity we just take the first number...  

df_liu3['subjects_train'] = df_liu3['subjects'].apply(lambda s: re.split("[/:\s]",s)[0])

df_liu['subjects'] = df_liu3['subjects_train']
df_liu['subjects'] = df_liu['subjects'].replace('NR', np.nan)
df_liu['subjects'] = df_liu['subjects'].replace('NotApplicable', np.nan)

df_liu['subjects'] = pd.to_numeric(df_liu['subjects'])
df_liu = df_liu.dropna()

df_liu.plot.scatter(x='year', y='subjects')



################################################################################
# Gautam et al 2020 - Neuro, with modality, several metrics, classfication (binary, ternary etc), dataset...
# Table 5 is the table for AD diagnosis, other tables are for other diseases (and have less entries) 
file_name = 'gautam2020prevalence_table5.html'
df_gautam = pd.read_html(os.path.join(data_path,file_name))
df_gautam = df_gautam[0]
df_gautam = df_gautam.dropna()
df_gautam['origin'] = 'Gautam et al, 2020'

#Extract year from reference
df_gautam['year'] = df_gautam['Author'].apply(lambda s: re.split("[,\[]",s)[1])
df_gautam['year'] = pd.to_numeric(df_gautam['year'])

#Subjects
df_gautam['subjects'] = pd.to_numeric(df_gautam['Instances'])

#Accuracy
df_gautam['acc'] = df_gautam['Accuracy'].apply(lambda s: re.split("%",s)[0])
df_gautam['acc'] = pd.to_numeric(df_gautam['acc'], errors='coerce')


df_gautam.plot.scatter(x='year', y='subjects')
df_gautam.plot.scatter(x='subjects', y='acc')


################################################################################
# Wen et al 2020, Neuro with 6 tables:
# table1 - overview of studies with year and performance (but no sample size?) and data leakage
# table2, table3, table4 - details of ADNI, AIBL and OASIS datasets
# table5, table6 - own experiments they performed (systematic comparison)

file_name = 'wen2020convolutional_tables.xlsx'
df_wen = pd.read_excel(os.path.join(data_path,file_name), sheet_name='table1')
df_wen['origin'] = 'Wen et al, 2020'

# 5 different classification tasks, now taking only AD vs CN
# Also avalable are: acc_sMCI_pMCI	acc_MCI_CN	acc_AD_MCI	acc_Multi-class
# In two cases where BA was reported instead of ACC, it was dropped in the acc_ column 

df_wen['acc_AD_CN'] = pd.to_numeric(df_wen['acc_AD_CN'], errors='coerce')
df_wen = df_wen.dropna()

df_wen.plot.scatter(x='year', y='acc_AD_CN') #Only 2017-2019 is represented


##################################################################################
# Ansart et al 2019, Neuro (pMCI vs sMCI). Number subjects, dataset (ADNI, local etc), various details about method, AUC/accuracy 
#Different types of validation (k-fold etc) are used, the AUC and accuracy columns aggregate these

file_name = 'ansart2019predicting_table.csv'
df_ansart = pd.read_csv(os.path.join(data_path,file_name))
df_ansart['origin'] = 'Ansart et al, 2019'

df_ansart['subjects'] = df_ansart['nb pMCI'] + df_ansart['nb sMCI']

df_ansart.plot.scatter(x='publication year', y='subjects')
df_ansart.plot.scatter(x='publication year', y='AUC')
#df_ansart.plot.scatter(x='publication year', y='accuracy')
df_ansart.plot.scatter(x='subjects', y='AUC')




