# -*- coding: utf-8 -*-
"""
Plot the performances reported in publications
"""


from matplotlib import pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
import numpy as np
import os
import re

from statsmodels.nonparametric.smoothers_lowess import lowess

np.random.seed(42)

data_path = 'paper_tables_clean'


###############################################################################
#Sakai et al, studies on AD (different subproblems) available with AUC, sample size, other details
file_name = 'sakai2019machine_table3.html'

df_sakai = pd.read_html(os.path.join(data_path,file_name))
df_sakai = df_sakai[0]
df_sakai = df_sakai.dropna()

subjects = df_sakai['Number of subjects'].str.split(pat='total = ', expand=True)[1]
subjects = subjects.astype(int)
df_sakai['subjects'] = subjects

# Add an indication of the origin of the data
df_sakai['origin'] = 'Sakai et al, 2019'


# Build a df with the accuracy per task
query = 'MCI vs HC'
acc = df_sakai['Overall accuracy'].str.split(pat=', ',
                                            expand=True)
acc = acc.reset_index().melt(id_vars='index')
acc = acc.dropna()
df_sakai_task = acc.merge(df_sakai.reset_index(), left_on='index',
                    right_on='index')
df_sakai_task = df_sakai_task.drop(['index', 'variable'], axis=1)
values = df_sakai_task['value'].str.split(' \(', expand=True)
df_sakai_task['acc'] = pd.to_numeric(values[0], errors='coerce')
df_sakai_task['task'] = values[1].str.split('\)', expand=True)[0].str.replace('NC', 'HC')
df_sakai_task = df_sakai_task.dropna()

###############################################################################
# Arbabshirani 2017

df_arbabshirani = pd.read_csv(os.path.join(data_path,
                                'arbabshirani_2017/table_1.csv'))
performance = [float(s[:-1]) for s in df_arbabshirani['Overall accuracy']]
df_arbabshirani['subjects'] = df_arbabshirani['Number of subjects']
df_arbabshirani['acc'] = [float(s[:-1]) / 100.
                          for s in df_arbabshirani['Overall accuracy']]

# Get the year of the publication
year = df_arbabshirani['Reference'].str.split('\(', expand=True)[1]
year = [float(y[:4]) for y in year]
df_arbabshirani['Year'] = year

df_arbabshirani['task'] = df_arbabshirani['Disorder'].map({
                                'AD/MCI': 'MCI vs AD',
                                'AD': 'AD vs HC',
                                'MCI': 'MCI vs HC',
                                })
df_arbabshirani['origin'] = 'Arbabshirani et al, 2017'


###############################################################################
# Dallora et al, 37 papers total (but missing data), best accuracy achieved & year (no sample size)
file_name = 'dallora2017machine_tableS1.xlsx'
df_dallora = pd.read_excel(os.path.join(data_path,file_name))

df_dallora['acc'] = pd.to_numeric(df_dallora['Accuracy_Clean']) / 100.

# Add an indication of the origin of the data
df_dallora['origin'] = 'Dallora et al, 2017'

# Build the df with accuracy per task
df_dallora_task = df_dallora.copy()
df_dallora_task['task'] = df_dallora_task['Conditions Studied'].copy()
df_dallora_task['task'] = df_dallora_task['task'].replace({
                                'AD, MCI': 'MCI vs AD',
                                'AD, HD': 'AD vs HC',
                                })

################################################################################
# Gautam et al 2020 - Neuro, with modality, several metrics, classfication (binary, ternary etc), dataset...
# Table 5 is the table for AD diagnosis, other tables are for other diseases (and have less entries) 
file_name = 'gautam2020prevalence_table5.html'
df_gautam = pd.read_html(os.path.join(data_path,file_name))
df_gautam = df_gautam[0]
df_gautam['origin'] = 'Gautam et al, 2020'

df_gautam = df_gautam.query('Classification == "Binary"')

df_gautam['task'] = 'AD vs HC' # Not quite sure

#Extract year from reference
df_gautam['Year'] = df_gautam['Author'].apply(lambda s: re.split("[,\[]",s)[1])
df_gautam['Year'] = pd.to_numeric(df_gautam['Year'])

#Subjects
df_gautam['subjects'] = pd.to_numeric(df_gautam['Instances'])

#Accuracy
df_gautam['acc'] = df_gautam['Accuracy'].apply(lambda s: re.split("%",s)[0])
df_gautam['acc'] = pd.to_numeric(df_gautam['acc'], errors='coerce') / 100.
df_gautam = df_gautam.dropna()

##################################################################################
# Ansart et al 2019, Neuro (pMCI vs sMCI). Number subjects, dataset (ADNI, local etc), various details about method, AUC/accuracy 
#Different types of validation (k-fold etc) are used, the AUC and accuracy columns aggregate these

file_name = 'ansart2019predicting_table.csv'
df_ansart = pd.read_csv(os.path.join(data_path,file_name))
df_ansart['origin'] = 'Ansart et al, 2019'
df_ansart['task'] = 'pMCI vs sMCI'
df_ansart['acc'] = pd.to_numeric(df_ansart['accuracy'], errors='coerce') / 100.

df_ansart['subjects'] = df_ansart['nb pMCI'] + df_ansart['nb sMCI']

df_ansart['Year'] = df_ansart['reference'].apply(
            lambda s: s.split(" ")[-1][:4])
df_ansart['Year'] = pd.to_numeric(df_ansart['Year'])

################################################################################
# Wen et al 2020, Neuro with 6 tables:
# table1 - overview of studies with year and performance (but no sample size?) and data leakage
# table2, table3, table4 - details of ADNI, AIBL and OASIS datasets
# table5, table6 - own experiments they performed (systematic comparison)

file_name = 'wen2020convolutional_tables.xlsx'
df_wen = pd.read_excel(os.path.join(data_path,file_name), sheet_name='table1')
df_wen = df_wen[~ df_wen['Data leakage'].str.startswith('Clear')]
df_wen['origin'] = 'Wen et al, 2020'

# 5 different classification tasks, now taking only AD vs CN
# Also avalable are: acc_sMCI_pMCI	acc_MCI_CN	acc_AD_MCI	acc_Multi-class
# In two cases where BA was reported instead of ACC, it was dropped in the acc_ column

df_wen['Year'] = df_wen['year']


# 5 different classification tasks, now taking only AD vs CN
# Also avalable are: acc_sMCI_pMCI	acc_MCI_CN	acc_AD_MCI	acc_Multi-class
# In two cases where BA was reported instead of ACC, it was dropped in the acc_ column 

wen_conversion = {'acc_sMCI_pMCI': 'pMCI vs sMCI',
                  'acc_MCI_CN': 'MCI vs HC',
                  'acc_AD_MCI': 'MCI vs AD',
                  'acc_AD_CN': 'AD vs HC',
                 }

df_wen_task = list()
for col, task in wen_conversion.items():
    this_df = df_wen.copy()
    this_df['acc'] = pd.to_numeric(this_df[col], errors='coerce')
    this_df = this_df.dropna()

    this_df['task'] = task
    df_wen_task.append(this_df)

df_wen_task = pd.concat(df_wen_task)

# Drop data after 2019, because of too few data points
#df_wen = df_wen.query('Year < 2019')

###############################################################################
# Concatenate the different studies


df_task = pd.concat([
        df_sakai_task[['task', 'acc', 'subjects', 'Year', 'origin']],
        df_dallora_task[['task', 'acc', 'Year', 'origin']],
        df_wen_task[['task', 'acc', 'Year', 'origin']],
        df_arbabshirani[['task', 'acc', 'subjects', 'Year', 'origin']],
        df_gautam[['task', 'acc', 'subjects', 'Year', 'origin']],
        df_ansart[['task', 'acc', 'subjects', 'Year', 'origin']],
        ])

# Trick: keep the non noisy year with no capitalized letter
df_task['year'] = df_task['Year'].copy()
df_task['Year'] = df_task['Year'].astype(float)
df_task['Year'] += .3 * np.random.random(size=len(df_task))

df_n_subjects = pd.concat([
                    df_sakai[['subjects', 'Year', 'origin']],
                    df_arbabshirani[['subjects', 'Year', 'origin']],
                    df_gautam[['subjects', 'Year', 'origin']],
                    df_ansart[['subjects', 'Year', 'origin']],
                ])

df_n_subjects['Year'] = df_n_subjects['Year'].astype(float)
df_n_subjects['Year'] += .3 * np.random.random(size=len(df_n_subjects))

###############################################################################
# Plots

# TODO: bootstrap the lowess for error bars

# TODO: performance_vs_subjects : semilogx

# different symbols depending on the origin of the data
symbols = {
    'Dallora et al, 2017': '+',
    'Sakai et al, 2019': 'x',
    'Arbabshirani et al, 2017': '1',
    'Gautam et al, 2020': '2',
    'Ansart et al, 2019': '3',
    'Wen et al, 2020': '4',
}

colors = {
    'AD vs HC': 'C0',
    'MCI vs HC': 'C1',
    'MCI vs AD': 'C2',
    'pMCI vs sMCI': 'k',
}

# number of subjects per year
plt.figure(figsize=(3.8, 3))
for data_name, marker in symbols.items():
    plt.scatter(df_n_subjects.query(f'origin == "{data_name}"')['Year'],
                df_n_subjects.query(f'origin == "{data_name}"')['subjects'],
                marker=marker, c='.2', alpha=.4)
xs, ys = lowess(df_n_subjects['subjects'], df_n_subjects['Year'],
                return_sorted=True, frac=.8).T

plt.plot(xs, ys, zorder=11, linewidth=3, color='.2')

plt.yscale('log')
plt.ylim(ymin=20)
plt.xlim(xmin=2004, xmax=2019.4)
plt.xticks([2006, 2010, 2014, 2018], ['2006', '2010', '2014', '2018'])

plt.ylabel('Number of subjects in study    ', size=14)
plt.xlabel("Publication year", size=14)

legend_artists = [mlines.Line2D([], [], color='.2', marker=v, linestyle='None',
                                markersize=4, label=k, mew=2)
                  for k, v in symbols.items()]
plt.legend(handles=legend_artists, loc='upper left',
           title='Review article', fontsize=9, handlelength=1,
           handletextpad=.2)

#ax = plt.gca()
#ax.add_artist(legend1)


plt.tight_layout(pad=.01)
plt.subplots_adjust(left=.16)
plt.savefig('subjects_vs_year.pdf', transparent=True)


# Plot the accuracy vs Year for the most common task
plt.figure(figsize=(3.8, 3))
for task in colors.keys():
    task_df = df_task.query(f'task == "{task}"')

    for data_name, marker in symbols.items():
        plt.scatter(task_df.query(f'origin == "{data_name}"')['Year'],
                    task_df.query(f'origin == "{data_name}"')['acc'],
                    marker=marker, c=colors[task], alpha=.5)

    xs, ys = lowess(task_df['acc'], task_df['Year'],
                    return_sorted=True, frac=.8).T

    plt.plot(xs, ys, zorder=5, linewidth=3,
                    label=task, color=colors[task])

plt.ylim(ymax=1)
plt.xlim(xmin=2004, xmax=2019.4)
plt.xticks([2006, 2010, 2014, 2018], ['2006', '2010', '2014', '2018'])

legend1 = plt.legend(loc='lower left')


plt.xlabel("Publication year", size=14)
plt.ylabel("Reported prediction accuracy        ", size=14)

plt.tight_layout(pad=.01)
plt.subplots_adjust(left=.16)
plt.savefig('performance_vs_year.pdf', transparent=True)


# Plot the accuracy vs # subjects for the most common task
plt.figure(figsize=(3.8, 3))
for task in colors.keys():
    task_df = df_task.query(f'task == "{task}"')
    task_df = task_df.dropna()
    for data_name, marker in symbols.items():
        plt.scatter(task_df.query(f'origin == "{data_name}"')['subjects'],
                    task_df.query(f'origin == "{data_name}"')['acc'],
                    marker=marker, c=colors[task], alpha=.5)


    xs, ys = lowess(task_df['acc'], task_df['subjects'].astype(float),
                    return_sorted=True, frac=.8).T

    plt.plot(xs, ys, zorder=11, linewidth=3, label=task, color=colors[task])

ax = plt.gca()
ax.set_xscale('log')
plt.xlim(28, 1400)
plt.legend(loc='lower left')

plt.xlabel('Number of subjects in study', size=14)
plt.ylabel("Reported prediction accuracy          ", size=14)

plt.tight_layout(pad=.01)
plt.subplots_adjust(left=.16)
plt.savefig('performance_vs_subjects.pdf', transparent=True)


# Plot the accuracy vs # subjects for the most common task at differen
# times
time_symbols = {
    "year >= 2017": (">", "-"),
    "2013 < year < 2017": ("^", "--"),
    "year <= 2013": ("<", ":"),
}

plt.figure(figsize=(3.8, 3))
for task in colors.keys():
    if not task in ('pMCI vs sMCI', 'AD vs HC'):
        continue
    task_df = df_task.query(f'(task == "{task}")')
    task_df = task_df.dropna()
    for date_query, (marker, linestyle) in time_symbols.items():
        this_df = task_df.query(date_query)
        #plt.scatter(this_df['subjects'],
        #            this_df['acc'], s=4,
        #            marker=marker, c=colors[task], alpha=.8)

        xs, ys = lowess(this_df['acc'], this_df['subjects'].astype(float),
                        return_sorted=True, frac=.8).T

        plt.plot(xs, ys, linewidth=3,
                linestyle=linestyle,
                label=(date_query if task == 'pMCI vs sMCI' else None),
                color=colors[task])

ax = plt.gca()
ax.set_xscale('log')
plt.xlim(28, 1400)
plt.yticks([.7, .8, .9, 1])

plt.text(4.55e2, .64, 'pMCI vs\nsMCI', color=colors['pMCI vs sMCI'], size=13)
plt.text(3e2, .87, 'AD vs HC', color=colors['AD vs HC'], size=13)

plt.xlabel('Number of subjects in study', size=14)
plt.ylabel("Reported prediction acuracy          ", size=14)
plt.legend(loc='lower left')

plt.tight_layout(pad=.01)
plt.subplots_adjust(left=.17)
plt.savefig('performance_vs_subjects_time.pdf', transparent=True)

# Compute multivariate regression to explain performance as a function of
# time vs sample size

from statsmodels.formula.api import ols

for task in colors.keys():
    if not task in ('pMCI vs sMCI', 'AD vs HC'):
        continue
    task_df = df_task.query(f'(task == "{task}")')
    # task_df['Year'] /= 1000
    # Note: in statsmodels, the coefficients already account for the
    # scaling of the columns, hence they can readily be compared

    model = ols("acc ~ subjects + Year", task_df).fit()
    print(task)
    print(model.summary())
