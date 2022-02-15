# -*- coding: utf-8 -*-
"""
Plot the publication counts on a few topics
"""
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
import numpy as np
import os
import re



data_path = 'publication_counts'


df_counts = pd.read_excel(os.path.join(data_path,'publication_counts.xlsx'))



df_counts['lung_vs_all_medical'] = df_counts['lung_medical'] / df_counts['all_medical'] * 100
df_counts['lung_vs_all_ai'] = df_counts['lung_ai'] / df_counts['all_ai'] * 100


df_counts['breast_vs_all_medical'] = df_counts['breast_medical'] / df_counts['all_medical'] * 100
df_counts['breast_vs_all_ai'] = df_counts['breast_ai'] / df_counts['all_ai'] * 100


plt.figure(figsize=(5.8, 3))

plt.axvline(2016, color='.3', linestyle='-')
plt.text(2015.9, 3.95, 'Kaggle lung\n  challenge', #rotation=90,
         va='top', ha='right',
         size=13)


plt.plot(df_counts['year'], df_counts['lung_vs_all_medical'],
         label='lung, medical', color='C0', linestyle='-',
         linewidth=2)
plt.plot(df_counts['year'], df_counts['lung_vs_all_ai'],
         label='lung, AI', color='C0', linestyle='--',
         linewidth=2)

plt.plot(df_counts['year'], df_counts['breast_vs_all_medical'],
         label='breast, medical', color='C1', linestyle='-',
         linewidth=2)
plt.plot(df_counts['year'], df_counts['breast_vs_all_ai'],
         label='breast, AI', color='C1', linestyle='--',
         linewidth=2)


plt.ylim(1, 4)
plt.xlim(2011, 2019)

plt.ylabel("Fraction of papers in category       ", size=14)
plt.xlabel("Publication year", size=14)
plt.yticks((1, 2, 3, 4), ('1%', '2%', '3%', '4%'))

#plt.legend(loc='upper left',
#           title="Topic and field\n of the publication",
#           #frameon=False,
#           borderaxespad=.2,
#           )
plt.text(2019.1, 3.7,
         'lung-cancer studies\npublished in AI',
         color='C0', va='center', size=12.5)

plt.text(2019.1, 2.43,
         'breast-cancer studies\npublished in AI',
         color='C1', va='center', size=12.5)

plt.text(2019.1, 1.4,
         'lung-cancer studies\npublished in medical\noncology',
         color='C0', va='bottom', size=12.5)

plt.text(2019.1, 1.3,
         'breast-cancer studies\npublished in medical\noncology',
         color='C1', va='top', size=12.5)

ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.tight_layout(pad=.01)
plt.savefig('figures/publication_counts.pdf', transparent=True)
plt.savefig('figures/publication_counts.eps')
