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


plt.figure(figsize=(3.8, 3))

plt.axvline(2016, color='.3', linestyle='-')
plt.text(2016.1, 4.95, 'Kaggle lung\n  challenge', rotation=90, va='top',
         size=13)


plt.plot(df_counts['year'], df_counts['lung_vs_all_medical'],
         label='lung, medical', color='C0', linestyle='-')
plt.plot(df_counts['year'], df_counts['lung_vs_all_ai'],
         label='lung, AI', color='C0', linestyle='--')

plt.plot(df_counts['year'], df_counts['breast_vs_all_medical'],
         label='breast, medical', color='C1', linestyle='-')
plt.plot(df_counts['year'], df_counts['breast_vs_all_ai'],
         label='breast, AI', color='C1', linestyle='--')


plt.ylim(1, 5)
plt.xlim(2011, 2019)

plt.ylabel("Fraction of papers in category       ", size=14)
plt.xlabel("Publication year", size=14)
plt.yticks((1, 2, 3, 4, 5), ('1%', '2%', '3%', '4%', '5%'))

plt.legend(loc='upper left',
           title="Topic and field\n of the publication",
           #frameon=False,
           borderaxespad=.2,
           )

plt.tight_layout(pad=.01)
plt.savefig('figures/publication_counts.pdf', transparent=True)
