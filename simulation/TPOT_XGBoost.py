from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from tpot.config import classifier_config_dict

import pandas as pd
import numpy as np
import os
import glob

accuracy_ls = []
n_gen = 100
n_pop = 100
personal_config = classifier_config_dict

dat_name = 'simulatedGenex'
tpot_data = pd.read_csv('simulatedGenex.csv')
Xdata = tpot_data.loc[:, tpot_data.columns != 'class']
Xdata = Xdata.drop(Xdata.columns[0], axis=1)
Ydata = tpot_data['class']

subset_df = pd.read_csv('subsets.csv')
all_features = ";".join(subset_df['Features'].tolist())
uniq_features = set(all_features.split(';')) # unique features in all subsets
overlap_features = list(uniq_features.intersection(set(list(Xdata.columns.values))))
X_subset = Xdata[overlap_features]

X_train, X_test, y_train, y_test = train_test_split(X_subset, Ydata, random_state = 1618,
                                                    train_size=0.75, test_size=0.25)

del X_subset
del Xdata
del Ydata
del tpot_data

seed = 1618
tpot = TPOTClassifier(generations=n_gen, config_dict=personal_config,
                      population_size=n_pop, verbosity=2, random_state=seed,
                      early_stop=10,
                      template='XGBClassifier')

tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))

tpot.export('XGBoost' + dat_name + str(seed) + '.py')
accuracy_ls.append([tpot._optimized_pipeline_score, tpot.score(X_test, y_test)])
accuracy_mat = pd.DataFrame(accuracy_ls, columns = ['Training CV Accuracy', 'Testing Accuracy'])
accuracy_mat.to_csv("XGBoost" + str(n_gen) + '_' + str(n_pop) + '_' + str(seed) + ".tsv", sep = "\t")

                      