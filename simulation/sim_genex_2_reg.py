from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split, cross_val_score
# from tpot.config.classifier_nn import classifier_config_nn
from sklearn.pipeline import make_pipeline
# from tpot.config import classifier_config_dict_light
from tpot.config import classifier_config_dict
from sklearn.neighbors import KNeighborsClassifier

import pandas as pd
import numpy as np
import os
import glob
# personal_config = classifier_config_dict_light
personal_config = classifier_config_dict
personal_config['tpot.builtins.DatasetSelector'] = {
    'subset_list': ['subsets.csv'],
    'sel_subset': range(19)
}


accuracy_ls = []
n_gen = 100
n_pop = 100

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

for seed in range(100):
    # X_train, X_test, y_train, y_test = train_test_split(X_subset, Ydata, random_state=seed,
    #                                                     train_size=0.75, test_size=0.25)
    tpot = TPOTClassifier(generations=n_gen, config_dict=personal_config,
                          population_size=n_pop, verbosity=2, random_state=seed,
                          early_stop=10,
    #                       template = 'DatasetSelector-CombineDFs-Transformer-Classifier')
                          template='Transformer-Classifier')
    tpot.fit(X_train, y_train)
    print(tpot.score(X_test, y_test))

    tpot.export('pipelines_reg/' + dat_name + str(seed) + '.py')
    accuracy_ls.append([tpot._optimized_pipeline_score, tpot.score(X_test, y_test)])
    accuracy_mat = pd.DataFrame(accuracy_ls, columns = ['Training CV Accuracy', 'Testing Accuracy'])
    accuracy_mat.to_csv("accuracies_reg/" + str(n_gen) + '_' + str(n_pop) + '_' + str(seed) + ".tsv", sep = "\t")
