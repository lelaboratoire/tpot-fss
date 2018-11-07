import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from tpot.builtins import DatasetSelector, OneHotEncoder

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=9)

# Average CV score on the training set was:0.7101449275362318
exported_pipeline = make_pipeline(
    DatasetSelector(sel_subset=4, subset_list="module23.csv"),
    OneHotEncoder(minimum_fraction=0.2, sparse=False, threshold=10),
    ExtraTreesClassifier(bootstrap=False, criterion="gini", max_features=0.35000000000000003, min_samples_leaf=4, min_samples_split=10, n_estimators=100)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
