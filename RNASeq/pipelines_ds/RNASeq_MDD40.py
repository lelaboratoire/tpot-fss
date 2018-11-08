import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from tpot.builtins import DatasetSelector

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=40)

# Average CV score on the training set was:0.6911304347826086
exported_pipeline = make_pipeline(
    DatasetSelector(sel_subset=16, subset_list="module23.csv"),
    Normalizer(norm="max"),
    GradientBoostingClassifier(learning_rate=0.1, max_depth=5, max_features=1.0, min_samples_leaf=11, min_samples_split=2, n_estimators=100, subsample=0.9500000000000001)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
