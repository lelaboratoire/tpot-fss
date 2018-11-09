import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Binarizer
from tpot.builtins import DatasetSelector

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=18)

# Average CV score on the training set was:0.743304347826087
exported_pipeline = make_pipeline(
    DatasetSelector(sel_subset=4, subset_list="module23.csv"),
    Binarizer(threshold=0.2),
    GradientBoostingClassifier(learning_rate=0.01, max_depth=7, max_features=0.15000000000000002, min_samples_leaf=5, min_samples_split=12, n_estimators=100, subsample=0.6500000000000001)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)