import numpy as np
import pandas as pd
from sklearn.cluster import FeatureAgglomeration
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from tpot.builtins import DatasetSelector

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=88)

# Average CV score on the training set was:0.6741564701520207
exported_pipeline = make_pipeline(
    DatasetSelector(sel_subset=0, subset_list="subsets.csv"),
    FeatureAgglomeration(affinity="l1", linkage="average"),
    GradientBoostingClassifier(learning_rate=1.0, max_depth=4, max_features=0.2, min_samples_leaf=10, min_samples_split=20, n_estimators=100, subsample=0.8)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
