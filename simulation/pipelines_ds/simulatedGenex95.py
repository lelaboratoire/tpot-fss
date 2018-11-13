import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from tpot.builtins import DatasetSelector

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=95)

# Average CV score on the training set was:0.8027067111605488
exported_pipeline = make_pipeline(
    DatasetSelector(sel_subset=0, subset_list="subsets.csv"),
    Nystroem(gamma=0.6000000000000001, kernel="linear", n_components=7),
    GradientBoostingClassifier(learning_rate=0.1, max_depth=10, max_features=0.8500000000000001, min_samples_leaf=3, min_samples_split=10, n_estimators=100, subsample=0.45)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
