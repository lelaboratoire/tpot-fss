import numpy as np
import pandas as pd
from sklearn.kernel_approximation import RBFSampler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from tpot.builtins import DatasetSelector

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=63)

# Average CV score on the training set was:0.7008379681127179
exported_pipeline = make_pipeline(
    DatasetSelector(sel_subset=1, subset_list="subsets.csv"),
    RBFSampler(gamma=0.2),
    DecisionTreeClassifier(criterion="gini", max_depth=6, min_samples_leaf=3, min_samples_split=8)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
