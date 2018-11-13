import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from tpot.builtins import DatasetSelector

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=97)

# Average CV score on the training set was:0.6668446421950316
exported_pipeline = make_pipeline(
    DatasetSelector(sel_subset=16, subset_list="subsets.csv"),
    PCA(iterated_power=1, svd_solver="randomized"),
    DecisionTreeClassifier(criterion="gini", max_depth=6, min_samples_leaf=2, min_samples_split=13)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
