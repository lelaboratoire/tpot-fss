import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Binarizer
from sklearn.tree import DecisionTreeClassifier
from tpot.builtins import DatasetSelector

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=36)

# Average CV score on the training set was:0.726608695652174
exported_pipeline = make_pipeline(
    DatasetSelector(sel_subset=4, subset_list="module23.csv"),
    Binarizer(threshold=0.15000000000000002),
    DecisionTreeClassifier(criterion="entropy", max_depth=3, min_samples_leaf=6, min_samples_split=15)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
