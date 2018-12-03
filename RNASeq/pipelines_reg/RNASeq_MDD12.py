import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=12)

# Average CV score on the training set was:0.7346086956521738
exported_pipeline = make_pipeline(
    Nystroem(gamma=0.25, kernel="linear", n_components=6),
    GradientBoostingClassifier(learning_rate=0.01, max_depth=1, max_features=0.55, min_samples_leaf=9, min_samples_split=12, n_estimators=100, subsample=0.5)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
