import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=1618)

# Average CV score on the training set was:0.5854060066740823
exported_pipeline = XGBClassifier(learning_rate=1.0, max_depth=5, min_child_weight=7, n_estimators=100, nthread=1, subsample=0.4)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
