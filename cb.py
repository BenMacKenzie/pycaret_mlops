from pycaret.datasets import get_data
from pycaret.classification import *
import numpy as np
import mlflow

#mlflow.set_tracking_uri("")
data = get_data('titanic')
data = data.drop(['Cabin'], axis=1)
data=data.dropna()
clf1 = setup(data, preprocess=False, target='Survived', session_id=124, log_experiment=False, silent=True)
X_train = get_config('X_train')
X_test = get_config('X_test')
cat_features  = np.where(X_train.dtypes != np.float32)[0]
cb = create_model('catboost', cat_features=cat_features)
predict_model(cb)
cb_results = pull()

cb.save_model('model.cbm')
X_train.to_csv("X_train.csv")
mlflow.create_experiment('test4')
mlflow.log_artifact('model.cbm')
mlflow.log_artifact('X_train.csv')
mlflow.log_param('cat_features', cat_features)
mlflow.log_params(cb.get_all_params())
cb_results.drop(['Model'], axis=1, inplace=True)
cb_results = cb_results.to_dict(orient='records')[0]
mlflow.log_metrics(cb_results)

save_config()