from pycaret.datasets import get_data
from pycaret.classification import *
import numpy as np
import mlflow
import os



def save_model_and_metadata(expirement_name, model, cat_features_index):
    exp_id = mlflow.create_experiment(expirement_name)
    mlflow.start_run(experiment_id=exp_id)
    X_train = get_config('X_train')
    y_train = get_config('y_train')
    X_test = get_config('X_test')
    y_test = get_config('y_test')
    X_train.join(y_train).to_csv("Train.csv")
    X_test.join(y_test).to_csv("Test.csv")

    save_config('config.pkl')

    mlflow.log_artifact('Train.csv')
    mlflow.log_artifact('Test.csv')
    mlflow.log_artifact('config.pkl')
    os.remove("Train.csv")
    os.remove("Test.csv")

    os.remove('config.pkl')
    cat_features = []
    columns = X_train.columns.to_list()
    for i in cat_features_index:
        cat_features.append(columns[i])

    mlflow.log_param('cat_features', cat_features)
    mlflow.log_params(model.get_all_params())
    predict_model(model)
    cb_results = pull()
    cb_results.drop(['Model'], axis=1, inplace=True)
    cb_results = cb_results.to_dict(orient='records')[0]
    mlflow.log_metrics(cb_results)
    final_model = finalize_model(model)
    final_model.save_model('model.cbm')
    mlflow.log_artifact('model.cbm')
    os.remove('model.cbm')
    mlflow.end_run()

def rebuild_model(expirement_id):
    # navigate to the one and only run subdirectory under the expirement
    # load_config(f"{artifacts}/config.pkl"...this should make X_train, X_test, target and session id available
    # could acquire train and test in artifact directory also....split off the target and
    # then use set_config()...
    # read in all params from params directory.  this are individual files..
    # need to convert to a dictionary and pass to create_model('catboost', dict)
    #
    pass



def build_model(name):
    #this would be done from notebook
    data = get_data('titanic')
    data = data.drop(['Cabin'], axis=1)
    data=data.dropna()
    clf1 = setup(data, preprocess=False, target='Survived', session_id=124, log_experiment=False, silent=True)
    X_train = get_config('X_train')
    cat_features  = np.where(X_train.dtypes != np.float32)[0]
    cb = create_model('catboost', cat_features=cat_features)
    cb_tune = tune_model(cb)
    save_model_and_metadata(name, cb_tune, cat_features)


build_model("feb25_2")