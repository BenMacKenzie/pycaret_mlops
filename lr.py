from pycaret.datasets import get_data
from pycaret.classification import *


data = get_data('titanic')
clf1 = setup(data, preprocess=True, target='Survived', session_id=124, log_experiment=True, experiment_name='tt6', log_data=True, silent=True)
lr = create_model('lr')
save_model(lr, model_name="titanic_lr")
