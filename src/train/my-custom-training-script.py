import logging
import json
import os
import glob
import sys
import sklearn
import pandas as pd
from sklearn import set_config
set_config(transform_output="pandas") 
set_config(display='diagram')
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler, FunctionTransformer
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer,make_column_selector, make_column_transformer


logging.basicConfig(filename='/opt/ml/output/data/logs-training.txt', level=logging.DEBUG)

if __name__ == '__main__':
    logging.debug('Hello my custom SageMaker init script!')

    my_model_weights = {
        "yes": [1, 2, 3],
        "no": [4]
    }
    f_output_model = open("/opt/ml/model/my-model-weights.json", "w")
    f_output_model.write(json.dumps(my_model_weights, sort_keys=True, indent=4))
    f_output_model.close()
    logging.debug('model weights dumped to my-model-weights.json')

    f_output_data = open("/opt/ml/output/data/environment-variables.json", "w")
    f_output_data.write(json.dumps(dict(os.environ), sort_keys=True, indent=4))
    f_output_data.close()
    logging.debug('environment variables dumped to environment-variables.json')

    f_output_data = open("/opt/ml/output/data/sys-args.json", "w")
    f_output_data.write(json.dumps(sys.argv[1:], sort_keys=True, indent=4))
    f_output_data.close()
    logging.debug('sys args dumped to sys-args.json')

    f_output_data = open("/opt/ml/output/data/sm-input-dir.json", "w")
    f_output_data.write(json.dumps(glob.glob("{}/*/*/*.*".format(os.environ['SM_INPUT_DIR']))))
    f_output_data.close()
    logging.debug('SM_INPUT_DIR files list dumped to sm-input-dir.json')
