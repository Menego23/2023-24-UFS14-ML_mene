import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import os
import logging
import traceback

if __name__ == '__main__':
    try:
        logging.basicConfig(level=logging.DEBUG)

        logging.debug('Loading the dataset...')
        df = pd.read_csv('2023-24-UFS14-ML_mene/data/input/housing.csv')
        logging.debug('Dataset loaded.')

        X = df.drop(columns=["median_house_value"], axis=1)
        y = df['median_house_value']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(random_state=42)

        logging.debug('Training the model...')
        model.fit(X_train, y_train)
        logging.debug('Model training completed.')

        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        logging.debug(f"Mean Squared Error: {mse}")

        model_output_dir = os.environ['SM_MODEL_DIR']
        logging.debug(f"Model output directory: {model_output_dir}")

        logging.debug('Saving the model...')
        joblib.dump(model, f"{model_output_dir}/model.joblib")
        logging.debug('Model saved.')

    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        logging.error(traceback.format_exc())
        
        
