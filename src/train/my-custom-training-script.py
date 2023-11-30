import logging
import os
import pandas as pd
import traceback
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

if __name__ == '__main__':
    try:
        logging.basicConfig(level=logging.DEBUG)

        logging.debug('Loading the dataset...')
        df = pd.read_csv('2023-24-UFS14-ML_mene/data/input/housing.csv')
        logging.debug('Dataset loaded.')

        X = df.drop(columns=["median_house_value"], axis=1)
        y = df['median_house_value']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object']).columns

        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('regressor', RandomForestRegressor(random_state=42))])

        logging.debug('Training the model...')
        pipeline.fit(X_train, y_train)
        logging.debug('Model training completed.')

        y_pred = pipeline.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        logging.debug(f"Mean Squared Error: {mse}")

        model_output_dir = os.environ['SM_MODEL_DIR']
        logging.debug(f"Model output directory: {model_output_dir}")

        logging.debug('Saving the model...')
        joblib.dump(pipeline, f"{model_output_dir}/model.joblib")
        logging.debug('Model saved.')

    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        logging.error(traceback.format_exc())
