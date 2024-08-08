import configparser
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import logging
import time
import newrelic.agent
from ml_performance_monitoring.monitor import MLPerformanceMonitoring

# Initialize New Relic agent with the configuration file
newrelic.agent.initialize('newrelic.ini')

# Load configuration from the config.ini file
config = configparser.ConfigParser()
config.read('newrelic.ini')

# Check if the 'newrelic' section and 'license_key' exist
if 'newrelic' in config and 'license_key' in config['newrelic']:
    insert_key = config['newrelic']['license_key']
else:
    logging.error("Missing 'newrelic' section or 'license_key' in config.ini")
    raise KeyError("Missing 'newrelic' section or 'license_key' in config.ini")

# Configure logging to display information with timestamps
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Start timing the entire script execution
start_time = time.time()

# Load the dataset from a Parquet file
logging.info('Loading dataset...')
load_start_time = time.time()
data = pd.read_parquet('rideshare_data.parquet')
load_end_time = time.time()
logging.info(f'Dataset loaded. Shape: {data.shape}. Time taken: {load_end_time - load_start_time:.2f} seconds')
newrelic.agent.record_custom_metric('Custom/LoadDatasetTime', load_end_time - load_start_time)

# Print the column names to verify the dataset structure
logging.info(f'Column names: {data.columns}')

# Preprocess the data
logging.info('Preprocessing data...')
preprocess_start_time = time.time()
# Select relevant columns and fill missing values
data = data[['pickup_location', 'dropoff_location', 'trip_length', 'passenger_fare', 'total_ride_time']]
data.fillna(method='ffill', inplace=True)
preprocess_end_time = time.time()
logging.info(f'Data preprocessing completed. Time taken: {preprocess_end_time - preprocess_start_time:.2f} seconds')
newrelic.agent.record_custom_metric('Custom/PreprocessDataTime', preprocess_end_time - preprocess_start_time)

# Define features and target variable
features = ['pickup_location', 'dropoff_location', 'trip_length', 'passenger_fare']
X = data[features]
y = data['total_ride_time']

# Split the data into training and testing sets
logging.info('Splitting data into training and testing sets...')
split_start_time = time.time()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
split_end_time = time.time()
logging.info(f'Data split completed. Time taken: {split_end_time - split_start_time:.2f} seconds')
newrelic.agent.record_custom_metric('Custom/SplitDataTime', split_end_time - split_start_time)

# Train a Linear Regression model
logging.info('Training Linear Regression model...')
train_start_time = time.time()
model = LinearRegression()
model.fit(X_train, y_train)
train_end_time = time.time()
logging.info(f'Model training completed. Time taken: {train_end_time - train_start_time:.2f} seconds')
newrelic.agent.record_custom_metric('Custom/TrainModelTime', train_end_time - train_start_time)

# Evaluate the model
logging.info('Evaluating model...')
eval_start_time = time.time()
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
eval_end_time = time.time()
logging.info(f'Model evaluation completed. Mean Squared Error: {mse:.2f}. Time taken: {eval_end_time - eval_start_time:.2f} seconds')
newrelic.agent.record_custom_metric('Custom/EvaluateModelTime', eval_end_time - eval_start_time)
newrelic.agent.record_custom_metric('Custom/MeanSquaredError', mse)

# Initialize MLPerformanceMonitoring with the insert key from the config file
metadata = {"environment": "notebook"}
model_version = "1.0"
features_columns = ['pickup_location', 'dropoff_location', 'trip_length', 'passenger_fare']
labels_columns = ['total_ride_time']

ml_monitor = MLPerformanceMonitoring(
    insert_key=insert_key,  # Use the insert key from the config file
    model_name="RideSharePredictionModel",
    metadata=metadata,
    features_columns=features_columns,
    labels_columns=labels_columns,
    event_client_host="insights-collector.newrelic.com",
    metric_client_host="metric-api.newrelic.com",
    model_version=model_version
)

# Make a prediction for a sample input
logging.info('Making a sample prediction...')
predict_start_time = time.time()
sample_input = {
    'pickup_location': [1],
    'dropoff_location': [2],
    'trip_length': [5.0],
    'passenger_fare': [20.0]
}
sample_input_df = pd.DataFrame(sample_input)
sample_prediction = model.predict(sample_input_df)
predict_end_time = time.time()
logging.info(f'Sample prediction completed. Time taken: {predict_end_time - predict_start_time:.2f} seconds')
newrelic.agent.record_custom_metric('Custom/SamplePredictionTime', predict_end_time - predict_start_time)

# End timing the entire script execution
end_time = time.time()
logging.info(f'Total execution time: {end_time - start_time:.2f} seconds')
newrelic.agent.record_custom_metric('Custom/TotalExecutionTime', end_time - start_time)