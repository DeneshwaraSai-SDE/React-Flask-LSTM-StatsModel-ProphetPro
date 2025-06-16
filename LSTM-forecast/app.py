'''
Goal of LSTM microservice:
1. LSTM microservice will accept the GitHub data from Flask microservice and will forecast the data for next 1 year based on past 30 days
2. It will also plot three different graph (i.e.  "Model Loss", "LSTM Generated Data", "All Issues Data") using matplot lib 
3. This graph will be stored as image in Google Cloud Storage.
4. The image URL are then returned back to Flask microservice.
'''
# Import all the required packages
from flask import Flask, jsonify, request, make_response
import os
from dateutil import *
from datetime import timedelta
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import time
from flask_cors import CORS

import statsmodels
import statsmodels.api as sm

from prophet import Prophet 

# Tensorflow (Keras & LSTM) related packages
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

# Import required storage package from Google Cloud Storage
from google.cloud import storage

# Initilize flask app
app = Flask(__name__)
# Handles CORS (cross-origin resource sharing)
CORS(app)
# Initlize Google cloud storage client
client = storage.Client()

# Add response headers to accept all types of  requests

def build_preflight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type")
    response.headers.add("Access-Control-Allow-Methods",
                         "PUT, GET, POST, DELETE, OPTIONS")
    return response

#  Modify response headers when returning to the origin

def build_actual_response(response):
    response.headers.set("Access-Control-Allow-Origin", "*")
    response.headers.set("Access-Control-Allow-Methods",
                         "PUT, GET, POST, DELETE, OPTIONS")
    return response

'''
API route path is  "/api/forecast"
This API will accept only POST request
'''

@app.route('/api/pulls', methods=['POST'])
def pulls():
    payload = request.get_json()
    pull_data = payload["pull"]
    repository = payload["repo"]
    request_type = payload["type"]
    print("Request Type:", request_type)

    df_raw = pd.DataFrame(pull_data)

    df_grouped = df_raw.groupby("created_at", as_index=False).count()
    df_trimmed = df_grouped[["created_at", "pull_req_number"]]
    df_trimmed.columns = ['ds', 'y']
    df_trimmed['ds'] = pd.to_datetime(df_trimmed['ds'])

    time_series = df_trimmed.to_numpy()
    x_vals = np.array([time.mktime(timestamp[0].timetuple()) for timestamp in time_series])
    y_vals = np.array([val[1] for val in time_series])
    print("Y values:", y_vals)

    start_date = df_trimmed['ds'].min()
    total_days = (df_trimmed['ds'].max() - start_date).days + 1
    y_series_full = [0] * total_days
    all_dates = pd.Series([start_date + timedelta(days=i) for i in range(total_days)])

    for date, val in zip(df_trimmed['ds'], y_vals):
        y_series_full[(date - start_date).days] = val

    y_series_array = np.array(y_series_full).astype('float32').reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaled = scaler.fit_transform(y_series_array)

    split_index = int(len(y_scaled) * 0.8)
    training_data, testing_data = y_scaled[:split_index], y_scaled[split_index:]

    print(f"Training size: {len(training_data)}, Testing size: {len(testing_data)}")

    def preprocess(data, window_size=1):
        X, Y = [], []
        for i in range(len(data) - window_size - 1):
            X.append(data[i:i + window_size, 0])
            Y.append(data[i + window_size, 0])
        return np.array(X), np.array(Y)

    lookback_period = min(30, len(testing_data) - 2)
    if len(testing_data) > lookback_period + 1:
        X_test, y_test = preprocess(testing_data, lookback_period)
    else:
        print("Insufficient testing data for given lookback period.")

    X_train, y_train = preprocess(training_data, lookback_period)

    if len(X_train) > 0:
        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    if len(X_test) > 0:
        X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    print(f"Shapes: X_train={X_train.shape}, X_test={X_test.shape}, y_train={y_train.shape}, y_test={y_test.shape}")

    # Define and train LSTM model
    model = Sequential()
    model.add(LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    history = model.fit(X_train, y_train, epochs=20, batch_size=70, validation_data=(X_test, y_test),
                        callbacks=[EarlyStopping(monitor='val_loss', patience=10)], verbose=1, shuffle=False)

    image_base_url = os.environ.get('BASE_IMAGE_PATH', 'https://storage.googleapis.com/lstm-storage/')

    image_dir = "static/images/"
    loss_img = f"model_loss_{request_type}_{repository}.png"
    forecast_img = f"lstm_forecast_{request_type}_{repository}.png"
    full_data_img = f"complete_data_{request_type}_{repository}.png"

    loss_url = image_base_url + loss_img
    forecast_url = image_base_url + forecast_img
    data_url = image_base_url + full_data_img

    # Plot and save model loss
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training vs Validation Loss (Pull Requests)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(image_dir + loss_img)

    y_predicted = model.predict(X_test)

    # Plot forecast results
    plt.figure(figsize=(10, 4))
    plt.plot(range(len(y_train)), y_train, 'g', label="Historical Data")
    plt.plot(range(len(y_train), len(y_train) + len(y_test)), y_test, marker='.', label="Actual")
    plt.plot(range(len(y_train), len(y_train) + len(y_test)), y_predicted, 'r', label="Forecast")
    plt.title('LSTM Forecast of Pull Requests')
    plt.xlabel('Time')
    plt.ylabel('Pull Requests')
    plt.legend()
    plt.savefig(image_dir + forecast_img)

    # Plot full time-series pull request data
    plt.figure(figsize=(10, 4))
    plt.plot(mdates.date2num(all_dates), y_scaled, 'purple', marker='.')
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gca().xaxis.set_major_formatter(mdates.AutoDateFormatter(mdates.AutoDateLocator()))
    plt.title('Complete Pull Request Timeline')
    plt.xlabel('Date')
    plt.ylabel('Pull Requests')
    plt.savefig(image_dir + full_data_img)

    bucket_name = os.environ.get('BUCKET_NAME', 'lstm-storage')
    bucket = client.get_bucket(bucket_name)
    for file in [loss_img, forecast_img, full_data_img]:
        blob = bucket.blob(file)
        blob.upload_from_filename(image_dir + file)

    return jsonify({
        "model_loss_image_url": loss_url,
        "lstm_generated_image_url": forecast_url,
        "all_issues_data_image": data_url
    })


@app.route('/api/commits', methods=['POST'])
def commits():
    payload = request.get_json()
    commit_entries = payload["pull"]
    repository = payload["repo"]
    category = payload["type"]
    print("Processing Type:", category)

    commit_df = pd.DataFrame(commit_entries)
    commit_count_by_date = commit_df.groupby("created_at", as_index=False).count()
    processed_df = commit_count_by_date[["created_at", 'commit_number']]
    processed_df.columns = ['ds', 'y']
    print(processed_df)
    processed_df['ds'] = pd.to_datetime(processed_df['ds'])

    raw_data = processed_df.to_numpy()
    timestamp_array = np.array([time.mktime(entry[0].timetuple()) for entry in raw_data])
    commit_array = np.array([entry[1] for entry in raw_data])
    print("Commit Count Array:", commit_array)

    start_date = processed_df['ds'].min()

    # Generate daily commit counts with zeros for missing days
    complete_days = [start_date + timedelta(days=i) for i in range((max(processed_df['ds']) - start_date).days + 1)]
    daily_commit_counts = [0] * len(complete_days)
    for date, count in zip(processed_df['ds'], commit_array):
        daily_commit_counts[(date - start_date).days] = count

    commit_series = np.array(daily_commit_counts, dtype='float32').reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(commit_series)

    train_len = int(len(scaled_data) * 0.80)
    train_set, test_set = scaled_data[:train_len], scaled_data[train_len:]
    print('Train Length:', len(train_set), "Test Length:", len(test_set))

    def preprocess(data, window=1):
        input_seq, target_seq = [], []
        for i in range(len(data) - window - 1):
            input_seq.append(data[i:(i + window), 0])
            target_seq.append(data[i + window, 0])
        return np.array(input_seq), np.array(target_seq)

    # Set window size (number of days the model looks back)
    look_back = min(30, len(test_set) - 2)
    if len(test_set) > look_back + 1:
        X_test, y_test = preprocess(test_set, look_back)
    else:
        print("Insufficient test data for selected look_back window.")
        X_test, y_test = np.empty((0,)), np.empty((0,))

    X_train, y_train = preprocess(train_set, look_back)

    # if X_train.size == 0 or X_test.size == 0:
    #     return jsonify({"error": "Not enough data for LSTM training."}), 400

    # Reshape sequences for LSTM input
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    print('Shape Summary:', X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # Build and train LSTM model
    model = Sequential()
    model.add(LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    training_log = model.fit(
        X_train, y_train,
        epochs=20, batch_size=70,
        validation_data=(X_test, y_test),
        callbacks=[EarlyStopping(monitor='val_loss', patience=10)],
        verbose=1, shuffle=False
    )
    
    future_days = 365
    future_input = test_set[-look_back:].reshape(1, 1, look_back)  # Start from last window in test set
    future_forecast = []

    for _ in range(future_days):
        pred = model.predict(future_input)[0][0]
        future_forecast.append(pred)
        # Update input for next prediction
        future_seq = np.append(future_input[0][0][1:], pred)
        future_input = future_seq.reshape(1, 1, look_back)

    # Rescale future forecast back to original scale
    future_forecast_rescaled = scaler.inverse_transform(np.array(future_forecast).reshape(-1, 1)).flatten()

    last_date = complete_days[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, future_days + 1)]

    # Plot future predictions
    plt.figure(figsize=(12, 5))
    plt.plot(complete_days, commit_series, label='Historical Commits')
    plt.plot(future_dates, future_forecast_rescaled, color='red', linestyle='--', label='365-day Forecast')
    plt.title('Commit Forecast - Next 365 Days')
    plt.xlabel('Date')
    plt.ylabel('Commit Count')
    plt.legend()
    future_plot = f"lstm_future_forecast_{category}_{repository}.png"
    plt.savefig(image_dir + future_plot)
    # Upload to cloud
    # blob = cloud_bucket.blob(future_plot)
    # blob.upload_from_filename(filename=image_dir + future_plot)


    # Define image paths
    base_url = os.environ.get('BASE_IMAGE_PATH', 'https://storage.googleapis.com/lstm-storage/')
    
    bucket_id = os.environ.get('BUCKET_NAME', 'lstm-storage')

    image_dir = "static/images/"
    loss_image = f"model_loss_{category}_{repository}.png"
    prediction_image = f"lstm_generated_data_{category}_{repository}.png"
    data_image = f"all_issues_data_{category}_{repository}.png"

    # Plot model loss
    plt.figure(figsize=(8, 4))
    plt.plot(training_log.history['loss'], label='Training Loss')
    plt.plot(training_log.history['val_loss'], label='Validation Loss')
    plt.title('LSTM Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(image_dir + loss_image)

    # Plot predictions
    future_preds = model.predict(X_test)
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(0, len(y_train)), y_train, 'g', label='History')
    plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_test, marker='.', label='Actual')
    plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), future_preds, 'r', label='Forecast')
    plt.title('LSTM Predictions')
    plt.xlabel('Time Steps')
    plt.ylabel('Normalized Commit Count')
    plt.legend()
    plt.savefig(image_dir + prediction_image)

    # Plot original time series data
    plt.figure(figsize=(10, 4))
    date_nums = mdates.date2num(complete_days)
    plt.plot(date_nums, scaled_data, color='purple', marker='.')
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gca().xaxis.set_major_formatter(mdates.AutoDateFormatter(mdates.AutoDateLocator()))
    plt.title('Commit Time Series')
    plt.xlabel('Date')
    plt.ylabel('Normalized Commits')
    plt.savefig(image_dir + data_image)

    # Upload generated charts to cloud storage
    cloud_bucket = client.get_bucket(bucket_id)
    for file_name in [loss_image, prediction_image, data_image, future_plot]:
        blob = cloud_bucket.blob(file_name)
        blob.upload_from_filename(filename=image_dir + file_name)

    return jsonify({
        "model_loss_image_url": base_url + loss_image,
        "lstm_generated_image_url": base_url + prediction_image,
        "all_issues_data_image": base_url + data_image,
        "future_commits_Plots": base_url + future_plot
    })


@app.route('/api/prophetcommits', methods=['POST'])
def prophetcommits():
    payload = request.get_json()
    commit_data = payload["pull"]
    repository = payload["repo"]
    analysis_type = payload["type"]
    print("Forecast Type:", analysis_type)

    # Environment config or defaults

    cloud_image_base = os.environ.get('BASE_IMAGE_PATH', 'https://storage.googleapis.com/lstm-storage/')
    storage_bucket_name = os.environ.get('BUCKET_NAME', 'lstm-storage')

    local_dir = "static/images/"
    forecast_img_filename = f"prophet_commit_forecast_{analysis_type}_{repository}.png"
    components_img_filename = f"prophet_commit_forecast_components_{analysis_type}_{repository}.png"

    forecast_img_url = f"{cloud_image_base}{forecast_img_filename}"
    components_img_url = f"{cloud_image_base}{components_img_filename}"

    # Prepare and model the data
    df_raw = pd.DataFrame(commit_data)
    grouped_df = df_raw.groupby(['created_at'], as_index=False).count()
    print(grouped_df.head())
    grouped_df.columns = ['created_at', 'count']
    time_series_df = grouped_df.rename(columns={'created_at': 'ds', 'count': 'y'})

    model = Prophet(daily_seasonality=True)
    model.fit(time_series_df)
    
    future_dates = model.make_future_dataframe(periods=365)
    forecast_result = model.predict(future_dates)

    fig_forecast = model.plot(forecast_result)
    fig_components = model.plot_components(forecast_result)

    fig_forecast.savefig(os.path.join(local_dir, forecast_img_filename))
    fig_components.savefig(os.path.join(local_dir, components_img_filename))

    # Upload generated images to GCS
    bucket = client.get_bucket(storage_bucket_name)
    bucket.blob(forecast_img_filename).upload_from_filename(os.path.join(local_dir, forecast_img_filename))
    bucket.blob(components_img_filename).upload_from_filename(os.path.join(local_dir, components_img_filename))

    # Response to client
    response_payload = {
        "prophet_commit_forecast_url": forecast_img_url,
        "prophet_commit_components_url": components_img_url
    }
    print("-------------- DONNEEE sending back ---------------")
    return jsonify(response_payload)


@app.route('/api/fbprophetpull', methods=['POST'])
def fbprophetpull():
    payload = request.get_json()
    pull_requests = payload["pull"]
    repository = payload["repo"]
    forecast_type = payload["type"]

    base_url = os.environ.get('BASE_IMAGE_PATH', 'https://storage.googleapis.com/lstm-storage/')
    bucket_name = os.environ.get('BUCKET_NAME', 'lstm-storage')
    local_dir = "static/images/"
    print("DONEEEE 1")

    forecast_image_name = f"fbprophet_pulls_forecast_{forecast_type}_{repository}.png"
    forecast_components_image_name = f"fbprophet_pulls_forecast_components_{forecast_type}_{repository}.png"

    forecast_url = base_url + forecast_image_name
    components_url = base_url + forecast_components_image_name

    df = pd.DataFrame(pull_requests)
    daily_counts = df.groupby(['created_at'], as_index=False).count()
    time_series_data = daily_counts[['created_at', 'pull_req_number']]
    time_series_data.columns = ['ds', 'y']

    print(df.head())
    # Forecast modeling
    prophet_model = Prophet(daily_seasonality=True)
    prophet_model.fit(time_series_data, stan_backend='CMDSTANPY', algorithm='LBFGS', n_chains=1)
    future_dates = prophet_model.make_future_dataframe(periods=365)
    forecast_result = prophet_model.predict(future_dates)

    # Plotting and saving figures
    forecast_plot = prophet_model.plot(forecast_result)
    components_plot = prophet_model.plot_components(forecast_result)

    forecast_plot.savefig(os.path.join(local_dir, forecast_image_name))
    components_plot.savefig(os.path.join(local_dir, forecast_components_image_name))

    # Upload to Google Cloud Storage
    bucket = client.get_bucket(bucket_name)
    for filename in [forecast_image_name, forecast_components_image_name]:
        blob = bucket.blob(filename)
        blob.upload_from_filename(os.path.join(local_dir, filename))

    print("Done with PULLS")
    # Return URLs as JSON
    return jsonify({
        "fbprophet_pulls_forecast_url": forecast_url,
        "fbprophet_pulls_forecast_components_url": components_url
    })


@app.route('/api/fbprophetIssuesCls', methods=['POST'])
def fbprophetIssuesCls():
    request_data = request.get_json()
    forecast_type = request_data["type"]
    repository_name = request_data["repo"]
    issues_data = request_data["issues"]

    base_image_url = os.environ.get('BASE_IMAGE_PATH', 'https://storage.googleapis.com/lstm-storage/')

    gcs_bucket_name = os.environ.get('BUCKET_NAME', 'lstm-storage')

    local_image_dir = "static/images/"

    forecast_image_filename = f"forecast_plot_{forecast_type}_{repository_name}.png"
    forecast_components_filename = f"forecast_components_plot_{forecast_type}_{repository_name}.png"

    forecast_image_url = base_image_url + forecast_image_filename
    forecast_components_url = base_image_url + forecast_components_filename

    issues_df = pd.DataFrame(issues_data)
    print(issues_df.head())
    grouped_issues = issues_df.groupby(['closed_at'], as_index=False).count()
    timeseries_df = grouped_issues[['closed_at', 'issue_number']].rename(columns={'closed_at': 'ds', 'issue_number': 'y'})

    model = Prophet(daily_seasonality=True)
    model.fit(timeseries_df)
    future_dates = model.make_future_dataframe(periods=365)
    forecast_results = model.predict(future_dates)

    forecast_fig = model.plot(forecast_results)
    components_fig = model.plot_components(forecast_results)

    forecast_fig.savefig(os.path.join(local_image_dir, forecast_image_filename))
    components_fig.savefig(os.path.join(local_image_dir, forecast_components_filename))

    bucket = client.get_bucket(gcs_bucket_name)

    forecast_blob = bucket.blob(forecast_image_filename)
    forecast_blob.upload_from_filename(os.path.join(local_image_dir, forecast_image_filename))

    components_blob = bucket.blob(forecast_components_filename)
    components_blob.upload_from_filename(os.path.join(local_image_dir, forecast_components_filename))

    response = {
        "forecast_plot_url": forecast_image_url,
        "forecast_components_plot_url": forecast_components_url
    }

    print("------------- DONE WITH CLOSED Prophet ------------- ")
    return jsonify(response)


@app.route('/api/fbprophetCtd', methods=['POST'])
def fbprophetCtd():
    request_data = request.get_json()
    issue_type = request_data["type"]
    repository_name = request_data["repo"]
    issues_list = request_data["issues"]

    base_image_url = os.environ.get('BASE_IMAGE_PATH', 'https://storage.googleapis.com/lstm-storage/')
    bucket_name = os.environ.get('BUCKET_NAME', 'lstm-storage')

    local_image_dir = "static/images/"
    forecast_filename = f"fbprophet_forecast_{issue_type}_{repository_name}.png"
    forecast_url = base_image_url + forecast_filename

    components_filename = f"fbprophet_forecast_components_{issue_type}_{repository_name}.png"
    components_url = base_image_url + components_filename

    # Convert and prepare data for forecasting
    issues_df = pd.DataFrame(issues_list)
    print(issues_df.head())
    issue_counts = issues_df.groupby('created_at', as_index=False).count()
    forecast_data = issue_counts[['created_at', 'issue_number']]
    forecast_data.columns = ['ds', 'y']

    prophet_model = Prophet(daily_seasonality=True)
    prophet_model.fit(forecast_data)
    future_dates = prophet_model.make_future_dataframe(periods=365)
    forecast_results = prophet_model.predict(future_dates)

    forecast_fig = prophet_model.plot(forecast_results)
    components_fig = prophet_model.plot_components(forecast_results)
    forecast_fig.savefig(os.path.join(local_image_dir, forecast_filename))
    components_fig.savefig(os.path.join(local_image_dir, components_filename))

    # Upload forecast images to Google Cloud Storage
    gcs_bucket = client.get_bucket(bucket_name)
    forecast_blob = gcs_bucket.blob(forecast_filename)
    forecast_blob.upload_from_filename(os.path.join(local_image_dir, forecast_filename))

    components_blob = gcs_bucket.blob(components_filename)
    components_blob.upload_from_filename(os.path.join(local_image_dir, components_filename))

    # Return URLs of the generated images
    response_payload = {
        "fbprophet_forecast_url": forecast_url,
        "fbprophet_forecast_components_url": components_url
    }
    
    return jsonify(response_payload)


@app.route('/api/statmodelpulls', methods=['POST'])
def statmodelpulls():
    request_data = request.get_json()
    pull_data = request_data.get("pull", [])
    repository = request_data.get("repo", "default_repo")
    pull_type = request_data.get("type", "general")

    cloud_image_base_url = os.environ.get('BASE_IMAGE_PATH', 'https://storage.googleapis.com/lstm-storage/')
    gcs_bucket_name = os.environ.get('BUCKET_NAME', 'lstm-storage')
    local_image_dir = "static/images/"

    obs_image_filename = f"stats_observation_{pull_type}_{repository}.png"
    forecast_image_filename = f"stats_forecast_{pull_type}_{repository}.png"

    obs_image_url = cloud_image_base_url + obs_image_filename
    forecast_image_url = cloud_image_base_url + forecast_image_filename

    pulledDataDF = pd.DataFrame(pull_data)
    grouped_df = pulledDataDF.groupby(['created_at'], as_index=False).count()
    timeseries_df = grouped_df[['created_at', 'pull_req_number']]
    timeseries_df.columns = ['ds', 'y']
    timeseries_df['ds'] = pd.to_datetime(timeseries_df['ds'])
    timeseries_df.set_index('ds', inplace=True)

    seasonal_period = max(2, len(timeseries_df) // 2)
    decomposition = sm.tsa.seasonal_decompose(timeseries_df['y'], period=seasonal_period)
    fig_obs = decomposition.plot()
    fig_obs.set_size_inches(12, 7)
    plt.title("Pull Request Activity - Seasonal Decomposition")
    fig_obs.get_figure().savefig(os.path.join(local_image_dir, obs_image_filename))

    model = sm.tsa.ARIMA(timeseries_df['y'], order=(1, 0, 0))
    results = model.fit()

    future_steps = 365
    forecast = results.forecast(steps=future_steps)

    last_date = timeseries_df.index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, future_steps + 1)]
    forecast_df = pd.DataFrame({'y': forecast, 'ds': future_dates}).set_index('ds')

    combined_df = pd.concat([timeseries_df[['y']], forecast_df.rename(columns={'y': 'forecast'})], axis=1)

    plt.figure(figsize=(14, 7))
    plt.plot(combined_df.index, combined_df['y'], label='Observed')
    plt.plot(combined_df.index, combined_df['forecast'], label='Forecast', linestyle='--')
    plt.title("Pull Request Activity - Forecast Next 365 Days")
    plt.legend(rotation = 90)
    plt.grid(True)
    plt.savefig(os.path.join(local_image_dir, forecast_image_filename))

    bucket = client.get_bucket(gcs_bucket_name)
    for filename in [obs_image_filename, forecast_image_filename]:
        blob = bucket.blob(filename)
        blob.upload_from_filename(os.path.join(local_image_dir, filename))

    return jsonify({
        "stats_observation_url": obs_image_url,
        "stats_forecast_url": forecast_image_url
    })


@app.route('/api/allPullsModel', methods=['POST'])
def allPullsModel():
    print("Started Pulls Code:")
    body = request.get_json()
    pull_response = body["pull"]
    repo_name = body["repo"]
    type = body["type"]
    print("type:", type) 

    BASE_IMAGE_PATH = os.environ.get('BASE_IMAGE_PATH', 'https://storage.googleapis.com/lstm-storage/')
    BUCKET_NAME = os.environ.get('BUCKET_NAME', 'lstm-storage')

    LOCAL_IMAGE_PATH = "static/images/"
    FORECAST_IMAGE_NAME = "fbprophet_forecast_pulls_" + type +"_"+ repo_name + ".png"
    FORECAST_IMAGE_URL = BASE_IMAGE_PATH + FORECAST_IMAGE_NAME

    FORECAST_COMPONENTS_IMAGE_NAME = "fbprophet_forecast_pulls_components_" + type +"_" + repo_name + ".png"
    FORECAST_COMPONENTS_IMAGE_URL = BASE_IMAGE_PATH + FORECAST_COMPONENTS_IMAGE_NAME    

    pullsDF = pd.DataFrame(pull_response)
    pullsDF['created_at'] = pd.to_datetime(pullsDF['created_at'])

    daily_commits = pullsDF.groupby(pullsDF['created_at'].dt.date).size().reset_index(name='y')
    daily_commits.rename(columns={'created_at': 'ds'}, inplace=True)
    print(daily_commits.head())

    model = Prophet(daily_seasonality=True)
    model.fit(daily_commits, algorithm='LBFGS', iter=1000)

    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)

    forcast_fig1 = model.plot(forecast)
    forcast_fig2 = model.plot_components(forecast)
    forcast_fig1.savefig(LOCAL_IMAGE_PATH + FORECAST_IMAGE_NAME)
    forcast_fig2.savefig(LOCAL_IMAGE_PATH + FORECAST_COMPONENTS_IMAGE_NAME)

    bucket = client.get_bucket(BUCKET_NAME)

    new_blob = bucket.blob(FORECAST_IMAGE_NAME)
    new_blob.upload_from_filename(filename=LOCAL_IMAGE_PATH + FORECAST_IMAGE_NAME)

    new_blob = bucket.blob(FORECAST_COMPONENTS_IMAGE_NAME)
    new_blob.upload_from_filename(filename=LOCAL_IMAGE_PATH + FORECAST_COMPONENTS_IMAGE_NAME)

    ## ---------------------------------------------------------------------------------------------------- ##
    #### FOR LSTM

    daily_pulls_lstm = daily_commits.copy()
    daily_pulls_lstm['ds'] = pd.to_datetime(daily_pulls_lstm['ds'])
    daily_pulls_lstm.set_index('ds', inplace=True)

    scaler = MinMaxScaler()
    scaled_y = scaler.fit_transform(daily_pulls_lstm[['y']])

    def create_sequences(data, seq_len):
        X, y = [], []
        for i in range(len(data) - seq_len):
            X.append(data[i:i + seq_len])
            y.append(data[i + seq_len])
        return np.array(X), np.array(y)

    SEQ_LEN = 30
    X, y = create_sequences(scaled_y, SEQ_LEN)

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(SEQ_LEN, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=20, verbose=1)

    forecast_input = scaled_y[-SEQ_LEN:]
    forecast = []

    for _ in range(365):
        pred = model.predict(forecast_input.reshape(1, SEQ_LEN, 1), verbose=0)
        forecast.append(pred[0][0])
        forecast_input = np.append(forecast_input[1:], pred, axis=0)

    forecast_lstm = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
    forecast_dates = pd.date_range(start=daily_pulls_lstm.index[-1] + pd.Timedelta(days=1), periods=365)

    MODEL_IMAGE_NAME = "model_lstm_" + type +"_"+ repo_name + ".png"
    FULL_PATH_LSTM_pullsUTIONS = BASE_IMAGE_PATH + MODEL_IMAGE_NAME

    plt.figure(figsize=(12,6))
    plt.plot(daily_pulls_lstm.index, daily_pulls_lstm['y'], label="Actual")
    plt.plot(forecast_dates, forecast_lstm, label="LSTM Pulls Forecast")
    plt.legend()
    plt.title("LSTM Pulls Forecast")
    plt.grid(True)
    plt.savefig(LOCAL_IMAGE_PATH + MODEL_IMAGE_NAME)

    bucket = client.get_bucket(BUCKET_NAME)

    new_blob = bucket.blob(MODEL_IMAGE_NAME)
    new_blob.upload_from_filename(filename=LOCAL_IMAGE_PATH + MODEL_IMAGE_NAME)

    ## ---------------------------------------------------------------------------------------------------- ##
    #### FOR STATSMODEL

    sarima_series = daily_commits.set_index('ds')['y'].asfreq('D').fillna(0)

    sarima_model = sm.tsa.statespace.SARIMAX(sarima_series, order=(1,1,1), seasonal_order=(1,1,1,7), enforce_stationarity=False, enforce_invertibility=False)
    sarima_result = sarima_model.fit()

    sarima_forecast = sarima_result.get_forecast(steps=365)
    sarima_pred = sarima_forecast.predicted_mean
    sarima_ci = sarima_forecast.conf_int()

    STATS_IMAGE_NAME = "model_stats_" + type +"_"+ repo_name + ".png"
    FULL_PATH_STATS_MODELS = BASE_IMAGE_PATH + STATS_IMAGE_NAME

    plt.figure(figsize=(12,6))
    plt.plot(sarima_series, label='Observed')
    plt.plot(sarima_pred.index, sarima_pred, label='SARIMA pulls Forecast', color='r')
    plt.fill_between(sarima_ci.index, sarima_ci.iloc[:, 0], sarima_ci.iloc[:, 1], color='pink', alpha=0.3)
    plt.legend()
    plt.title("SARIMA Pulls Forecast")
    plt.grid(True)
    plt.savefig(LOCAL_IMAGE_PATH + STATS_IMAGE_NAME )

    bucket = client.get_bucket(BUCKET_NAME)
    new_blob = bucket.blob(STATS_IMAGE_NAME)
    new_blob.upload_from_filename(filename=LOCAL_IMAGE_PATH + STATS_IMAGE_NAME)

    json_response = {
        "fbprophet_forecast_pulls_url": FORECAST_IMAGE_URL,
        "fbprophet_forecast_pulls_components_url": FORECAST_COMPONENTS_IMAGE_URL,
        "forecast_lstm_pulls_url": FULL_PATH_LSTM_pullsUTIONS,
        "forecast_stats_pulls_url": FULL_PATH_STATS_MODELS
    }
    print("Done with pulls data")
    return jsonify(json_response)  


@app.route('/api/allCommitsModel', methods=['POST'])
def allCommitsModel():
    print("Started Commits Code:")
    body = request.get_json()
    commits_response = body["commits"]
    repo_name = body["repo"]
    type = body["type"]
    print("type:", type) 

    BASE_IMAGE_PATH = os.environ.get('BASE_IMAGE_PATH', 'https://storage.googleapis.com/lstm-storage/')
    BUCKET_NAME = os.environ.get('BUCKET_NAME', 'lstm-storage')

    LOCAL_IMAGE_PATH = "static/images/"
    FORECAST_IMAGE_NAME = "fbprophet_forecast_commits_" + type +"_"+ repo_name + ".png"
    FORECAST_IMAGE_URL = BASE_IMAGE_PATH + FORECAST_IMAGE_NAME

    FORECAST_COMPONENTS_IMAGE_NAME = "fbprophet_forecast_commits_components_" + type +"_" + repo_name + ".png"
    FORECAST_COMPONENTS_IMAGE_URL = BASE_IMAGE_PATH + FORECAST_COMPONENTS_IMAGE_NAME
    

    commitsDF = pd.DataFrame(commits_response)
    commitsDF['committed_date'] = pd.to_datetime(commitsDF['committed_date'])

    daily_commits = commitsDF.groupby(commitsDF['committed_date'].dt.date).size().reset_index(name='y')
    daily_commits.rename(columns={'committed_date': 'ds'}, inplace=True)
    print(daily_commits.head())

    model = Prophet(daily_seasonality=True)
    model.fit(daily_commits)

    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)

    forcast_fig1 = model.plot(forecast)
    forcast_fig2 = model.plot_components(forecast)
    forcast_fig1.savefig(LOCAL_IMAGE_PATH + FORECAST_IMAGE_NAME)
    forcast_fig2.savefig(LOCAL_IMAGE_PATH + FORECAST_COMPONENTS_IMAGE_NAME)

    bucket = client.get_bucket(BUCKET_NAME)

    new_blob = bucket.blob(FORECAST_IMAGE_NAME)
    new_blob.upload_from_filename(filename=LOCAL_IMAGE_PATH + FORECAST_IMAGE_NAME)

    new_blob = bucket.blob(FORECAST_COMPONENTS_IMAGE_NAME)
    new_blob.upload_from_filename(filename=LOCAL_IMAGE_PATH + FORECAST_COMPONENTS_IMAGE_NAME) 

    ## ---------------------------------------------------------------------------------------------------- ##
    #### FOR LSTM

    daily_pulls_lstm = daily_commits.copy()
    daily_pulls_lstm['ds'] = pd.to_datetime(daily_pulls_lstm['ds'])
    daily_pulls_lstm.set_index('ds', inplace=True)

    scaler = MinMaxScaler()
    scaled_y = scaler.fit_transform(daily_pulls_lstm[['y']])

    def create_sequences(data, seq_len):
        X, y = [], []
        for i in range(len(data) - seq_len):
            X.append(data[i:i + seq_len])
            y.append(data[i + seq_len])
        return np.array(X), np.array(y)

    SEQ_LEN = 30
    X, y = create_sequences(scaled_y, SEQ_LEN)

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(SEQ_LEN, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=20, verbose=1)

    forecast_input = scaled_y[-SEQ_LEN:]
    forecast = []

    for _ in range(365):
        pred = model.predict(forecast_input.reshape(1, SEQ_LEN, 1), verbose=0)
        forecast.append(pred[0][0])
        forecast_input = np.append(forecast_input[1:], pred, axis=0)

    forecast_lstm = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
    forecast_dates = pd.date_range(start=daily_pulls_lstm.index[-1] + pd.Timedelta(days=1), periods=365)

    MODEL_IMAGE_NAME = "model_lstm_" + type +"_"+ repo_name + ".png"
    FULL_PATH_LSTM_pullsUTIONS = BASE_IMAGE_PATH + MODEL_IMAGE_NAME

    plt.figure(figsize=(12,6))
    plt.plot(daily_pulls_lstm.index, daily_pulls_lstm['y'], label="Actual")
    plt.plot(forecast_dates, forecast_lstm, label="LSTM Forecast")
    plt.legend()
    plt.title("LSTM Commit Forecast")
    plt.grid(True)
    plt.savefig(LOCAL_IMAGE_PATH + MODEL_IMAGE_NAME)

    bucket = client.get_bucket(BUCKET_NAME)

    new_blob = bucket.blob(MODEL_IMAGE_NAME)
    new_blob.upload_from_filename(filename=LOCAL_IMAGE_PATH + MODEL_IMAGE_NAME)

    ## ---------------------------------------------------------------------------------------------------- ##
    #### FOR STATSMODEL

    sarima_series = daily_commits.set_index('ds')['y'].asfreq('D').fillna(0)

    sarima_model = sm.tsa.statespace.SARIMAX(sarima_series, order=(1,1,1), seasonal_order=(1,1,1,7), enforce_stationarity=False, enforce_invertibility=False)
    sarima_result = sarima_model.fit()

    sarima_forecast = sarima_result.get_forecast(steps=365)
    sarima_pred = sarima_forecast.predicted_mean
    sarima_ci = sarima_forecast.conf_int()

    STATS_IMAGE_NAME = "model_stats_" + type +"_"+ repo_name + ".png"
    FULL_PATH_STATS_MODELS = BASE_IMAGE_PATH + STATS_IMAGE_NAME

    plt.figure(figsize=(12,6))
    plt.plot(sarima_series, label='Observed')
    plt.plot(sarima_pred.index, sarima_pred, label='SARIMA commits Forecast', color='r')
    plt.fill_between(sarima_ci.index, sarima_ci.iloc[:, 0], sarima_ci.iloc[:, 1], color='pink', alpha=0.3)
    plt.legend()
    plt.title("SARIMA commits Forecast")
    plt.grid(True)
    plt.savefig(LOCAL_IMAGE_PATH + STATS_IMAGE_NAME )

    bucket = client.get_bucket(BUCKET_NAME)
    new_blob = bucket.blob(STATS_IMAGE_NAME)
    new_blob.upload_from_filename(filename=LOCAL_IMAGE_PATH + STATS_IMAGE_NAME)

    json_response = {
        "fbprophet_forecast_commits_url": FORECAST_IMAGE_URL,
        "fbprophet_forecast_commits_components_url": FORECAST_COMPONENTS_IMAGE_URL,
        "forecast_lstm_commits_url": FULL_PATH_LSTM_pullsUTIONS,
        "forecast_stats_commits_url": FULL_PATH_STATS_MODELS
    }
    print("Done with commits data")
    return jsonify(json_response)  


@app.route('/api/statmodelCommits', methods=['POST'])
def statmodelCommits():
    request_data = request.get_json()
    commit_data = request_data["pull"]
    repository = request_data["repo"]
    analysis_type = request_data["type"]
    print("Analysis type:", analysis_type)

    BASE_URL_CLOUD = os.environ.get('BASE_IMAGE_PATH', 'https://storage.googleapis.com/lstm-storage/')
    BUCKET_NAME = os.environ.get('BUCKET_NAME', 'lstm-storage')

    IMAGE_DIR = "static/images/"

    obs_img_filename = f"stats_observation_{analysis_type}_{repository}.png"
    obs_img_url = BASE_URL_CLOUD + obs_img_filename

    forecast_img_filename = f"stats_forecast_{analysis_type}_{repository}.png"
    forecast_img_url = BASE_URL_CLOUD + forecast_img_filename

    commit_df = pd.DataFrame(commit_data)
    grouped_df = commit_df.groupby(['created_at'], as_index=False).count()
    timeseries_df = grouped_df[['created_at', 'commit_number']]
    timeseries_df.columns = ['ds', 'y']
    print(timeseries_df)

    timeseries_df.set_index('ds', inplace=True)
    seasonal_period = max(2, len(timeseries_df) // 2) 
    
    decomposition = sm.tsa.seasonal_decompose(timeseries_df['y'], period=seasonal_period)
    obs_fig = decomposition.plot()
    obs_fig.set_size_inches(12, 7)
    plt.title("Commit Observation Trend")
    obs_fig.get_figure().savefig(os.path.join(IMAGE_DIR, obs_img_filename))

    arima_model = sm.tsa.ARIMA(timeseries_df['y'], order=(1, 0, 0))
    fitted_model = arima_model.fit()
    timeseries_df['forecast'] = fitted_model.fittedvalues
    forecast_plot = timeseries_df[['y', 'forecast']].plot(figsize=(12, 7))
    plt.title("Forecasted Commit Activity")
    forecast_plot.get_figure().savefig(os.path.join(IMAGE_DIR, forecast_img_filename))

    gcs_bucket = client.get_bucket(BUCKET_NAME)
    gcs_bucket.blob(obs_img_filename).upload_from_filename(os.path.join(IMAGE_DIR, obs_img_filename))
    gcs_bucket.blob(forecast_img_filename).upload_from_filename(os.path.join(IMAGE_DIR, forecast_img_filename))

    return jsonify({
        "stats_observation_url": obs_img_url,
        "stats_forecast_url": forecast_img_url
    })


@app.route('/api/statmodelClosed', methods=['POST'])
def statmodelClosed():
    payload = request.get_json()
    issue_type = payload["type"]
    repository = payload["repo"]
    issue_records = payload["issues"]

    base_image_url = os.environ.get('BASE_IMAGE_PATH', 'https://storage.googleapis.com/lstm-storage/')
    bucket_name = os.environ.get('BUCKET_NAME', 'lstm-storage')
    
    local_image_dir = "static/images/"

    observation_img_name = f"stats_observation_{issue_type}_{repository}.png"
    forecast_img_name = f"stats_forecast_{issue_type}_{repository}.png"
    observation_img_url = base_image_url + observation_img_name
    forecast_img_url = base_image_url + forecast_img_name

    df = pd.DataFrame(issue_records)
    df_grouped = df.groupby(['closed_at'], as_index=False).count()
    time_series_df = df_grouped[['closed_at', 'issue_number']]
    time_series_df.columns = ['ds', 'y']

    decompose_period = max(2, len(time_series_df) // 2)  
    
    decomposition = sm.tsa.seasonal_decompose(time_series_df.set_index('ds'), period=decompose_period)
    obs_fig = decomposition.plot()
    obs_fig.set_size_inches(12, 7)
    plt.title("Closed Issues - Observation Decomposition")
    obs_fig.get_figure().savefig(os.path.join(local_image_dir, observation_img_name))

    arima_model = sm.tsa.ARIMA(time_series_df['y'].iloc[1:], order=(1, 0, 0))
    forecast_results = arima_model.fit()
    time_series_df['forecast'] = forecast_results.fittedvalues

    forecast_fig = time_series_df[['y', 'forecast']].plot(figsize=(12, 7))
    plt.title("Forecast for Closed Issues")
    forecast_fig.get_figure().savefig(os.path.join(local_image_dir, forecast_img_name))

    cloud_bucket = client.get_bucket(bucket_name)
    cloud_bucket.blob(observation_img_name).upload_from_filename(
        filename=os.path.join(local_image_dir, observation_img_name)
    )
    cloud_bucket.blob(forecast_img_name).upload_from_filename(
        filename=os.path.join(local_image_dir, forecast_img_name)
    )

    return jsonify({
        "stats_observation_url": observation_img_url,
        "stats_forecast_url": forecast_img_url
    })


@app.route('/api/statmodelCreated', methods=['POST'])
def statmodelCreated():
    try:
        payload = request.get_json()
        analysis_type = payload["type"]
        repository = payload["repo"]
        issue_data = payload["issues"]

        base_image_url = os.environ.get('BASE_IMAGE_PATH', 'https://storage.googleapis.com/lstm-storage/')
        bucket_name = os.environ.get('BUCKET_NAME', 'lstm-storage')
        local_directory = "static/images/"

        obs_filename = f"stats_observation_{analysis_type}_{repository}.png"
        forecast_filename = f"stats_forecast_{analysis_type}_{repository}.png"
        
        obs_url = f"{base_image_url}{obs_filename}"
        forecast_url = f"{base_image_url}{forecast_filename}"

        df = pd.DataFrame(issue_data)
        df_grouped = df.groupby(['created_at'], as_index=False).count()
        ts_data = df_grouped[['created_at', 'issue_number']]
        ts_data.columns = ['ds', 'y']

        period_length = max(2, len(ts_data) // 2)  
        
        decomposition = sm.tsa.seasonal_decompose(ts_data.set_index('ds')['y'], period=period_length)
        obs_plot = decomposition.plot()
        obs_plot.set_size_inches(12, 7)
        plt.title("Issue Creation Observation")
        obs_plot.get_figure().savefig(os.path.join(local_directory, obs_filename))

        arima_model = sm.tsa.ARIMA(ts_data['y'].iloc[1:], order=(1, 0, 0))
        arima_results = arima_model.fit()
        ts_data['forecast'] = arima_results.fittedvalues

        forecast_plot = ts_data[['y', 'forecast']].plot(figsize=(12, 7))
        plt.title("Forecast of Created Issues")
        forecast_plot.get_figure().savefig(os.path.join(local_directory, forecast_filename))

        bucket = client.get_bucket(bucket_name)
        for filename in [obs_filename, forecast_filename]:
            blob = bucket.blob(filename)
            blob.upload_from_filename(os.path.join(local_directory, filename))

        return jsonify({
            "stats_observation_url": obs_url,
            "stats_forecast_url": forecast_url
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/allBranchesModel', methods=['POST'])
def allBranchesModel():
    body = request.get_json()
    branch_response = body["branch"]
    repo_name = body["repo"]
    type = body["type"]
    print("type:", type)

    BASE_IMAGE_PATH = os.environ.get('BASE_IMAGE_PATH', 'https://storage.googleapis.com/lstm-storage/')
    BUCKET_NAME = os.environ.get('BUCKET_NAME', 'lstm-storage')

    LOCAL_IMAGE_PATH = "static/images/"
    FORECAST_IMAGE_NAME = "fbprophet_forecast_branches_" + type +"_"+ repo_name + ".png"
    FORECAST_IMAGE_URL = BASE_IMAGE_PATH + FORECAST_IMAGE_NAME

    FORECAST_COMPONENTS_IMAGE_NAME = "fbprophet_forecast_branches_components_" + type +"_" + repo_name + ".png"
    FORECAST_COMPONENTS_IMAGE_URL = BASE_IMAGE_PATH + FORECAST_COMPONENTS_IMAGE_NAME

    branchDF = pd.DataFrame(branch_response)
    branchDF['commit_date'] = pd.to_datetime(branchDF['commit_date'])

    daily_commits = branchDF.groupby(branchDF['commit_date'].dt.date).size().reset_index(name='y')
    daily_commits.rename(columns={'commit_date': 'ds'}, inplace=True)
    print(daily_commits.head())


    #### FOR Prophet only
    model = Prophet(daily_seasonality=True)
    model.fit(daily_commits)

    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)

    forcast_fig1 = model.plot(forecast)
    forcast_fig2 = model.plot_components(forecast)
    forcast_fig1.savefig(LOCAL_IMAGE_PATH + FORECAST_IMAGE_NAME)
    forcast_fig2.savefig(LOCAL_IMAGE_PATH + FORECAST_COMPONENTS_IMAGE_NAME)

    bucket = client.get_bucket(BUCKET_NAME)

    new_blob = bucket.blob(FORECAST_IMAGE_NAME)
    new_blob.upload_from_filename(filename=LOCAL_IMAGE_PATH + FORECAST_IMAGE_NAME)

    new_blob = bucket.blob(FORECAST_COMPONENTS_IMAGE_NAME)
    new_blob.upload_from_filename(filename=LOCAL_IMAGE_PATH + FORECAST_COMPONENTS_IMAGE_NAME)

    ## ------------------------- ##
    
    #### FOR LSTM
    daily_commits_lstm = daily_commits.copy()
    daily_commits_lstm['ds'] = pd.to_datetime(daily_commits_lstm['ds'])
    daily_commits_lstm.set_index('ds', inplace=True)

    scaler = MinMaxScaler()
    scaled_y = scaler.fit_transform(daily_commits_lstm[['y']])

    # Create sequences
    def create_sequences(data, seq_len):
        X, y = [], []
        for i in range(len(data) - seq_len):
            X.append(data[i:i + seq_len])
            y.append(data[i + seq_len])
        return np.array(X), np.array(y)

    SEQ_LEN = 30
    X, y = create_sequences(scaled_y, SEQ_LEN)

    # Build LSTM Model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(SEQ_LEN, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=20, verbose=1)

    # Forecast next 365 days
    forecast_input = scaled_y[-SEQ_LEN:]
    forecast = []

    for _ in range(365):
        pred = model.predict(forecast_input.reshape(1, SEQ_LEN, 1), verbose=0)
        forecast.append(pred[0][0])
        forecast_input = np.append(forecast_input[1:], pred, axis=0)

    # Inverse transform forecast
    forecast_lstm = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
    forecast_dates = pd.date_range(start=daily_commits_lstm.index[-1] + pd.Timedelta(days=1), periods=365)
    
    MODEL_IMAGE_NAME = "model_lstm_" + type +"_"+ repo_name + ".png"
    FULL_PATH_LSTM_BRANCH = BASE_IMAGE_PATH + MODEL_IMAGE_NAME
    
    plt.figure(figsize=(12,6))
    plt.plot(daily_commits_lstm.index, daily_commits_lstm['y'], label="Actual")
    plt.plot(forecast_dates, forecast_lstm, label="LSTM Forecast")
    plt.legend()
    plt.title("LSTM Branches Forecast")
    plt.grid(True)
    plt.savefig(LOCAL_IMAGE_PATH + MODEL_IMAGE_NAME)

    bucket = client.get_bucket(BUCKET_NAME)

    new_blob = bucket.blob(MODEL_IMAGE_NAME)
    new_blob.upload_from_filename(filename=LOCAL_IMAGE_PATH + MODEL_IMAGE_NAME)

    ## ------------------------- ##
    
    #### FOR Statsmodel
    sarima_series = daily_commits.set_index('ds')['y'].asfreq('D').fillna(0)
    sarima_model = sm.tsa.statespace.SARIMAX(sarima_series, 
                                         order=(1,1,1), 
                                         seasonal_order=(1,1,1,7), 
                                         enforce_stationarity=False, 
                                         enforce_invertibility=False)
    sarima_result = sarima_model.fit()
    sarima_forecast = sarima_result.get_forecast(steps=365)
    sarima_pred = sarima_forecast.predicted_mean
    sarima_ci = sarima_forecast.conf_int()

    STATS_IMAGE_NAME = "model_stats_" + type +"_"+ repo_name + ".png"
    FULL_PATH_STATS_BRANCH = BASE_IMAGE_PATH + STATS_IMAGE_NAME

    plt.figure(figsize=(12,6))
    plt.plot(sarima_series, label='Observed')
    plt.plot(sarima_pred.index, sarima_pred, label='SARIMA Branch Forecast', color='r')
    plt.fill_between(sarima_ci.index, sarima_ci.iloc[:, 0], sarima_ci.iloc[:, 1], color='pink', alpha=0.3)
    plt.legend()
    plt.title("SARIMA Branch Forecast")
    plt.grid(True)
    plt.savefig(LOCAL_IMAGE_PATH + STATS_IMAGE_NAME )

    bucket = client.get_bucket(BUCKET_NAME)
    new_blob = bucket.blob(STATS_IMAGE_NAME)
    new_blob.upload_from_filename(filename=LOCAL_IMAGE_PATH + STATS_IMAGE_NAME)

    json_response = {
        "fbprophet_forecast_branch_url": FORECAST_IMAGE_URL,
        "fbprophet_forecast_branch_components_url": FORECAST_COMPONENTS_IMAGE_URL,
        "forecast_lstm_branch_url": FULL_PATH_LSTM_BRANCH,
        "forecast_stats_branch_url": FULL_PATH_STATS_BRANCH
    }
    print("Done with branch data")
    return jsonify(json_response)


@app.route('/api/allContributorsModel', methods=['POST'])
def allContributorsModel():
    print("Started Contributions Code:")
    body = request.get_json()
    contributors_response = body["contributor"]
    repo_name = body["repo"]
    type = body["type"]
    print("type:", type)

    BASE_IMAGE_PATH = os.environ.get('BASE_IMAGE_PATH', 'https://storage.googleapis.com/lstm-storage/')
    BUCKET_NAME = os.environ.get('BUCKET_NAME', 'lstm-storage')

    LOCAL_IMAGE_PATH = "static/images/"
    FORECAST_IMAGE_NAME = "fbprophet_forecast_contributors_" + type +"_"+ repo_name + ".png"
    FORECAST_IMAGE_URL = BASE_IMAGE_PATH + FORECAST_IMAGE_NAME

    FORECAST_COMPONENTS_IMAGE_NAME = "fbprophet_forecast_contributors_components_" + type +"_" + repo_name + ".png"
    FORECAST_COMPONENTS_IMAGE_URL = BASE_IMAGE_PATH + FORECAST_COMPONENTS_IMAGE_NAME

    ###### PROPHET MODEL
    contributorsDF = pd.DataFrame(contributors_response)

    contributorsDF['event_date'] = pd.to_datetime(contributorsDF['event_date'])
    
    print(contributorsDF.columns)

    daily_commits = contributorsDF.groupby(contributorsDF['event_date'].dt.date).size().reset_index(name='y')
    daily_commits.rename(columns={'event_date': 'ds'}, inplace=True)
    print(daily_commits.head())

    model = Prophet(daily_seasonality=True)
    model.fit(daily_commits)

    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)

    forcast_fig1 = model.plot(forecast)
    forcast_fig2 = model.plot_components(forecast)
    forcast_fig1.savefig(LOCAL_IMAGE_PATH + FORECAST_IMAGE_NAME)
    forcast_fig2.savefig(LOCAL_IMAGE_PATH + FORECAST_COMPONENTS_IMAGE_NAME)

    bucket = client.get_bucket(BUCKET_NAME)

    new_blob = bucket.blob(FORECAST_IMAGE_NAME)
    new_blob.upload_from_filename(filename=LOCAL_IMAGE_PATH + FORECAST_IMAGE_NAME)

    new_blob = bucket.blob(FORECAST_COMPONENTS_IMAGE_NAME)
    new_blob.upload_from_filename(filename=LOCAL_IMAGE_PATH + FORECAST_COMPONENTS_IMAGE_NAME)

    ## ---------------------------------------------------------------------------------------------------- ##
    #### FOR LSTM
    daily_commits_lstm = daily_commits.copy()
    daily_commits_lstm['ds'] = pd.to_datetime(daily_commits_lstm['ds'])
    daily_commits_lstm.set_index('ds', inplace=True)

    scaler = MinMaxScaler()
    scaled_y = scaler.fit_transform(daily_commits_lstm[['y']])

    # Create sequences
    def create_sequences(data, seq_len):
        X, y = [], []
        for i in range(len(data) - seq_len):
            X.append(data[i:i + seq_len])
            y.append(data[i + seq_len])
        return np.array(X), np.array(y)

    SEQ_LEN = 10
    X, y = create_sequences(scaled_y, SEQ_LEN)

    # Build LSTM Model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(SEQ_LEN, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=20, verbose=1)

    # Forecast next 365 days
    forecast_input = scaled_y[-SEQ_LEN:]
    forecast = []

    for _ in range(365):
        pred = model.predict(forecast_input.reshape(1, SEQ_LEN, 1), verbose=0)
        forecast.append(pred[0][0])
        forecast_input = np.append(forecast_input[1:], pred, axis=0)

    # Inverse transform forecast
    forecast_lstm = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
    forecast_dates = pd.date_range(start=daily_commits_lstm.index[-1] + pd.Timedelta(days=1), periods=365)
    
    MODEL_IMAGE_NAME = "model_lstm_" + type +"_"+ repo_name + ".png"
    FULL_PATH_LSTM_CONTRIBUTIONS = BASE_IMAGE_PATH + MODEL_IMAGE_NAME
    
    plt.figure(figsize=(12,6))
    plt.plot(daily_commits_lstm.index, daily_commits_lstm['y'], label="Actual")
    plt.plot(forecast_dates, forecast_lstm, label="LSTM Contributors Forecast")
    plt.legend()
    plt.title("LSTM Contributors Forecast")
    plt.grid(True)
    plt.savefig(LOCAL_IMAGE_PATH + MODEL_IMAGE_NAME)

    bucket = client.get_bucket(BUCKET_NAME)

    new_blob = bucket.blob(MODEL_IMAGE_NAME)
    new_blob.upload_from_filename(filename=LOCAL_IMAGE_PATH + MODEL_IMAGE_NAME)

    ## ---------------------------------------------------------------------------------------------------- ##
    #### FOR STATSMODEL

    sarima_series = daily_commits.set_index('ds')['y'].asfreq('D').fillna(0)
    sarima_model = sm.tsa.statespace.SARIMAX(sarima_series, order=(1,1,1), seasonal_order=(1,1,1,7), enforce_stationarity=False, enforce_invertibility=False)
    sarima_result = sarima_model.fit()
    sarima_forecast = sarima_result.get_forecast(steps=365)
    sarima_pred = sarima_forecast.predicted_mean
    sarima_ci = sarima_forecast.conf_int()

    STATS_IMAGE_NAME = "model_stats_" + type +"_"+ repo_name + ".png"
    FULL_PATH_STATS_CONTRIBUTIONS = BASE_IMAGE_PATH + STATS_IMAGE_NAME

    plt.figure(figsize=(12,6))
    plt.plot(sarima_series, label='Observed')
    plt.plot(sarima_pred.index, sarima_pred, label='SARIMA Contributors Forecast', color='r')
    plt.fill_between(sarima_ci.index, sarima_ci.iloc[:, 0], sarima_ci.iloc[:, 1], color='pink', alpha=0.3)
    plt.legend()
    plt.title("SARIMA Contributors Forecast")
    plt.grid(True)
    plt.savefig(LOCAL_IMAGE_PATH + STATS_IMAGE_NAME )

    bucket = client.get_bucket(BUCKET_NAME)
    new_blob = bucket.blob(STATS_IMAGE_NAME)
    new_blob.upload_from_filename(filename=LOCAL_IMAGE_PATH + STATS_IMAGE_NAME)

    json_response = {
        "fbprophet_forecast_contributors_url": FORECAST_IMAGE_URL,
        "fbprophet_forecast_contributors_components_url": FORECAST_COMPONENTS_IMAGE_URL,
        "forecast_lstm_contributors_url": FULL_PATH_LSTM_CONTRIBUTIONS,
        "forecast_stats_contributors_url": FULL_PATH_STATS_CONTRIBUTIONS
    }
    print("Done with contributors data")
    return jsonify(json_response)  


@app.route('/api/allReleasesModel', methods=['POST'])
def allReleasesModel():
    print("Started Releases Code:")
    body = request.get_json()
    release_response = body["release"]
    repo_name = body["repo"]
    type = body["type"]
    print("type:", type) 

    BASE_IMAGE_PATH = os.environ.get('BASE_IMAGE_PATH', 'https://storage.googleapis.com/lstm-storage/')
    BUCKET_NAME = os.environ.get('BUCKET_NAME', 'lstm-storage')

    LOCAL_IMAGE_PATH = "static/images/"
    FORECAST_IMAGE_NAME = "fbprophet_forecast_releases_" + type +"_"+ repo_name + ".png"
    FORECAST_IMAGE_URL = BASE_IMAGE_PATH + FORECAST_IMAGE_NAME

    FORECAST_COMPONENTS_IMAGE_NAME = "fbprophet_forecast_releases_components_" + type +"_" + repo_name + ".png"
    FORECAST_COMPONENTS_IMAGE_URL = BASE_IMAGE_PATH + FORECAST_COMPONENTS_IMAGE_NAME
    
    releasesDF = pd.DataFrame(release_response)
    releasesDF['published_at'] = pd.to_datetime(releasesDF['published_at'])

    daily_commits = releasesDF.groupby(releasesDF['published_at'].dt.date).size().reset_index(name='y')
    daily_commits.rename(columns={'published_at': 'ds'}, inplace=True)
    print(daily_commits.head())

    model = Prophet(daily_seasonality=True)
    model.fit(daily_commits)

    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)

    forcast_fig1 = model.plot(forecast)
    forcast_fig2 = model.plot_components(forecast)
    forcast_fig1.savefig(LOCAL_IMAGE_PATH + FORECAST_IMAGE_NAME)
    forcast_fig2.savefig(LOCAL_IMAGE_PATH + FORECAST_COMPONENTS_IMAGE_NAME)

    bucket = client.get_bucket(BUCKET_NAME)

    new_blob = bucket.blob(FORECAST_IMAGE_NAME)
    new_blob.upload_from_filename(filename=LOCAL_IMAGE_PATH + FORECAST_IMAGE_NAME)

    new_blob = bucket.blob(FORECAST_COMPONENTS_IMAGE_NAME)
    new_blob.upload_from_filename(filename=LOCAL_IMAGE_PATH + FORECAST_COMPONENTS_IMAGE_NAME)    

    ## ---------------------------------------------------------------------------------------------------- ##
    #### FOR LSTM

    daily_contrib_lstm = daily_commits.copy()
    daily_contrib_lstm['ds'] = pd.to_datetime(daily_contrib_lstm['ds'])
    daily_contrib_lstm.set_index('ds', inplace=True)

    scaler = MinMaxScaler()
    scaled_y = scaler.fit_transform(daily_contrib_lstm[['y']])

    def create_sequences(data, seq_len):
        X, y = [], []
        for i in range(len(data) - seq_len):
            X.append(data[i:i + seq_len])
            y.append(data[i + seq_len])
        return np.array(X), np.array(y)

    SEQ_LEN = 30
    X, y = create_sequences(scaled_y, SEQ_LEN)

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(SEQ_LEN, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=20, verbose=1)

    forecast_input = scaled_y[-SEQ_LEN:]
    forecast = []

    for _ in range(365):
        pred = model.predict(forecast_input.reshape(1, SEQ_LEN, 1), verbose=0)
        forecast.append(pred[0][0])
        forecast_input = np.append(forecast_input[1:], pred, axis=0)

    forecast_lstm = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
    forecast_dates = pd.date_range(start=daily_contrib_lstm.index[-1] + pd.Timedelta(days=1), periods=365)

    MODEL_IMAGE_NAME = "model_lstm_" + type +"_"+ repo_name + ".png"
    FULL_PATH_LSTM_RELEASES = BASE_IMAGE_PATH + MODEL_IMAGE_NAME

    plt.figure(figsize=(12,6))
    plt.plot(daily_contrib_lstm.index, daily_contrib_lstm['y'], label="Actual")
    plt.plot(forecast_dates, forecast_lstm, label="LSTM Releases Forecast")
    plt.legend()
    plt.title("LSTM Releases Forecast")
    plt.grid(True)
    plt.savefig(LOCAL_IMAGE_PATH + MODEL_IMAGE_NAME)

    bucket = client.get_bucket(BUCKET_NAME)

    new_blob = bucket.blob(MODEL_IMAGE_NAME)
    new_blob.upload_from_filename(filename=LOCAL_IMAGE_PATH + MODEL_IMAGE_NAME)

    ## ---------------------------------------------------------------------------------------------------- ##
    #### FOR STATSMODEL

    sarima_series = daily_commits.set_index('ds')['y'].asfreq('D').fillna(0)

    sarima_model = sm.tsa.statespace.SARIMAX(sarima_series, order=(1,1,1), seasonal_order=(1,1,1,7), enforce_stationarity=False, enforce_invertibility=False)
    sarima_result = sarima_model.fit()

    sarima_forecast = sarima_result.get_forecast(steps=365)
    sarima_pred = sarima_forecast.predicted_mean
    sarima_ci = sarima_forecast.conf_int()

    STATS_IMAGE_NAME = "model_stats_" + type +"_"+ repo_name + ".png"
    FULL_PATH_STATS_CONTRIBUTIONS = BASE_IMAGE_PATH + STATS_IMAGE_NAME

    plt.figure(figsize=(12,6))
    plt.plot(sarima_series, label='Observed')
    plt.plot(sarima_pred.index, sarima_pred, label='SARIMA Releases Forecast', color='r')
    plt.fill_between(sarima_ci.index, sarima_ci.iloc[:, 0], sarima_ci.iloc[:, 1], color='pink', alpha=0.3)
    plt.legend()
    plt.title("SARIMA Releases Forecast")
    plt.grid(True)
    plt.savefig(LOCAL_IMAGE_PATH + STATS_IMAGE_NAME )

    bucket = client.get_bucket(BUCKET_NAME)
    new_blob = bucket.blob(STATS_IMAGE_NAME)
    new_blob.upload_from_filename(filename=LOCAL_IMAGE_PATH + STATS_IMAGE_NAME)

    json_response = {
        "fbprophet_forecast_releases_url": FORECAST_IMAGE_URL,
        "fbprophet_forecast_releases_components_url": FORECAST_COMPONENTS_IMAGE_URL,
        "forecast_lstm_releases_url": FULL_PATH_LSTM_RELEASES,
        "forecast_stats_releases_url": FULL_PATH_STATS_CONTRIBUTIONS
    }
    print("Done with releases data")
    return jsonify(json_response)  


@app.route('/api/allIssuesCreatedAt', methods=['POST'])
def allIssuesCreatedAt():
    print("Started Issues CreatedAT Code:")
    body = request.get_json()
    issues_response = body["issues"]
    repo_name = body["repo"]
    type = body["type"]
    print("type:", type) 

    BASE_IMAGE_PATH = os.environ.get('BASE_IMAGE_PATH', 'https://storage.googleapis.com/lstm-storage/')
    BUCKET_NAME = os.environ.get('BUCKET_NAME', 'lstm-storage')

    LOCAL_IMAGE_PATH = "static/images/"
    FORECAST_IMAGE_NAME = "fbprophet_forecast_issues_created_at_" + type +"_"+ repo_name + ".png"
    FORECAST_IMAGE_URL = BASE_IMAGE_PATH + FORECAST_IMAGE_NAME

    FORECAST_COMPONENTS_IMAGE_NAME = "fbprophet_forecast_issues_created_at_components_" + type +"_" + repo_name + ".png"
    FORECAST_COMPONENTS_IMAGE_URL = BASE_IMAGE_PATH + FORECAST_COMPONENTS_IMAGE_NAME
        
    issuesDF = pd.DataFrame(issues_response)
    issuesDF['created_at'] = pd.to_datetime(issuesDF['created_at'])

    daily_issues = issuesDF.groupby(issuesDF['created_at'].dt.date).size().reset_index(name='y')
    daily_issues.rename(columns={'created_at': 'ds'}, inplace=True)
    print(daily_issues.head())

    model = Prophet(daily_seasonality=True)
    model.fit(daily_issues)

    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)

    forcast_fig1 = model.plot(forecast)
    forcast_fig2 = model.plot_components(forecast)
    forcast_fig1.savefig(LOCAL_IMAGE_PATH + FORECAST_IMAGE_NAME)
    forcast_fig2.savefig(LOCAL_IMAGE_PATH + FORECAST_COMPONENTS_IMAGE_NAME)

    bucket = client.get_bucket(BUCKET_NAME)

    new_blob = bucket.blob(FORECAST_IMAGE_NAME)
    new_blob.upload_from_filename(filename=LOCAL_IMAGE_PATH + FORECAST_IMAGE_NAME)

    new_blob = bucket.blob(FORECAST_COMPONENTS_IMAGE_NAME)
    new_blob.upload_from_filename(filename=LOCAL_IMAGE_PATH + FORECAST_COMPONENTS_IMAGE_NAME)  

    ## ---------------------------------------------------------------------------------------------------- ##
    #### FOR LSTM

    daily_issue_created_at_lstm = daily_issues.copy()
    daily_issue_created_at_lstm['ds'] = pd.to_datetime(daily_issue_created_at_lstm['ds'])
    daily_issue_created_at_lstm.set_index('ds', inplace=True)

    scaler = MinMaxScaler()
    scaled_y = scaler.fit_transform(daily_issue_created_at_lstm[['y']])

    def create_sequences(data, seq_len):
        X, y = [], []
        for i in range(len(data) - seq_len):
            X.append(data[i:i + seq_len])
            y.append(data[i + seq_len])
        return np.array(X), np.array(y)

    SEQ_LEN = 30
    X, y = create_sequences(scaled_y, SEQ_LEN)

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(SEQ_LEN, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=20, verbose=1)

    forecast_input = scaled_y[-SEQ_LEN:]
    forecast = []

    for _ in range(365):
        pred = model.predict(forecast_input.reshape(1, SEQ_LEN, 1), verbose=0)
        forecast.append(pred[0][0])
        forecast_input = np.append(forecast_input[1:], pred, axis=0)

    forecast_lstm = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
    forecast_dates = pd.date_range(start=daily_issue_created_at_lstm.index[-1] + pd.Timedelta(days=1), periods=365)

    MODEL_IMAGE_NAME = "model_lstm_" + type +"_"+ repo_name + ".png"
    FULL_PATH_LSTM_ISSUES_CREATED_AT = BASE_IMAGE_PATH + MODEL_IMAGE_NAME

    plt.figure(figsize=(12,6))
    plt.plot(daily_issue_created_at_lstm.index, daily_issue_created_at_lstm['y'], label="Actual")
    plt.plot(forecast_dates, forecast_lstm, label="LSTM created at Forecast")
    plt.legend()
    plt.title("LSTM created at Forecast")
    plt.grid(True)
    plt.savefig(LOCAL_IMAGE_PATH + MODEL_IMAGE_NAME)

    bucket = client.get_bucket(BUCKET_NAME)

    new_blob = bucket.blob(MODEL_IMAGE_NAME)
    new_blob.upload_from_filename(filename=LOCAL_IMAGE_PATH + MODEL_IMAGE_NAME)

    ## ---------------------------------------------------------------------------------------------------- ##
    #### FOR STATSMODEL

    sarima_series = daily_issues.set_index('ds')['y'].asfreq('D').fillna(0)

    sarima_model = sm.tsa.statespace.SARIMAX(sarima_series, order=(1,1,1), seasonal_order=(1,1,1,7), enforce_stationarity=False, enforce_invertibility=False)
    sarima_result = sarima_model.fit()

    sarima_forecast = sarima_result.get_forecast(steps=365)
    sarima_pred = sarima_forecast.predicted_mean
    sarima_ci = sarima_forecast.conf_int()

    STATS_IMAGE_NAME = "model_stats_" + type +"_"+ repo_name + ".png"
    FULL_PATH_STATS_ISSUES_CREATED_AT = BASE_IMAGE_PATH + STATS_IMAGE_NAME

    plt.figure(figsize=(12,6))
    plt.plot(sarima_series, label='Observed')
    plt.plot(sarima_pred.index, sarima_pred, label='SARIMA created at Forecast', color='r')
    plt.fill_between(sarima_ci.index, sarima_ci.iloc[:, 0], sarima_ci.iloc[:, 1], color='pink', alpha=0.3)
    plt.legend()
    plt.title("SARIMA issues created_at Forecast")
    plt.grid(True)
    plt.savefig(LOCAL_IMAGE_PATH + STATS_IMAGE_NAME )

    bucket = client.get_bucket(BUCKET_NAME)
    new_blob = bucket.blob(STATS_IMAGE_NAME)
    new_blob.upload_from_filename(filename=LOCAL_IMAGE_PATH + STATS_IMAGE_NAME)

    issuesDF['day_of_week'] = issuesDF['created_at'].dt.day_name()
    day_counts = issuesDF['day_of_week'].value_counts()
    max_day = day_counts.idxmax()

    json_response = {
        "fbprophet_forecast_issues_created_at_url": FORECAST_IMAGE_URL,
        "fbprophet_forecast_issues_created_at_components_url": FORECAST_COMPONENTS_IMAGE_URL,
        "forecast_lstm_issues_created_at_url": FULL_PATH_LSTM_ISSUES_CREATED_AT,
        "forecast_stats_issues_created_at_url": FULL_PATH_STATS_ISSUES_CREATED_AT,
        "week_of_the_day": f"The Day of the week with maximum issues closed are: {max_day}"
    }
    print("Done with issues created_at data")
    return jsonify(json_response)  


@app.route('/api/allIssuesClosedAt', methods=['POST'])
def allIssuesClosedAt():
    print("Started Issues closed At Code:")
    body = request.get_json()
    issues_response = body["issues"]
    repo_name = body["repo"]
    type = body["type"]
    print("type:", type) 

    BASE_IMAGE_PATH = os.environ.get('BASE_IMAGE_PATH', 'https://storage.googleapis.com/lstm-storage/')
    BUCKET_NAME = os.environ.get('BUCKET_NAME', 'lstm-storage')

    LOCAL_IMAGE_PATH = "static/images/"
    FORECAST_IMAGE_NAME = "fbprophet_forecast_issues_closed_at_" + type +"_"+ repo_name + ".png"
    FORECAST_IMAGE_URL = BASE_IMAGE_PATH + FORECAST_IMAGE_NAME

    FORECAST_COMPONENTS_IMAGE_NAME = "fbprophet_forecast_issues_closed_at_components_" + type +"_" + repo_name + ".png"
    FORECAST_COMPONENTS_IMAGE_URL = BASE_IMAGE_PATH + FORECAST_COMPONENTS_IMAGE_NAME
        
    issuesDF = pd.DataFrame(issues_response)
    issuesDF['closed_at'] = pd.to_datetime(issuesDF['closed_at'])

    daily_issues = issuesDF.groupby(issuesDF['closed_at'].dt.date).size().reset_index(name='y')
    daily_issues.rename(columns={'closed_at': 'ds'}, inplace=True)
    print(daily_issues.head())

    model = Prophet(daily_seasonality=True)
    model.fit(daily_issues)

    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)

    forcast_fig1 = model.plot(forecast)
    forcast_fig2 = model.plot_components(forecast)
    forcast_fig1.savefig(LOCAL_IMAGE_PATH + FORECAST_IMAGE_NAME)
    forcast_fig2.savefig(LOCAL_IMAGE_PATH + FORECAST_COMPONENTS_IMAGE_NAME)

    bucket = client.get_bucket(BUCKET_NAME)

    new_blob = bucket.blob(FORECAST_IMAGE_NAME)
    new_blob.upload_from_filename(filename=LOCAL_IMAGE_PATH + FORECAST_IMAGE_NAME)

    new_blob = bucket.blob(FORECAST_COMPONENTS_IMAGE_NAME)
    new_blob.upload_from_filename(filename=LOCAL_IMAGE_PATH + FORECAST_COMPONENTS_IMAGE_NAME)  

    ## ---------------------------------------------------------------------------------------------------- ##
    #### FOR LSTM

    daily_issue_closed_at_lstm = daily_issues.copy()
    daily_issue_closed_at_lstm['ds'] = pd.to_datetime(daily_issue_closed_at_lstm['ds'])
    daily_issue_closed_at_lstm.set_index('ds', inplace=True)

    scaler = MinMaxScaler()
    scaled_y = scaler.fit_transform(daily_issue_closed_at_lstm[['y']])

    def create_sequences(data, seq_len):
        X, y = [], []
        for i in range(len(data) - seq_len):
            X.append(data[i:i + seq_len])
            y.append(data[i + seq_len])
        return np.array(X), np.array(y)

    SEQ_LEN = 30
    X, y = create_sequences(scaled_y, SEQ_LEN)

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(SEQ_LEN, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=20, verbose=1)

    forecast_input = scaled_y[-SEQ_LEN:]
    forecast = []

    for _ in range(365):
        pred = model.predict(forecast_input.reshape(1, SEQ_LEN, 1), verbose=0)
        forecast.append(pred[0][0])
        forecast_input = np.append(forecast_input[1:], pred, axis=0)

    forecast_lstm = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
    forecast_dates = pd.date_range(start=daily_issue_closed_at_lstm.index[-1] + pd.Timedelta(days=1), periods=365)

    MODEL_IMAGE_NAME = "model_lstm_" + type +"_"+ repo_name + ".png"
    FULL_PATH_LSTM_ISSUES_CLOSED_AT = BASE_IMAGE_PATH + MODEL_IMAGE_NAME

    plt.figure(figsize=(12,6))
    plt.plot(daily_issue_closed_at_lstm.index, daily_issue_closed_at_lstm['y'], label="Actual")
    plt.plot(forecast_dates, forecast_lstm, label="LSTM closed at Forecast")
    plt.legend()
    plt.title("LSTM closed at Forecast")
    plt.grid(True)
    plt.savefig(LOCAL_IMAGE_PATH + MODEL_IMAGE_NAME)

    bucket = client.get_bucket(BUCKET_NAME)

    new_blob = bucket.blob(MODEL_IMAGE_NAME)
    new_blob.upload_from_filename(filename=LOCAL_IMAGE_PATH + MODEL_IMAGE_NAME)

    ## ---------------------------------------------------------------------------------------------------- ##
    #### FOR STATSMODEL

    sarima_series = daily_issues.set_index('ds')['y'].asfreq('D').fillna(0)

    sarima_model = sm.tsa.statespace.SARIMAX(sarima_series, order=(1,1,1), seasonal_order=(1,1,1,7), enforce_stationarity=False, enforce_invertibility=False)
    sarima_result = sarima_model.fit()

    sarima_forecast = sarima_result.get_forecast(steps=365)
    sarima_pred = sarima_forecast.predicted_mean
    sarima_ci = sarima_forecast.conf_int()

    STATS_IMAGE_NAME = "model_stats_" + type +"_"+ repo_name + ".png"
    FULL_PATH_STATS_ISSUES_CLOSED_AT = BASE_IMAGE_PATH + STATS_IMAGE_NAME

    plt.figure(figsize=(12,6))
    plt.plot(sarima_series, label='Observed')
    plt.plot(sarima_pred.index, sarima_pred, label='SARIMA Forecast', color='r')
    plt.fill_between(sarima_ci.index, sarima_ci.iloc[:, 0], sarima_ci.iloc[:, 1], color='pink', alpha=0.3)
    plt.legend()
    plt.title("SARIMA issues closed_at Forecast")
    plt.grid(True)
    plt.savefig(LOCAL_IMAGE_PATH + STATS_IMAGE_NAME )

    bucket = client.get_bucket(BUCKET_NAME)
    new_blob = bucket.blob(STATS_IMAGE_NAME)
    new_blob.upload_from_filename(filename=LOCAL_IMAGE_PATH + STATS_IMAGE_NAME)

    issuesDF['day_of_week'] = issuesDF['closed_at'].dt.day_name()
    day_counts = issuesDF['day_of_week'].value_counts()
    max_day = day_counts.idxmax()

    issuesDF['closed_at'] = pd.to_datetime(issuesDF['closed_at'], errors='coerce')
    issuesDF['month'] = issuesDF['closed_at'].dt.month_name()

    closed_month_counts = issuesDF['month'].value_counts()
    max_closed_month = closed_month_counts.idxmax()

    json_response = {
        "fbprophet_forecast_issues_closed_at_url": FORECAST_IMAGE_URL,
        "fbprophet_forecast_issues_closed_at_components_url": FORECAST_COMPONENTS_IMAGE_URL,
        "forecast_lstm_issues_closed_at_url": FULL_PATH_LSTM_ISSUES_CLOSED_AT,
        "forecast_stats_issues_closed_at_url": FULL_PATH_STATS_ISSUES_CLOSED_AT,
        "week_of_the_day": f"The Day of the week with maximum issues closed are: {max_day}",
        "month_of_the_year": f"The month of the year with maximum issues closed are: {max_closed_month}"
    }
    print("Done with issues closed_at data")
    return jsonify(json_response)  


@app.route('/api/forecast', methods=['POST'])
def forecast():
    body = request.get_json()
    issues = body["issues"]
    type = body["type"]
    repo_name = body["repo"]
    data_frame = pd.DataFrame(issues)
    df1 = data_frame.groupby([type], as_index=False).count()
    df = df1[[type, 'issue_number']]
    df.columns = ['ds', 'y']

    df['ds'] = df['ds'].astype('datetime64[ns]')
    array = df.to_numpy()
    x = np.array([time.mktime(i[0].timetuple()) for i in array])
    y = np.array([i[1] for i in array])

    lzip = lambda *x: list(zip(*x))

    days = df.groupby('ds')['ds'].value_counts()
    Y = df['y'].values
    X = lzip(*days.index.values)[0]
    firstDay = min(X)

    '''
    To achieve data consistancy with both actual data and predicted values, 
    add zeros to dates that do not have orders
    [firstDay + timedelta(days=day) for day in range((max(X) - firstDay).days + 1)]
    '''
    Ys = [0, ]*((max(X) - firstDay).days + 1)
    days = pd.Series([firstDay + timedelta(days=i)
                      for i in range(len(Ys))])
    for x, y in zip(X, Y):
        Ys[(x - firstDay).days] = y

    # Modify the data that is suitable for LSTM
    Ys = np.array(Ys)
    Ys = Ys.astype('float32')
    Ys = np.reshape(Ys, (-1, 1))
    # Apply min max scaler to transform the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    Ys = scaler.fit_transform(Ys)
    # Divide training - test data with 80-20 split
    train_size = int(len(Ys) * 0.80)
    test_size = len(Ys) - train_size
    train, test = Ys[0:train_size, :], Ys[train_size:len(Ys), :]
    print('train size:', len(train), ", test size:", len(test))

    # Create the training and test dataset
    def create_dataset(dataset, look_back=1):
        X, Y = [], []
        for i in range(len(dataset)-look_back-1):
            a = dataset[i:(i+look_back), 0]
            X.append(a)
            Y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(Y)
    '''
    Look back decides how many days of data the model looks at for prediction
    Here LSTM looks at approximately one month data
    '''
    look_back = 30
    X_train, Y_train = create_dataset(train, look_back)
    X_test, Y_test = create_dataset(test, look_back)

    # Reshape input to be [samples, time steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    # Verifying the shapes
    X_train.shape, X_test.shape, Y_train.shape, Y_test.shape

    print("TEST ---------")
    print(X_train.shape)
    print(Y_train.shape)
    print(X_test.shape)
    print(Y_test.shape)

    # Model to forecast
    model = Sequential()
    model.add(LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Fit the model with training data and set appropriate hyper parameters
    history = model.fit(X_train, Y_train, epochs=20, batch_size=70, validation_data=(X_test, Y_test),
                        callbacks=[EarlyStopping(monitor='val_loss', patience=10)], verbose=1, shuffle=False)

    '''
    Creating image URL
    BASE_IMAGE_PATH refers to Google Cloud Storage Bucket URL.Add your Base Image Path in line 145
    if you want to run the application local
    LOCAL_IMAGE_PATH refers local directory where the figures generated by matplotlib are stored
    These locally stored images will then be uploaded to Google Cloud Storage
    '''
    BASE_IMAGE_PATH = os.environ.get(
        'BASE_IMAGE_PATH', 'https://storage.googleapis.com/lstm-storage/')
    LOCAL_IMAGE_PATH = "static/images/"

    # Creating the image path for model loss, LSTM generated image and all issues data image
    MODEL_LOSS_IMAGE_NAME = "model_loss_" + type +"_"+ repo_name + ".png"
    MODEL_LOSS_URL = BASE_IMAGE_PATH + MODEL_LOSS_IMAGE_NAME

    LSTM_GENERATED_IMAGE_NAME = "lstm_generated_data_" + type +"_" + repo_name + ".png"
    LSTM_GENERATED_URL = BASE_IMAGE_PATH + LSTM_GENERATED_IMAGE_NAME

    ALL_ISSUES_DATA_IMAGE_NAME = "all_issues_data_" + type + "_"+ repo_name + ".png"
    ALL_ISSUES_DATA_URL = BASE_IMAGE_PATH + ALL_ISSUES_DATA_IMAGE_NAME

    DAY_MAX_ISSUE_CREATED_IMAGE_NAME = "day_max_issues_created_data_" + type + "_"+ repo_name + ".png"
    DAY_MAX_ISSUE_CREATED_DATA_URL = BASE_IMAGE_PATH + DAY_MAX_ISSUE_CREATED_IMAGE_NAME

    DAY_MAX_ISSUE_CLOSED_IMAGE_NAME = "day_max_issues_closed_data_" + type + "_"+ repo_name + ".png"
    DAY_MAX_ISSUE_CLOSED_DATA_URL = BASE_IMAGE_PATH + DAY_MAX_ISSUE_CLOSED_IMAGE_NAME

    MONTH_MAX_ISSUE_CLOSED_IMAGE_NAME = "month_max_issues_closed_data_" + type + "_"+ repo_name + ".png"
    MONTH_MAX_ISSUE_CLOSED_DATA_URL = BASE_IMAGE_PATH + MONTH_MAX_ISSUE_CLOSED_IMAGE_NAME

    # Add your unique Bucket Name if you want to run it local
    BUCKET_NAME = os.environ.get('BUCKET_NAME', 'lstm-storage')

    # Model summary()

    # Plot the model loss image
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Test Loss')
    plt.title('Model Loss For ' + type)
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(loc='upper right')
    # Save the figure in /static/images folder
    plt.savefig(LOCAL_IMAGE_PATH + MODEL_LOSS_IMAGE_NAME)

    # Predict issues for test data
    y_pred = model.predict(X_test)

    # Plot the LSTM Generated image
    fig, axs = plt.subplots(1, 1, figsize=(10, 4))
    X = mdates.date2num(days)
    axs.plot(np.arange(0, len(Y_train)), Y_train, 'g', label="history")
    axs.plot(np.arange(len(Y_train), len(Y_train) + len(Y_test)),
             Y_test, marker='.', label="true")
    axs.plot(np.arange(len(Y_train), len(Y_train) + len(Y_test)),
             y_pred, 'r', label="prediction")
    axs.legend()
    axs.set_title('LSTM Generated Data For ' + type)
    axs.set_xlabel('Time Steps')
    axs.set_ylabel('Issues')
    # Save the figure in /static/images folder
    plt.savefig(LOCAL_IMAGE_PATH + LSTM_GENERATED_IMAGE_NAME)

    # Plot the All Issues data images
    fig, axs = plt.subplots(1, 1, figsize=(10, 4))
    X = mdates.date2num(days)
    axs.plot(X, Ys, 'purple', marker='.')
    locator = mdates.AutoDateLocator()
    axs.xaxis.set_major_locator(locator)
    axs.xaxis.set_major_formatter(mdates.AutoDateFormatter(locator))
    axs.legend()
    axs.set_title('All Issues Data')
    axs.set_xlabel('Date')
    axs.set_ylabel('Issues')
    # Save the figure in /static/images folder
    plt.savefig(LOCAL_IMAGE_PATH + ALL_ISSUES_DATA_IMAGE_NAME)

    x = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df = pd.DataFrame(issues)
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
    weekDF = df.groupby(df['created_at'].dt.day_name()).size()
    weekDF = pd.DataFrame({'Created_On':weekDF.index, 'Count':weekDF.values})
    weekDF = weekDF.groupby(['Created_On']).sum().reindex(x)
    max_issue_count = weekDF.max()
    max_issue_day = weekDF['Count'].idxmax()
    plt.figure(figsize=(12, 7))
    plt.plot(weekDF['Count'], label='Issues')
    plt.title(f"Number of Issues Created for particular Week Days. {max_issue_count} and  day is {max_issue_day}")
    plt.ylabel('Number of Issues count')
    plt.xlabel('Week in Days')
    plt.savefig(LOCAL_IMAGE_PATH + DAY_MAX_ISSUE_CREATED_IMAGE_NAME)

    df['closed_at'] = pd.to_datetime(df['closed_at'], errors='coerce')
    weekDF = df.groupby(df['closed_at'].dt.day_name()).size()
    weekDF = pd.DataFrame({'Closed_On':weekDF.index, 'Count':weekDF.values})
    weekDF = weekDF.groupby(['Closed_On']).sum().reindex(x)
    max_issue_count_closed = weekDF.max()
    max_issue_day_closed = weekDF['Count'].idxmax()
    plt.figure(figsize=(12, 7))
    plt.plot(weekDF['Count'], label='Issues')
    plt.title('Number of Issues Closed for particular Week Days with day as {max_issue_day_closed} and count as {max_issue_count_closed}')
    plt.ylabel('Number of Issues')
    plt.xlabel('Week Days')
    plt.savefig(LOCAL_IMAGE_PATH + DAY_MAX_ISSUE_CLOSED_IMAGE_NAME)


    df = pd.DataFrame(issues)
    x = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    df['closed_at'] = pd.to_datetime(df['closed_at'], errors='coerce')
    monthDF = df.groupby(df['closed_at'].dt.month_name()).size()
    monthDF = pd.DataFrame({'Closed_On':monthDF.index, 'Count':monthDF.values})
    monthDF = monthDF.groupby(['Closed_On']).sum().reindex(x)
    max_issue_count_closed_month = monthDF.max()
    max_issue_closed_month = monthDF['Count'].idxmax()
    plt.figure(figsize=(12, 7))
    plt.plot(monthDF['Count'], label='Issues')
    plt.title('Number of Issues Closed for particular Month has month name :{max_issue_closed_month} with the count {max_issue_count_closed_month}.')
    plt.ylabel('Number of Issues')
    plt.xlabel('Names of Month')
    plt.savefig(LOCAL_IMAGE_PATH + MONTH_MAX_ISSUE_CLOSED_IMAGE_NAME)

    # Uploads an images into the google cloud storage bucket
    bucket = client.get_bucket(BUCKET_NAME)
    
    new_blob = bucket.blob(MODEL_LOSS_IMAGE_NAME)
    new_blob.upload_from_filename(filename=LOCAL_IMAGE_PATH + MODEL_LOSS_IMAGE_NAME)

    new_blob = bucket.blob(ALL_ISSUES_DATA_IMAGE_NAME)
    new_blob.upload_from_filename(filename=LOCAL_IMAGE_PATH + ALL_ISSUES_DATA_IMAGE_NAME)

    new_blob = bucket.blob(LSTM_GENERATED_IMAGE_NAME)
    new_blob.upload_from_filename(filename=LOCAL_IMAGE_PATH + LSTM_GENERATED_IMAGE_NAME)

    new_blob = bucket.blob(DAY_MAX_ISSUE_CREATED_IMAGE_NAME)    
    new_blob.upload_from_filename(filename=LOCAL_IMAGE_PATH + DAY_MAX_ISSUE_CREATED_IMAGE_NAME)

    new_blob = bucket.blob(DAY_MAX_ISSUE_CLOSED_IMAGE_NAME)    
    new_blob.upload_from_filename(filename=LOCAL_IMAGE_PATH + DAY_MAX_ISSUE_CLOSED_IMAGE_NAME)

    new_blob = bucket.blob(MONTH_MAX_ISSUE_CLOSED_IMAGE_NAME)    
    new_blob.upload_from_filename(filename=LOCAL_IMAGE_PATH + MONTH_MAX_ISSUE_CLOSED_IMAGE_NAME)
    

    # Construct the response
    json_response = {
        "model_loss_image_url": MODEL_LOSS_URL,
        "lstm_generated_image_url": LSTM_GENERATED_URL,
        "all_issues_data_image": ALL_ISSUES_DATA_URL,
        "day_max_issue_created": DAY_MAX_ISSUE_CREATED_DATA_URL,
        "day_max_issue_closed": DAY_MAX_ISSUE_CLOSED_DATA_URL,
        "month_max_issues_closed": MONTH_MAX_ISSUE_CLOSED_DATA_URL
        
    }
    # Returns image url back to flask microservice
    return jsonify(json_response)


@app.route('/api/getAllForecasts', methods=['GET'])
def getAllForecasts():
    return jsonify({
        "name":"Deneshwara Sai Ila"
    })

# Run LSTM app server on port 8080
if __name__ == '__main__':
    print("Server Started")
    app.run(debug=True, host='0.0.0.0', port=8080)
