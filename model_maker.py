
"""
scripts trains models using time-series
and predicts future values


Best machine learning models for time series forecasting

Naïve model.
Exponential smoothing model.
ARIMA/SARIMA.
Linear regression method.
Multi-Layer Perceptron (MLP)
Recurrent Neural Network (RNN)
Long Short-Term Memory (LSTM)

Generative AI Models good for time series:
Autoregressive Models:
	Generate sequences by modeling the conditional 
	probability of each data point given the previous data points.
		ARIMA: Autoregressive Integrated Moving Average 
		SARIMA: Seasonal Autoregressive Integrated Moving-Average

RNN / Recurrent Neural Networks:
	Good for forecasting and anomaly detection.
	NN (neural networks) classs that has recurrent connections,
	allowing them to maintain a memory of previous inputs.
	* LSTM: Long Short-Term Memory Networks: is a specific type of 
		RNN that addresses the vanishing gradient problem, making it more 
		effective in capturing long-term dependencies in time series data.

GAN / Generative Adversarial Networks:
	Good for synthetic time series data, and data augmentation.
	GANs are a type of generative model that consists of two neural
	networks, a generator, and a discriminator.

VAE / Variational Autoencoders:
	Good for anomaly detection and missing data imputation.
	VAEs are another type of generative model that can learn a probabilistic
	representation of the input data.

Transformer Models:
	Good for parallel computation
	Transformers, originally developed for natural language processing,
	have shown promise in processing sequential data, 
	including time series. They can capture long-range dependencies.

"""
import os

import pandas
import numpy

from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense

#==================================================================================
#==================================================================================
# MODELING

def model_make_lstm(X, y,
                    path_2save_model,
                    dl_input_shape,
                    dl_units = 50,
                    dense_units = (5,),
                    epochs = 100,
                    train_batch_size = 32,
                    save_model = False):
    """ Define and train an LSTM model
        LSTM = Long Short-Term Memory Network
    Args:
    """    
    # Define the LSTM model architecture
    model = Sequential()
    model.add(LSTM(units = dl_units,
                   input_shape = dl_input_shape,
                   activation='relu',
                   kernel_initializer='lecun_uniform'))
    for dense_unit in dense_units:
        # Output layer to predict the next number of values to predict
        model.add(Dense(units = dense_unit))

    # Compile the model
    model.compile(optimizer='adam',
                  loss='mean_squared_error')
    
    # Train the model
    model.fit(X, y, epochs = epochs, batch_size = train_batch_size, verbose=0)
    if save_model:
        model.save(path_2save_model)
    return model

def model_make_ilqr(X,
                    column_to_predict = 0,
                    num_steps_to_predict = 5):
    """model based on the iLQR = Iterative linear quadratic regulator algorithm
    Args:
        X = numpy.array
        column_to_predict = int() number of columns to predict
        num_steps_to_predict = int(), number of steps to predict
        must verify cost
    """
    try:
        from ilqr import iLQR

        # Define the function to compute the cost
        def cost(x, u):
            # function to compute the cost function
            # will assume a quadratic cost
            return numpy.sum((x[:, column_to_predict] - u) ** 2)

        # Define the initial guess for the control sequence
        initial_guess = numpy.zeros(num_steps_to_predict)

        # Define the iLQR optimizer
        optimizer = iLQR(X[:, :grid.shape[1]], cost)

        # Run the optimization to find the optimal control sequence
        u_optimal, x_optimal = optimizer.fit(initial_guess)

        # Predict the next 5 values of the column using the optimal control sequence
        predicted_values = []
        current_state = X[-1, :]
        for u in u_optimal:
            # Apply control sequence to predict next state
            current_state = current_state + u
            predicted_values.append(current_state[column_to_predict])

        return predicted_values

    except ImportError:
        print("            cannot import ilqr module")
        return list()


#==================================================================================
#==================================================================================
# PREPARING DATA for model training

def prepare_data_4modelling(data,
                            look_back = 10,
                            nr_vals_2predict = 5,
                            nr_splits = 30):
    """
    Args:
        data: numpy.ndarray, data for modelling
        look_back: int(), number of splits/columns to create from the dataframe
        nr_vals_2predict: number of values to predict
    Return:
        X: list(), data X used to analyze
        y: list(), data y that is predicted
    """
    X, y = [], []
    if len(data.shape) > 1:
        for i in range(len(data) - look_back - nr_vals_2predict):
            X.append(data[i:i + look_back])
            # Predict next nr_vals_2predict values of column 0
            y.append(data[i + look_back:i + look_back + nr_vals_2predict, 0])
    else:
        for i in range(nr_splits, data.shape[0]):
            X.append(data[i-nr_splits:i, 0])
            y.append(data[i, 0])
    return numpy.array(X), numpy.array(y)

#==================================================================================
#==================================================================================

def predict_tick_with_correlations(data, look_back, nr_vals_2predict,
                                   user = "default"):
    """Use the trained model to make predictions on a new grid of n columns
    Args:
        data = pandas.DataFrame()
        look_back = 
        nr_vals_2predict =
    Return:
        predicted_values: list()
    """
    # Prepare new data for prediction
    _, _, _, p_models, _ = dm.get_path_sourcedata(user)
    model_name = f'model.keras'
    path_2save_model = os.path.join(p_models, model_name)

    data_np = data.values
    X, y = prepare_data_4modelling(data_np,
                                   look_back,
                                   nr_vals_2predict)
    # Reshape input data to match LSTM input shape
    X = numpy.reshape(X, (X.shape[0], X.shape[1], X.shape[-1]))  # Assuming n columns
    model = model_make_lstm(X, y,
                           path_2save_model,
                           dl_input_shape = (look_back, X.shape[-1]),
                           dl_units = 50,
                           dense_units = (nr_vals_2predict,),
                           epochs = 100,
                           train_batch_size = 32)

    X_new, _ = prepare_data_4modelling(data_np,
                                                 look_back,
                                                 nr_vals_2predict)

    # Reshape input data to match LSTM input shape
    X_new = numpy.reshape(X_new, (X_new.shape[0],
                                  X_new.shape[1],
                                  data_np.shape[1]))  # Assuming n columns

    # Make predictions using the trained model
    predicted_values = model.predict(X_new)

    return predicted_values[-1:]


def get_predicted_data(ticker,
                       df,
                       interval = '1d',
                       value_used = "Close",
                       days_backtrack = 35,
                       user = "default"):
    """predict trend of a ticker
    Args:
        ticker = str() of ticker
        interval = str() interval to use for analysis, same as interval for data
        value_used = column to be used for prediction, default is the Close column
        days_backtrack = days to use to calculate the prediction
        user = username to be used for analysis
    Return:
        plot images
    Initiation source:
        https://www.codespeedy.com/predict-next-sequence-using-deep-learning-in-python/#google_vignette
    """
    _, _, _, p_models, _ = dm.get_path_sourcedata(user)
    path_2save_model = os.path.join(p_models, f'model_{ticker}.keras')

    train = df[value_used].values

    # Adding one row that is a copy of the last row
    df.loc[len(df.index)] = [i for i in df.loc[df.index.tolist()[-1]]]
    test = df[value_used].values

    sc = MinMaxScaler(feature_range = (0, 1))
    train_sd = sc.fit_transform(train.reshape(-1,1))
    X_train_sd, y_train_sd = prepare_data_4modelling(train_sd,
                                                 nr_splits = 30)

    #CREATING the models
    X_train = numpy.reshape(X_train_sd,
                                (X_train_sd.shape[0],
                                 X_train_sd.shape[1], 1))
    look_back = X_train.shape[1]
    model = model_make_lstm(X_train, y_train_sd,
                                 path_2save_model,
                                 dl_input_shape = (look_back, 1),
                                 dl_units = 16,
                                 dense_units = (8, 4, 2, 1),
                                 epochs = 45,
                                 train_batch_size = 4)

    predicted_train_sd = model.predict(X_train_sd)
    predicted_train_sd = sc.inverse_transform(predicted_train_sd)

    test_sd = sc.fit_transform(test.reshape(-1,1))
    X_test_sd, y_test_sd = prepare_data_4modelling(test_sd,
                                               nr_splits = 30)
    # X_test_sd = numpy.reshape(X_test_sd, (X_test_sd.shape[0], X_test_sd.shape[1], 1))
    predicted_test_sd = model.predict(X_test_sd)
    predicted_test_sd = sc.inverse_transform(predicted_test_sd)

    data_models = {"ticker":ticker,
                   "path": path_2save_model,
                    "data": df,
                    "train": train,
                    "test": test,
                    "X_train": X_train_sd,
                    "y_train": y_train_sd,
                    "predicted_train": predicted_train_sd,
                    "predicted_test": predicted_test_sd}
    predicted_values = [i[0] for i in predicted_test_sd[-10:]]
    return predicted_values


def modmak(df,
           ticker,
           user,
           look_back = 10):
    """
    Creates models/ AI brains
    Args:
        look_back: number of previous time steps to use for prediction
    Return:
        saves model to ah h5 file
    """
    _, _, _, p_models, _ = dm.get_path_sourcedata(user)
    path_2save_model = os.path.join(p_models, f'lstm_{ticker}.keras')
    # Convert the "value" column to a NumPy array
    data = df['Close'].values

    # # Normalize the data if needed
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # normalized_data = scaler.fit_transform(data)


    # Initialize lists to store training data
    X_train = []
    y_train = []

    # Iterate through the data to create training sequences
    for i in range(len(data) - look_back):
        # Extract a sequence of look_back time steps as input
        X_train.append(data[i : i + look_back])
        # The next value is the target for prediction
        y_train.append(data[i + look_back])

    # Convert the lists to NumPy arrays
    X_train = numpy.array(X_train)
    y_train = numpy.array(y_train)

    # Reshape X_train to match the input shape expected by the LSTM model
    # X_train = numpy.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_train = numpy.reshape(X_train, (X_train.shape[0], look_back, 1))
    # Now, X_train contains the input sequences and y_train contains the target values


    model = model_make_lstm(X_train, y_train,
                                     path_2save_model,
                                     dl_input_shape = (look_back, 1),
                                     dl_units = 50,
                                     dense_units = (1,),
                                     epochs = 100,
                                     train_batch_size = 1)

    # denormalized_predictions = scaler.inverse_transform(predictions)
    # denormalized_y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# # Load the trained model
# loaded_model = tf.keras.models.load_model(path_2model)
# look_back = 10


# df = dm.get_data(ticker,
#                   interval = '1d',
#                   user = "default")

# predicted = {ticker:{}}
# for col in ("Close", "High", "Low", "Open"):
#     recent_values = df[col].values


#     # Prepare data for prediction
#     # recent_data = numpy.array([recent_values]) #! this code doesn't work
#     recent_data = []
#     for i in range(len(recent_values) - look_back):
#         # Extract a sequence of look_back time steps as input
#         recent_data.append(recent_values[i : i + look_back])
#     recent_data = numpy.array(recent_data)


#     # Reshape the recent_data to match the input shape expected by the model
#     recent_data = numpy.reshape(recent_data, (recent_data.shape[0], look_back, 1))

#     predicted_values = list()
#     # Make predictions using the loaded model
#     for i in range(2):
#         predicted_value = loaded_model.predict(recent_data)
#         predicted_values.append(predicted_value)

#         # # update the recent_data with the predicted value and rerun the prediction process
#         # recent_data = numpy.roll(recent_data, -1, axis=1)  # Shift the data by one step
#         # recent_data[0, -1, 0] = predicted_value

#     predicted[ticker][col] = predicted_values

# print("last date in dataframe: ", df["Date"].tolist()[-1])
# for ticker in predicted:
#     print(f"Predicting Value for ticker: {ticker}")
#     for col in predicted[ticker]:
#         print(f"{col} : {predicted[ticker][col]}")



# def predict_tick_with_correlations(data,
#                                    ticker,
#                                    tickers_correl,
#                                    look_back = 10,
#                                    nr_vals_2predict = 5,
#                                    save_model = False,
#                                    user = "default"):
#     """
#     script will:
#     1. Analyze the correlation between column A and all other columns.
#     2. Save the correlation results in new columns.
#     3. Use the newly created grid to predict the next 5 values of column A.
#     Args:
#         data = pandas.DataFrame() with columns of ticker and columns tickers_correl
#         ticker = name of the ticker and name of the column in data to be used
#         tickers_correl = names of other columns/tickers to be used for correlation
#         look_back = number of previous time steps to use for prediction
#         nr_vals_2predict = number of next values to predict
#         save_model = if True, model will be saved
#     Return:
#         tuple of predicted next 5 prices
#     """
#     _, _, _, p_models, _ = dm.get_path_sourcedata(user)
#     path_2save_model = os.path.join(p_models, f'lstm_{ticker}.h5')

#     model = model_make_lstm(data, path_2save_model, look_back, save_model)
#     # Step 1: Analyze correlation between column A and all other columns
#     correlation_results = data.corr()[ticker].drop(ticker)  # Correlation between ticker and other columns
#     print("Correlation Results:\n", correlation_results)

#     # # Step 2: Save the correlation results in new columns
#     # for col in correlation_results.index:
#     #     data[f'corr_{col}_{ticker}'] = correlation_results[col]

#     # Step 3: Prepare data for LSTM model to predict the next nr_vals_2predict values of column ticker
#     time_series_data = data.values
#     print(data.shape)
#     print(time_series_data.shape)

#     # Normalize the data
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     scaled_data = scaler.fit_transform(time_series_data)
#     X, y = prepare_data_4modelling(scaled_data, look_back, nr_vals_2predict)

#     # Reshape the input data to fit the LSTM input shape [samples, time steps, features]
#     X = numpy.reshape(X, (X.shape[0], X.shape[1], time_series_data.shape[1]))  

#     # # Load the trained LSTM model
#     # loaded_model = tf.keras.models.load_model('trained_lstm_model.h5')

#     # Predict the next nr_vals_2predict values of column A
#     predicted_values_scaled = model.predict(X[-1].reshape(1, look_back, nr_vals_2predict))
#     print(X.shape)
#     print(predicted_values_scaled.shape)

#     # Inverse scaling to get actual predicted values
#     print(time_series_data[-1])
#     print(time_series_data[-1][0])
#     print(predicted_values_scaled[0])
#     print((time_series_data[-1][0], predicted_values_scaled[0]))
#     new_predicted_data = numpy.concatenate((time_series_data[-1][0],
#                                             predicted_values_scaled[0]))[:, numpy.newaxis]
#     predicted_values = scaler.inverse_transform(new_predicted_data)

#     print("Predicted next 5 values of column A:", predicted_values[1:])
#     return predicted_values[1:]


# def model_make_lstm(data,
#                                path_2save_model,
#                                look_back = 10,
#                                save_model = False):
#     """make model for predicting values based on multiple correlations
#     Args:
#         data = pandas.DataFrame() with data for prediction
#         look_back = number of previous time steps to use for prediction
#         save_model = if True, model will be saved
#     Return:
#         model
#     """
#     # Define the LSTM model architecture
#     model = tf.keras.Sequential([
#         tf.keras.layers.LSTM(units=50,
#                             return_sequences=True,
#                             input_shape=(look_back, data.shape[1])),
#         tf.keras.layers.Dropout(0.2),
#         tf.keras.layers.LSTM(units=50,
#                             return_sequences=False),
#         tf.keras.layers.Dropout(0.2),
#         tf.keras.layers.Dense(units=1)
#     ])
#     # Compile the model
#     model.compile(optimizer='adam', loss='mean_squared_error')


#     # Select columns as features for training
#     features = data[list(data.columns)]

#     # Normalize the data
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     scaled_data = scaler.fit_transform(features)

#     X, y = prepare_data_4modelling(scaled_data, look_back)

#     # Reshape the input data to fit the LSTM input shape [samples, time steps, features]
#     X = numpy.reshape(X,
#                       (X.shape[0], X.shape[1],
#                         len(data.columns)))

#     # Train the model
#     model.fit(X, y, epochs=100, batch_size=32)

#     # Save the trained model
#     if save_model:
#         model.save(path_2save_model)
    
#     return model



# def model_make_lstm(X, y,
#                        look_back,
#                        path_2save_model,
#                        dl_units = 50,
#                        dense_units = (1,),
#                        epochs = 100,
#                        train_batch_size = 1):
#     model = Sequential()
#     model.add(LSTM(units = dl_units,
#                     input_shape=(look_back, 1)),
#                     activation='relu',
#                     kernel_initializer='lecun_uniform')
#     for dense_unit in dense_units:
#         model.add(Dense(units = dense_unit))
#     # model.add(Dense(1))
#     model.compile(optimizer='adam',
#                   loss='mean_squared_error')

#     # Train the model
#     model.fit(X, y, epochs=epochs, batch_size=train_batch_size)#, verbose=2)
#     model.save(path_2save_model)

#     return model
