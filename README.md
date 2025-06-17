# Price-Prediction-using-TensorFlow
Machine learning project using TensorFlow

Data Preprocessing

The code starts by installing the necessary libraries, including TensorFlow, Pandas, and Matplotlib.
It then reads in a CSV file containing stock data, likely historical stock prices and related features.
Set the `STOCK_CSV` environment variable to the path of the `all_stocks_5yr.csv` file before running the script.
The data is preprocessed by converting the 'date' column to a datetime format and scaling the 'close' prices using the MinMaxScaler from scikit-learn.
The scaled data is then split into training and testing sets, with the training set comprising 95% of the data.
The training data is further processed by creating input features (x_train) and labels (y_train) using a sliding window approach. This is a common technique for preparing time series data for sequence prediction tasks.

Model Architecture

The code defines a Sequential model in Keras, which is a popular deep learning library.
The improved model stacks three LSTM layers with 128, 128, and 64 units respectively.
After the LSTM layers, the model includes a Dense layer with 64 units and a Dropout layer with a rate of 0.5 to reduce overfitting.
The final layer is a Dense layer with a single unit, which is used for the regression task of predicting stock prices.

Model Training and Evaluation

Compiling the model with appropriate loss function and optimizer.
Training the model on the x_train and y_train data.
Evaluating the model's performance on the test set (x_test and y_test).
Potentially fine-tuning the model hyperparameters or architecture based on the evaluation results.

The script also calculates the Mean Absolute Percentage Error (MAPE) and an accuracy percentage defined as `100 - MAPE`.
