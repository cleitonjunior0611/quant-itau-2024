# 📈 IBOVESPA Prediction with LSTM and Technical Analysis

This project implements a **recurrent neural network (LSTM)** model to predict **IBOVESPA** prices using historical data from Yahoo Finance.  
In addition to the predictions, **technical indicators (RSI)** are applied to generate **buy and sell signals**, simulating trading operations with cumulative capital calculation.

## ⚙️ Features

- 📊 **Automatic download** of historical IBOVESPA data via `yfinance`
- 🔄 **Data preprocessing** with normalization and creation of time windows
- 🧠 **Optimized LSTM model** for financial time series prediction
- 📉 **RSI indicator** to identify overbought and oversold conditions
- 💰 **Trading simulation**, including selling fees and capital evolution
- 📈 **Graphical visualizations**:
  - Real price vs predicted price
  - Buy and sell signals
  - Capital evolution after each operation
  - Loss curve (training vs validation)

## 🛠️ Technologies Used

- [Python 3.9+](https://www.python.org/)
- [TensorFlow / Keras](https://www.tensorflow.org/)
- [yfinance](https://pypi.org/project/yfinance/)
- [scikit-learn](https://scikit-learn.org/)
- [pandas](https://pandas.pydata.org/)
- [numpy](https://numpy.org/)
- [matplotlib](https://matplotlib.org/)
- [ta (Technical Analysis)](https://technical-analysis-library-in-python.readthedocs.io/)

