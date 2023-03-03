## Trading Bot with Technical Analysis

This Python program is a trading bot that uses technical analysis to predict whether to buy, sell or hold an asset. It uses the yfinance library to download historical price data, the finta library to calculate technical indicators, and a deep learning model trained with Keras to predict future price movements.

## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

## Prerequisites
This project requires Python 3 and the following libraries:

*yfinance
*finta
*keras
*pickle
*requests
*schedule
*tqdm
*numpy
*matplotlib
You can install these libraries using pip by running:

Copy code:
"pip install -r requirements.txt"

## Running the Program
To run the program, simply execute the main.py file. This will start the trading bot, which will run every day at 8:00 am and send a Telegram message with the recommended action (buy, sell or hold) for the given asset.

## How It Works
The program first downloads historical price data using the yfinance library. It then calculates several technical indicators using the finta library, such as RSI, Williams %R, SMA, EMA, WMA, HMA, TEMA, CCI, CMO, MACD, PPO, ROC, CFI, DMI, and SAR.

Next, the program scales the indicator data using a MinMax scaler and feeds it into a deep learning model trained with Keras. The model predicts the recommended action (buy, sell or hold) based on the indicator data then push message to telegram.
