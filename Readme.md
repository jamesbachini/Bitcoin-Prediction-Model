
# Bitcoin Prediction Model

This repository contains scripts and machine learning models for predicting Bitcoin prices using Keras and Tensorflow.

The full tutorial for this script is available here: https://jamesbachini.com/chatgpt-keras-bitcoin-bot/

Below are the instructions on how to use each script.

## Table of Contents

- [Getting Started](#getting-started)
- [Collecting Data](#collecting-data)
- [Analyzing Data](#analyzing-data)
- [Creating Models](#creating-models)
- [Making Predictions](#making-predictions)
- [Simulating Trading](#simulating-trading)
- [More Information](#more-information)

## Getting Started

Clone this repository to your local machine:

```bash
git clone https://github.com/jamesbachini/Bitcoin-Prediction-Model.git
cd Bitcoin-Prediction-Model
```

Ensure you have the necessary Python packages installed. You can install the required packages using:

```bash
pip install -r requirements.txt
```

## Collecting Data

To collect information from the exchange, use the \`data-logger.py\` script. This script fetches market data and saves it for further analysis.

```bash
python data-logger.py
```

This will create a data file that contains the latest Bitcoin market information collected from the following exchanges:
    - binance_spot - BTC/USDT
    - binance_futures - BTC/USDT
    - coinbase - BTC/USD
    - kraken - BTC/USD
    - okx - BTC/USDT
    - bybit - BTC/USDT

A sample file is provided at bitcoin_prices_2024_07_23.csv

## Analyzing Data

Once the data is collected, you can analyze it using the \`analysis.py\` script. This script processes the collected data and prepares it for model training. Change the name of the csv file in the script to match your recently collected data.

```bash
python analysis.py
```

## Creating Models

### Low Time Frame Model

To create a low time frame (LTF) model, use the \`ltf-model.py\` script. This script trains a model on short time frame data. Again change the csv filename to match your recently collected data.

```bash
python ltf-model.py
```

### High Time Frame Model

To create a high time frame (HTF) model, use the \`htf-model.py\` script. This script trains a model on longer time frame data. Again change the csv filename to match your recently collected data.

```bash
python htf-model.py
```

## Making Predictions

### Low Time Frame Predictions

To get Bitcoin price predictions using the low time frame model, run the \`ltf-predictions.py\` script.

```bash
python ltf-predictions.py
```

### High Time Frame Predictions

To get Bitcoin price predictions using the high time frame model, run the \`htf-predictions.py\` script.

```bash
python htf-predictions.py
```

## Simulating Trading

You can simulate a trading situation and check profitability against live data using the \`mock-trading.py\` script. Note that this simulation does not take into account execution costs such as trading fees and slippage.

```bash
python mock-trading.py
```

## More Information

For more details and information, you can visit:

- [James Bachini's Website](https://jamesbachini.com)
- [James Bachini's YouTube Channel](https://www.youtube.com/c/JamesBachini?sub_confirmation=1)
- [James Bachini's Substack](https://bachini.substack.com)
- [James Bachini's Podcast on Spotify](https://podcasters.spotify.com/pod/show/jamesbachini)
- [James Bachini's Podcast on Spotify](https://open.spotify.com/show/2N0D9nvdxoe9rY3jxE4nOZ)
- [James Bachini on Twitter](https://twitter.com/james_bachini)
- [James Bachini on LinkedIn](https://www.linkedin.com/in/james-bachini/)
- [James Bachini on GitHub](https://github.com/jamesbachini)

Feel free to reach out or follow for more updates and information.
