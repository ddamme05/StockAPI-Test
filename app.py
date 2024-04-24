from flask import Flask, jsonify, request
import requests
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
from textblob import TextBlob

from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)

FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY')
NEWSAPI_API_KEY = os.getenv('NEWSAPI_API_KEY')
ALPHAVANTAGE_API_KEY = os.getenv('ALPHAVANTAGE_API_KEY')

ts = TimeSeries(key=ALPHAVANTAGE_API_KEY, output_format="pandas")

@app.route("/stock/finnhub/<symbol>")
def get_stock_data_finnhub(symbol):
    """Fetch stock data for a given symbol from Finnhub."""
    finnhub_url = (
        f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={FINNHUB_API_KEY}"
    )
    response = requests.get(finnhub_url)
    data = response.json()
    return jsonify(data)

@app.route("/stock/yfinance/historical/<symbol>")
def get_historical_data_yfinance(symbol):
    """Fetch historical stock data for a given symbol using yfinance with customizable period and interval."""
    period = request.args.get('period', '1y')  
    interval = request.args.get('interval', '1d') 

    ticker = yf.Ticker(symbol)
    # Fetch the historical data based on period and interval
    data = ticker.history(period=period, interval=interval)
    # Reset index to convert the index to a column
    data.reset_index(inplace=True)
    # Convert the 'Date' column to string to avoid serialization issues
    data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')
    # Convert the DataFrame to a dictionary for easy JSON serialization
    data_dict = data.to_dict(orient="records")
    return jsonify(data_dict)



@app.route("/stock/alphavantage/<symbol>")
def get_stock_data_alphavantage(symbol):
    """Fetch stock data for a given symbol from AlphaVantage."""
    data, meta_data = ts.get_quote_endpoint(symbol=symbol)
    return jsonify(data.to_dict())


@app.route("/stock/yfinance/<symbol>")
def get_stock_data_yfinance(symbol):
    """Fetch stock data for a given symbol using yfinance."""
    ticker = yf.Ticker(symbol)
    # Fetching the historical data for the past day
    data = ticker.history(period="1d")
    # Reset index to convert the index to a column
    data.reset_index(inplace=True)
    # Convert the 'Date' column to string to avoid serialization issues
    data["Date"] = data["Date"].dt.strftime("%Y-%m-%d")
    # Convert the DataFrame to a dictionary
    data_dict = data.to_dict(orient="records")
    return jsonify(data_dict)


@app.route("/news/<keyword>")
def get_news(keyword):
    """Fetch news articles related to a given keyword from NewsAPI."""
    newsapi_url = "https://newsapi.org/v2/everything"
    parameters = {
        "q": keyword,  # Query term
        "language": "en",  # Filter articles by English language
        "sortBy": "relevancy",  # Sort by relevancy to the keyword
        "apiKey": NEWSAPI_API_KEY,
    }
    response = requests.get(newsapi_url, params=parameters)
    data = response.json()

    # Check if the API request was successful and has articles
    if response.status_code == 200 and "articles" in data:
        articles = data["articles"][:10]  # Take the top 10 relevant articles
        return jsonify(articles)
    else:
        # Handle the error or no articles found scenario
        return jsonify({"error": "Unable to fetch articles or no articles found"}), 400


@app.route("/news/sentiment/<keyword>")
def get_news_sentiment(keyword):
    """Fetch news articles, analyze sentiment, and calculate overall sentiment for a given keyword."""
    parameters = {
        "q": keyword,
        "language": "en",
        "sortBy": "relevancy",
        "apiKey": NEWSAPI_API_KEY,
    }
    response = requests.get("https://newsapi.org/v2/everything", params=parameters)
    data = response.json()

    if response.status_code == 200 and "articles" in data:
        articles = data["articles"][
            :10
        ]  # Limit to the top 10 relevant articles for analysis
        total_polarity = 0
        analyzed_articles = 0

        for article in articles:
            if article["description"]:
                blob = TextBlob(article["description"])
                sentiment = blob.sentiment
                total_polarity += sentiment.polarity
                analyzed_articles += 1
                article["sentiment"] = {
                    "polarity": sentiment.polarity,
                    "subjectivity": sentiment.subjectivity,
                }

        # Calculate the average polarity if any articles were analyzed
        if analyzed_articles > 0:
            average_polarity = total_polarity / analyzed_articles
            overall_sentiment = "Neutral"
            if average_polarity > 0.1:  # These thresholds can be adjusted
                overall_sentiment = "Positive"
            elif average_polarity < -0.1:  # These thresholds can be adjusted
                overall_sentiment = "Negative"

            return jsonify(
                {
                    "average_polarity": average_polarity,
                    "overall_sentiment": overall_sentiment,
                    "analyzed_articles": analyzed_articles,
                }
            )
        else:
            return (
                jsonify(
                    {
                        "error": "No articles found with descriptions for sentiment analysis"
                    }
                ),
                400,
            )
    else:
        return jsonify({"error": "Unable to fetch articles or no articles found"}), 400


if __name__ == "__main__":
    app.run(debug=True)
