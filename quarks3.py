from jugaad_data.nse import NSELive, stock_df
from datetime import datetime, date, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import os
import base64
from typing import List, Dict
from matplotlib import pyplot as plt
import json


# Add this class just below your imports
class PortfolioEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        return super().default(obj)


# Initialize NSELive for live data
nse = NSELive()


def get_stock_price(symbol, live=True):
    """Fetch live or historical stock price"""
    if live:
        try:
            quote = nse.stock_quote(symbol)
            return quote['priceInfo']['lastPrice']
        except Exception as e:
            print(f"Error fetching live data for {symbol}: {e}")
            return None
    else:
        try:
            today = date.today()
            df = stock_df(symbol, from_date=today, to_date=today, series="EQ")
            return df.iloc[-1]['CLOSE'] if not df.empty else None
        except Exception as e:
            print(f"Error fetching historical data for {symbol}: {e}")
            return None


def get_historical_price(symbol, date_str):
    """Fetch historical closing price for a specific date (YYYY-MM-DD format)"""
    try:
        # Convert input date to date object
        target_date = datetime.strptime(date_str, "%Y-%m-%d").date()

        # Fetch data for a range of dates (7 days before and after the target date)
        from_date = target_date - timedelta(days=7)
        to_date = target_date + timedelta(days=7)

        # Fetch historical data
        df = stock_df(symbol, from_date=from_date, to_date=to_date, series="EQ")

        if df.empty:
            print(f"No data found for {symbol} between {from_date} and {to_date}.")
            return None

        # Filter for the exact target date
        target_data = df[df['DATE'] == target_date.strftime("%Y-%m-%d")]

        if target_data.empty:
            print(f"No data found for {symbol} on {target_date}.")
            return None
        print(target_data.iloc[-1]['CLOSE'])

        return target_data.iloc[-1]['CLOSE']

    except Exception as e:
        print(f"Error fetching historical price for {symbol} on {date_str}: {e}")
        return None


class Simulation:
    def __init__(self, name, cash):
        self.name = name
        self.timestamp = datetime.now()
        self.logs = []
        self.images = []  # Store image paths
        self.portfolio = {
            'cash': cash,
            'holdings': {},
            'transactions': [],
            'performance_images': []  # Store backtest result images
        }

    def buy_stock(self, symbol, quantity, price=None, live=True):
        """Buy a stock and add it to the portfolio"""
        if price is None:
            price = get_stock_price(symbol, live)
            if price is None:
                print(f"Failed to fetch price for {symbol}. Transaction aborted.")
                self.logs.append(f"Failed to fetch price for {symbol}. Transaction aborted.")
                return

        cost = price * quantity
        if self.portfolio['cash'] >= cost:
            self.portfolio['cash'] -= cost
            if symbol in self.portfolio['holdings']:
                # Update average price and quantity
                total_qty = self.portfolio['holdings'][symbol]['quantity'] + quantity
                total_invested = self.portfolio['holdings'][symbol]['avg_price'] * self.portfolio['holdings'][symbol][
                    'quantity'] + cost
                self.portfolio['holdings'][symbol]['quantity'] = total_qty
                self.portfolio['holdings'][symbol]['avg_price'] = total_invested / total_qty
            else:
                self.portfolio['holdings'][symbol] = {'quantity': quantity, 'avg_price': price}

            # Record transaction
            self.portfolio['transactions'].append({
                'type': 'BUY',
                'symbol': symbol,
                'quantity': quantity,
                'price': price,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            print(f"Bought {quantity} shares of {symbol} at ₹{price:.2f}")
            self.logs.append(f"Bought {quantity} shares of {symbol} at ₹{price:.2f}")
        else:
            print(f"Insufficient cash to buy {quantity} shares of {symbol}.")
            self.logs.append(f"Insufficient cash to buy {quantity} shares of {symbol}.")

    def sell_stock(self, symbol, quantity, price=None, live=True):
        """Sell a stock and update the portfolio"""
        if symbol not in self.portfolio['holdings']:
            print(f"{symbol} not in portfolio.")
            self.logs.append(f"{symbol} not in portfolio.")
            return

        if self.portfolio['holdings'][symbol]['quantity'] < quantity:
            print(f"Not enough {symbol} shares to sell.")
            self.logs.append(f"Not enough {symbol} shares to sell.")
            return

        if price is None:
            price = get_stock_price(symbol, live)
            if price is None:
                print(f"Failed to fetch price for {symbol}. Transaction aborted.")
                self.logs.append(f"Failed to fetch price for {symbol}. Transaction aborted.")
                return

        # Calculate profit/loss
        pl = (price - self.portfolio['holdings'][symbol]['avg_price']) * quantity
        self.portfolio['cash'] += price * quantity
        self.portfolio['holdings'][symbol]['quantity'] -= quantity

        # Record transaction
        self.portfolio['transactions'].append({
            'type': 'SELL',
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'pl': pl
        })
        print(f"Sold {quantity} shares of {symbol} at ₹{price:.2f}")
        self.logs.append(f"Sold {quantity} shares of {symbol} at ₹{price:.2f}")
        print(f"Profit/Loss: ₹{pl:.2f}")
        self.logs.append(f"Profit/Loss: ₹{pl:.2f}")

        # Remove stock if fully sold
        if self.portfolio['holdings'][symbol]['quantity'] == 0:
            del self.portfolio['holdings'][symbol]

    def add_historical_transaction(self, symbol, quantity, transaction_type, timestamp):
        """Add historical transaction with automatic price detection"""
        try:
            # Extract date from timestamp
            transaction_date = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d")
            price = get_historical_price(symbol, transaction_date)

            if price is None:
                print(f"Transaction failed: Could not find price for {symbol} on {transaction_date}")
                self.logs.append(f"Transaction failed: Could not find price for {symbol} on {transaction_date}")
                return

            if transaction_type.upper() == 'BUY':
                # Historical buy transaction
                if self.portfolio['cash'] >= price * quantity:
                    if symbol in self.portfolio['holdings']:
                        # Update average price
                        total_qty = self.portfolio['holdings'][symbol]['quantity'] + quantity
                        total_invested = (self.portfolio['holdings'][symbol]['avg_price'] *
                                          self.portfolio['holdings'][symbol]['quantity'] +
                                          price * quantity)
                        self.portfolio['holdings'][symbol]['quantity'] = total_qty
                        self.portfolio['holdings'][symbol]['avg_price'] = total_invested / total_qty
                    else:
                        self.portfolio['holdings'][symbol] = {
                            'quantity': quantity,
                            'avg_price': price
                        }

                    self.portfolio['cash'] -= price * quantity
                    self.portfolio['transactions'].append({
                        'type': 'BUY',
                        'symbol': symbol,
                        'quantity': quantity,
                        'price': price,
                        'timestamp': timestamp
                    })
                    print(f"Added historical BUY: {quantity} {symbol} @ ₹{price:.2f} on {transaction_date}")
                    self.logs.append(f"Added historical BUY: {quantity} {symbol} @ ₹{price:.2f} on {transaction_date}")
                else:
                    print(f"Historical BUY failed: Insufficient cash on {transaction_date}")
                    self.logs.append(f"Historical BUY failed: Insufficient cash on {transaction_date}")

            elif transaction_type.upper() == 'SELL':
                # Historical sell transaction
                if symbol not in self.portfolio['holdings']:
                    print(f"Historical SELL failed: {symbol} not in portfolio on {transaction_date}")
                    self.logs.append(f"Historical SELL failed: {symbol} not in portfolio on {transaction_date}")
                    return

                if self.portfolio['holdings'][symbol]['quantity'] < quantity:
                    print(f"Historical SELL failed: Not enough {symbol} shares on {transaction_date}")
                    self.logs.append(f"Historical SELL failed: Not enough {symbol} shares on {transaction_date}")
                    return

                pl = (price - self.portfolio['holdings'][symbol]['avg_price']) * quantity
                self.portfolio['cash'] += price * quantity
                self.portfolio['holdings'][symbol]['quantity'] -= quantity

                if self.portfolio['holdings'][symbol]['quantity'] == 0:
                    del self.portfolio['holdings'][symbol]

                self.portfolio['transactions'].append({
                    'type': 'SELL',
                    'symbol': symbol,
                    'quantity': quantity,
                    'price': price,
                    'timestamp': timestamp,
                    'pl': pl
                })
                print(f"Added historical SELL: {quantity} {symbol} @ ₹{price:.2f} on {transaction_date}")
                self.logs.append(f"Added historical SELL: {quantity} {symbol} @ ₹{price:.2f} on {transaction_date}")
                print(f"Historical P/L for this transaction: ₹{pl:.2f}")
                self.logs.append(f"Historical P/L for this transaction: ₹{pl:.2f}")

            else:
                print("Invalid transaction type. Use 'BUY' or 'SELL'")
                self.logs.append("Invalid transaction type. Use 'BUY' or 'SELL'")

        except Exception as e:
            print(f"Error processing historical transaction: {str(e)}")
            self.logs.append(f"Error processing historical transaction: {str(e)}")

    def view_portfolio(self):
        """Display portfolio with current valuations"""
        if not self.portfolio['holdings']:
            print("Portfolio is empty.")
            print("LOGS:", self.logs)
            self.logs.append("Portfolio is empty.")
            return

        total_invested = 0
        total_current = 0
        report = []

        for symbol, data in self.portfolio['holdings'].items():
            current_price = nse.stock_quote(symbol)['priceInfo']['lastPrice']
            invested = data['avg_price'] * data['quantity']
            current_value = current_price * data['quantity']
            pl = current_value - invested

            report.append({
                'Symbol': symbol,
                'Quantity': data['quantity'],
                'Avg Price': data['avg_price'],
                'Current Price': current_price,
                'Invested': invested,
                'Current Value': current_value,
                'P/L': pl
            })

            total_invested += invested
            total_current += current_value

        df = pd.DataFrame(report)
        print("\nPortfolio Summary:")
        self.logs.append("\nPortfolio Summary:")
        print(df.to_string(index=False))
        self.logs.append(df.to_string(index=False))
        print(f"\nTotal Invested: ₹{total_invested:.2f}")
        self.logs.append(f"\nTotal Invested: ₹{total_invested:.2f}")
        print(f"Current Value: ₹{total_current:.2f}")
        self.logs.append(f"Current Value: ₹{total_current:.2f}")
        print(f"Net Profit/Loss: ₹{total_current - total_invested:.2f}")
        self.logs.append(f"Net Profit/Loss: ₹{total_current - total_invested:.2f}")
        print(f"Cash Balance: ₹{self.portfolio['cash']:.2f}")
        self.logs.append(f"Cash Balance: ₹{self.portfolio['cash']:.2f}")
        print("\n\nLOGS:", self.logs)

    def buy_and_hold(self, symbol, initial_investment, start_date):
        price = get_historical_price(symbol, start_date)
        if price:
            quantity = initial_investment // price
            self.add_historical_transaction(symbol, quantity, "BUY", f"{start_date} 09:15:00")

    def momentum_strategy(self, symbol, current_date, lookback_days=14, threshold=0.05):
        # Convert current_date to datetime.date if it's a datetime.datetime object
        if isinstance(current_date, datetime):
            current_date = current_date.date()

        # Calculate start and end dates for historical data
        end_date = current_date
        start_date = end_date - timedelta(days=lookback_days * 2)

        # Fetch historical data
        df = stock_df(symbol, from_date=start_date, to_date=end_date, series="EQ")
        if len(df) < lookback_days:
            return

        # Calculate returns
        df['returns'] = df['CLOSE'].pct_change(lookback_days)
        current_return = df['returns'].iloc[-1]

        if current_return > threshold:
            # Buy signal
            # buy_stock(symbol, 1, live=True)
            self.add_historical_transaction(symbol, 1, "BUY", f"{current_date} 09:15:00")

        elif current_return < -threshold:
            # Sell signal
            if symbol in self.portfolio['holdings']:
                self.add_historical_transaction(symbol, self.portfolio['holdings'][symbol]['quantity'], "SELL",
                                                f"{current_date} 09:15:00")

    def bollinger_bands_strategy(self, symbol, current_date, window=20, num_std=2):
        # Get historical data
        # end_date = date.today()
        end_date = current_date
        start_date = end_date - timedelta(days=window * 2)
        df = stock_df(symbol, from_date=start_date, to_date=end_date, series="EQ")

        if len(df) < window:
            return

        # Calculate indicators
        df['MA'] = df['CLOSE'].rolling(window=window).mean()
        df['STD'] = df['CLOSE'].rolling(window=window).std()
        df['Upper'] = df['MA'] + (df['STD'] * num_std)
        df['Lower'] = df['MA'] - (df['STD'] * num_std)

        current_price = df['CLOSE'].iloc[-1]

        if current_price < df['Lower'].iloc[-1]:
            # Buy signal
            self.add_historical_transaction(symbol, 1, "BUY", f"{current_date} 09:15:00")
        elif current_price > df['Upper'].iloc[-1]:
            # Sell signal
            if symbol in self.portfolio['holdings']:
                self.add_historical_transaction(symbol, self.portfolio['holdings'][symbol]['quantity'], "SELL",
                                                f"{current_date} 09:15:00")

    def moving_average_crossover(self, symbol, start_date, short_window=50, long_window=200):
        # Get historical data
        end_date = start_date
        start_date = end_date - timedelta(days=long_window * 2)
        df = stock_df(symbol, from_date=start_date, to_date=end_date, series="EQ")
        # Debugging: Check DataFrame
        if df.empty or 'CLOSE' not in df.columns:
            print(f"Error: No 'CLOSE' column found or DataFrame is empty for {symbol}")
            return

        if not isinstance(short_window, int) or not isinstance(long_window,
                                                               int) or short_window <= 0 or long_window <= 0:
            print(f"Error: Invalid window sizes -> short_window: {short_window}, long_window: {long_window}")
            return

        if len(df) < long_window:
            print(f"Not enough data for {symbol}. Required: {long_window}, Available: {len(df)}")
            return

        # Ensure 'CLOSE' is numeric
        df['CLOSE'] = pd.to_numeric(df['CLOSE'], errors='coerce')

        # Calculate MAs
        df['SMA50'] = df['CLOSE'].rolling(short_window).mean()
        df['SMA200'] = df['CLOSE'].rolling(long_window).mean()
        # print(df['SMA50'])

        # Ensure enough non-null values exist
        if df[['SMA50', 'SMA200']].isnull().all().iloc[-1]:
            print(f"Insufficient data to compute moving averages for {symbol}")
            return

        # Generate signals
        if df['SMA50'].iloc[-2] < df['SMA200'].iloc[-2] and df['SMA50'].iloc[-1] > df['SMA200'].iloc[-1]:
            print(f"Golden Cross detected: Buying {symbol}")
            self.buy_stock(symbol, 1, live=True)

        elif df['SMA50'].iloc[-2] > df['SMA200'].iloc[-2] and df['SMA50'].iloc[-1] < df['SMA200'].iloc[-1]:
            print(f"Death Cross detected: Selling {symbol}")
            if symbol in self.portfolio['holdings']:
                self.sell_stock(symbol, self.portfolio['holdings'][symbol]['quantity'], live=True)

    def run_backtest(self, strategy, symbol, start_date, end_date):
        # Initialize with proper price history structure
        self.portfolio['price_history'] = {}
        initial_cash = self.portfolio['cash']

        # Convert dates
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d").date()

        current_date = start_date
        while current_date <= end_date:
            if current_date.weekday() < 5:  # Skip weekends
                date_str = current_date.strftime("%Y-%m-%d")
                price = get_historical_price(symbol, date_str)
                if price is not None:
                    # Store as string date to avoid serialization issues
                    self.portfolio['price_history'][date_str] = price
                    strategy(symbol, current_date)
                else:
                    print(f"No price data for {symbol} on {date_str}")
            current_date += timedelta(days=1)

        # Debug: Show collected price history
        print(f"\nCollected {len(self.portfolio['price_history'])} price points")
        print("Sample prices:", dict(list(self.portfolio['price_history'].items())[:3]))

        # Calculate returns
        final_value = self.portfolio['cash']
        for s, h in self.portfolio['holdings'].items():
            last_price = get_historical_price(s, end_date.strftime("%Y-%m-%d"))
            if last_price:
                final_value += last_price * h['quantity']

        self.portfolio['return'] = (final_value - initial_cash) / initial_cash

        # Generate plot
        image_path = self.plot_backtest_results(symbol)
        if image_path:
            if 'performance_images' not in self.portfolio:
                self.portfolio['performance_images'] = []
            self.portfolio['performance_images'].append(image_path)

        return {
            'return': self.portfolio['return'],
            'transactions': self.portfolio['transactions'],
            'graph_path': image_path,
            'price_history': self.portfolio['price_history']  # Return for debugging
        }

    def plot_backtest_results(self, symbol):
        """Plot price history with buy/sell signals"""
        if not self.portfolio.get('price_history'):
            print("DEBUG - No price history in portfolio:", self.portfolio.keys())
            return None

        try:
            # Convert string dates to datetime objects for plotting
            dates = [datetime.strptime(d, "%Y-%m-%d").date()
                     for d in sorted(self.portfolio['price_history'].keys())]
            prices = [self.portfolio['price_history'][d]
                      for d in sorted(self.portfolio['price_history'].keys())]

            plt.figure(figsize=(14, 7))
            plt.plot(dates, prices, label='Price', color='royalblue', linewidth=2)

            # Plot transactions if they exist
            if 'transactions' in self.portfolio:
                for t in self.portfolio['transactions']:
                    try:
                        trans_date = datetime.strptime(t['timestamp'], "%Y-%m-%d %H:%M:%S").date()
                        trans_type = t['type']
                        price = t['price']

                        if trans_type == 'BUY':
                            plt.scatter(trans_date, price, color='limegreen', marker='^',
                                        s=150, edgecolors='black', label='Buy')
                        elif trans_type == 'SELL':
                            plt.scatter(trans_date, price, color='crimson', marker='v',
                                        s=150, edgecolors='black', label='Sell')
                    except Exception as e:
                        print(f"Error plotting transaction: {e}")
                        continue

            plt.title(f"{symbol} Trading Performance")
            plt.xlabel('Date')
            plt.ylabel('Price (₹)')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()

            # Format dates
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gcf().autofmt_xdate()

            # Save image
            os.makedirs('static/graphs', exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = f"static/graphs/{self.name}_{symbol}_{timestamp}.png"
            plt.savefig(image_path)
            plt.close()

            return image_path

        except Exception as e:
            print(f"Error in plot_backtest_results: {e}")
            return None


################################## PART 1 : STRATEGIES; UNCOMMENT THE BELOW PART TO TRY STRATEGIES. SWITCH THE STRATEGY TO ALL AVAILABLE ONES TO CHECK THEM ########################################3


"""

simobject = Simulation("main1",20000)

results = simobject.run_backtest(
    strategy=simobject.bollinger_bands_strategy,
    symbol="TCS",
    start_date="2023-01-01",
    end_date="2023-12-31",
)

print(f"Strategy Return: {results['return']*100:.2f}%")

for transaction in results['transactions']:
    print(transaction)


"""


############################################

class Watchlist:
    def __init__(self, name):
        self.name = name
        self.watchlist = {}

    def add_to_watchlist(self, symbol, notes="added!"):
        """Add a stock to the watchlist."""
        if symbol in self.watchlist:
            print(f"{symbol} is already in the watchlist.")
            return

        price = get_stock_price(symbol)
        if price is not None:
            self.watchlist[symbol] = {'added_on': date.today().strftime('%Y-%m-%d'), 'last_price': price,
                                      'notes': notes}
            print(f"Added {symbol} to watchlist at ₹{price:.2f}.")
        else:
            print(f"Failed to add {symbol} to watchlist.")

    def remove_from_watchlist(self, symbol):
        """Remove a stock from the watchlist."""
        if symbol in self.watchlist:
            del self.watchlist[symbol]
            print(f"Removed {symbol} from watchlist.")
        else:
            print(f"{symbol} is not in the watchlist.")

    def view_watchlist(self):
        """Display the current watchlist with updated prices."""
        if not self.watchlist:
            print("Watchlist is empty.")
            return

        report = []
        for symbol, data in self.watchlist.items():
            current_price = get_stock_price(symbol)
            price_change = current_price - data['last_price'] if current_price else None
            report.append({
                'Symbol': symbol,
                'Added On': data['added_on'],
                'Initial Price': data['last_price'],
                'Current Price': current_price if current_price else "N/A",
                'Change': f"₹{price_change:.2f}" if price_change else "N/A"
            })

        df = pd.DataFrame(report)
        print("\nWatchlist Summary:")
        print(df.to_string(index=False))


"""
wtl1 = Watchlist("watchlist1")
wtl1.add_to_watchlist("RELIANCE")
wtl1.add_to_watchlist("TCS")
wtl1.view_watchlist()
wtl1.remove_from_watchlist("RELIANCE")
wtl1.view_watchlist()
"""


def generate_advice_sheet(symbol):
    """
    Generate an advice sheet for a given stock.

    Parameters:
        symbol (str): Stock symbol (e.g., "RELIANCE").
    """
    # Fetch current stock information
    current_price = get_stock_price(symbol)
    if current_price is None:
        print(f"Failed to fetch data for {symbol}.")
        return

    # Fetch historical data for 1-year return calculation
    end_date = date.today()
    start_date = end_date - timedelta(days=365)
    df = stock_df(symbol, from_date=start_date, to_date=end_date, series="EQ")
    if df.empty:
        print(f"No historical data found for {symbol}.")
        return

    # Calculate 1-year return
    initial_price = df['CLOSE'].iloc[0]
    final_price = df['CLOSE'].iloc[-1]
    one_year_return = ((final_price - initial_price) / initial_price) * 100

    # Generate recommendations
    buy_and_hold_recommendation = evaluate_buy_and_hold(symbol, df)
    momentum_recommendation = evaluate_momentum(symbol, df)
    bollinger_recommendation = evaluate_bollinger_bands(symbol, df)
    ma_crossover_recommendation = evaluate_moving_average_crossover(symbol, df)
    arima_prediction = predict_future_price(symbol, days_ahead=7)
    # lstm_prediction = lstm_predictor(symbol, lookback=60)

    # Print advice sheet
    print("\n" + "=" * 50)
    print(f"Advice Sheet for {symbol}")
    print("=" * 50)
    print(f"\nCurrent Price: ₹{current_price:.2f}")
    print(f"1-Year Return: {one_year_return:.2f}%")

    print("\n=== Buy-and-Hold Recommendation ===")
    print(buy_and_hold_recommendation)

    print("\n=== Momentum Strategy Recommendation ===")
    print(momentum_recommendation)

    print("\n=== Bollinger Bands Strategy Recommendation ===")
    print(bollinger_recommendation)

    print("\n=== Moving Average Crossover Recommendation ===")
    print(ma_crossover_recommendation)

    print("\n=== Price Predictions ===")
    print(
        f"ARIMA Predicted Price (7 days): ₹{arima_prediction:.2f}" if arima_prediction else "ARIMA prediction unavailable.")
    # print(
    #    f"LSTM Predicted Price (next day): ₹{lstm_prediction:.2f}" if lstm_prediction else "LSTM prediction unavailable.")

    print("\n=== Final Recommendation ===")
    final_recommendation = generate_final_recommendation(
        buy_and_hold_recommendation,
        momentum_recommendation,
        bollinger_recommendation,
        ma_crossover_recommendation,
        arima_prediction,
        # lstm_prediction
    )
    print(final_recommendation)
    print("=" * 50 + "\n")


def evaluate_buy_and_hold(symbol, df):
    """
    Evaluate Buy-and-Hold strategy.

    Parameters:
        symbol (str): Stock symbol.
        df (pd.DataFrame): Historical data.
    """
    # Calculate 1-year return
    initial_price = df['CLOSE'].iloc[0]
    final_price = df['CLOSE'].iloc[-1]
    one_year_return = ((final_price - initial_price) / initial_price) * 100

    if one_year_return > 10:  # Threshold for positive return
        return f"Buy and Hold: Recommended (1-Year Return: {one_year_return:.2f}%)."
    else:
        return f"Buy and Hold: Not Recommended (1-Year Return: {one_year_return:.2f}%)."


def evaluate_momentum(symbol, df, lookback_days=14, threshold=0.05):
    """
    Evaluate Momentum strategy.

    Parameters:
        symbol (str): Stock symbol.
        df (pd.DataFrame): Historical data.
        lookback_days (int): Lookback period for momentum calculation.
        threshold (float): Threshold for buy/sell signals.
    """
    df['returns'] = df['CLOSE'].pct_change(lookback_days)
    current_return = df['returns'].iloc[-1]

    if current_return > threshold:
        return f"Momentum: Buy (Recent Return: {current_return * 100:.2f}%)."
    elif current_return < -threshold:
        return f"Momentum: Sell (Recent Return: {current_return * 100:.2f}%)."
    else:
        return "Momentum: Hold (No significant momentum detected)."


def evaluate_bollinger_bands(symbol, df, window=20, num_std=2):
    """
    Evaluate Bollinger Bands strategy.

    Parameters:
        symbol (str): Stock symbol.
        df (pd.DataFrame): Historical data.
        window (int): Rolling window size.
        num_std (int): Number of standard deviations for bands.
    """
    df['MA'] = df['CLOSE'].rolling(window=window).mean()
    df['STD'] = df['CLOSE'].rolling(window=window).std()
    df['Upper'] = df['MA'] + (df['STD'] * num_std)
    df['Lower'] = df['MA'] - (df['STD'] * num_std)

    current_price = df['CLOSE'].iloc[-1]

    if current_price < df['Lower'].iloc[-1]:
        return "Bollinger Bands: Buy (Stock is oversold)."
    elif current_price > df['Upper'].iloc[-1]:
        return "Bollinger Bands: Sell (Stock is overbought)."
    else:
        return "Bollinger Bands: Hold (Stock is within bands)."


def evaluate_moving_average_crossover(symbol, df, short_window=50, long_window=200):
    """
    Evaluate Moving Average Crossover strategy.

    Parameters:
        symbol (str): Stock symbol.
        df (pd.DataFrame): Historical data.
        short_window (int): Short-term moving average window.
        long_window (int): Long-term moving average window.
    """
    df['SMA50'] = df['CLOSE'].rolling(short_window).mean()
    df['SMA200'] = df['CLOSE'].rolling(long_window).mean()

    if df['SMA50'].iloc[-2] < df['SMA200'].iloc[-2] and df['SMA50'].iloc[-1] > df['SMA200'].iloc[-1]:
        return "Moving Average Crossover: Buy (Golden Cross detected)."
    elif df['SMA50'].iloc[-2] > df['SMA200'].iloc[-2] and df['SMA50'].iloc[-1] < df['SMA200'].iloc[-1]:
        return "Moving Average Crossover: Sell (Death Cross detected)."
    else:
        return "Moving Average Crossover: Hold (No crossover detected)."


def generate_final_recommendation(*recommendations):
    """
    Generate a final recommendation based on all strategies.

    Parameters:
        recommendations: List of recommendations from all strategies.
    """
    buy_count = sum(1 for rec in recommendations if "Buy" in rec)
    sell_count = sum(1 for rec in recommendations if "Sell" in rec)

    if buy_count > sell_count:
        return "Final Recommendation: Buy (Majority of strategies recommend buying)."
    elif sell_count > buy_count:
        return "Final Recommendation: Sell (Majority of strategies recommend selling)."
    else:
        return "Final Recommendation: Hold (No clear consensus among strategies)."


def predict_future_price(symbol, days_ahead=7):
    # Get historical data
    end_date = date.today()
    start_date = end_date - timedelta(days=365)
    df = stock_df(symbol, from_date=start_date, to_date=end_date, series="EQ")

    if len(df) < 30:
        return None

    # Fit ARIMA model
    model = ARIMA(df['CLOSE'], order=(5, 1, 0))
    model_fit = model.fit()

    # Make prediction
    forecast = model_fit.forecast(steps=days_ahead)

    # Check if forecast is empty or not
    if forecast.size > 0:
        if isinstance(forecast, np.ndarray):  # If forecast is a numpy array
            return forecast[-1]
        elif isinstance(forecast, pd.Series):  # If forecast is a pandas Series
            return forecast.iloc[-1]
        else:
            return forecast  # If it's neither, just return the whole forecast
    else:
        return None


"""
generate_advice_sheet("SWIGGY")
"""

import sqlite3
import json
from datetime import datetime


def save_graph_image(fig, simulation_id, graph_name):
    """Save matplotlib figure and return path"""
    os.makedirs('static/graphs', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{simulation_id}_{graph_name}_{timestamp}.png"
    path = f"static/graphs/{filename}"
    fig.savefig(path)
    plt.close(fig)
    return path


def serialize_simulation(simulation):
    """Convert simulation to JSON-serializable dict"""
    # Convert price_history dates to strings if they exist
    portfolio_data = simulation.portfolio.copy()
    if 'price_history' in portfolio_data:
        portfolio_data['price_history'] = {
            k.strftime("%Y-%m-%d") if isinstance(k, date) else k: v
            for k, v in portfolio_data['price_history'].items()
        }

    data = {
        'name': simulation.name,
        'timestamp': simulation.timestamp,
        'portfolio': portfolio_data,
        'logs': simulation.logs,
        'images': simulation.images,
        'performance_images': simulation.portfolio.get('performance_images', [])
    }
    return json.loads(json.dumps(data, cls=PortfolioEncoder))


# --- Database Setup ---
def create_database():
    conn = sqlite3.connect('trading_system.db')
    c = conn.cursor()

    # Users Table
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 username TEXT UNIQUE NOT NULL,
                 password TEXT NOT NULL,
                 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')

    # Portfolios Table
    c.execute('''CREATE TABLE IF NOT EXISTS portfolios (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 user_id INTEGER NOT NULL,
                 name TEXT NOT NULL,
                 data TEXT NOT NULL,  
                 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                 FOREIGN KEY(user_id) REFERENCES users(id))''')

    # Watchlists Table
    c.execute('''CREATE TABLE IF NOT EXISTS watchlists (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 user_id INTEGER NOT NULL,
                 name TEXT NOT NULL,
                 symbols TEXT NOT NULL,  
                 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                 FOREIGN KEY(user_id) REFERENCES users(id))''')



    # NEW TABLE for simulation images
    c.execute('''CREATE TABLE IF NOT EXISTS simulation_images (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 simulation_id INTEGER NOT NULL,
                 image_path TEXT NOT NULL,
                 image_type TEXT NOT NULL,
                 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                 FOREIGN KEY(simulation_id) REFERENCES portfolios(id))''')

    c.execute('''CREATE TABLE IF NOT EXISTS strategies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    portfolio_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    strategy_type TEXT NOT NULL CHECK(strategy_type IN ('MOMENTUM', 'BOLLINGER', 'MACROSS')),
                    parameters TEXT NOT NULL,
                    is_active BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_executed TIMESTAMP,
                    FOREIGN KEY(user_id) REFERENCES users(id),
                    FOREIGN KEY(portfolio_id) REFERENCES portfolios(id))''')

    # Execution log table
    c.execute('''CREATE TABLE IF NOT EXISTS strategy_executions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_id INTEGER NOT NULL,
                    execution_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    action TEXT NOT NULL,
                    quantity INTEGER,
                    price REAL,
                    FOREIGN KEY(strategy_id) REFERENCES strategies(id))''')

    conn.commit()
    conn.close()


create_database()
########## STRATEGIES ###################

class StrategyManager:
    def __init__(self):
        self.conn = sqlite3.connect('trading_system.db')
        self.conn.row_factory = sqlite3.Row

    def create_strategy(self, user_id, portfolio_id, name, symbol, strategy_type, parameters):
        """Create a new trading strategy"""
        try:
            # Validate portfolio belongs to user
            c = self.conn.cursor()
            c.execute('SELECT id FROM portfolios WHERE id=? AND user_id=?',
                      (portfolio_id, user_id))
            if not c.fetchone():
                return False, "Portfolio not found or access denied"

            # Insert new strategy
            c.execute('''INSERT INTO strategies 
                        (user_id, portfolio_id, name, symbol, strategy_type, parameters)
                        VALUES (?, ?, ?, ?, ?, ?)''',
                      (user_id, portfolio_id, name, symbol.upper(),
                       strategy_type.upper(), json.dumps(parameters)))
            self.conn.commit()
            return True, "Strategy created successfully"
        except Exception as e:
            return False, f"Error creating strategy: {str(e)}"

    def delete_strategy(self, user_id, strategy_id):
        """Delete a strategy if it belongs to the user"""
        try:
            c = self.conn.cursor()
            c.execute('DELETE FROM strategies WHERE id=? AND user_id=?',
                      (strategy_id, user_id))
            self.conn.commit()
            return c.rowcount > 0, "Deleted" if c.rowcount else "Strategy not found"
        except Exception as e:
            return False, f"Error deleting strategy: {str(e)}"

    def list_strategies(self, user_id):
        """List all strategies for a user with portfolio info"""
        try:
            c = self.conn.cursor()
            c.execute('''SELECT s.id, s.name, s.symbol, s.strategy_type, s.is_active,
                         p.name as portfolio_name, s.last_executed
                      FROM strategies s
                      JOIN portfolios p ON s.portfolio_id = p.id
                      WHERE s.user_id=?''', (user_id,))
            return True, [dict(row) for row in c.fetchall()]
        except Exception as e:
            return False, f"Error listing strategies: {str(e)}"

    def list_strategies_json(self, user_id):
        """List all strategies for a user with portfolio info"""
        try:
            c = self.conn.cursor()
            c.execute('''SELECT s.id, s.name, s.symbol, s.strategy_type, s.is_active,
                         p.name as portfolio_name, s.last_executed
                      FROM strategies s
                      JOIN portfolios p ON s.portfolio_id = p.id
                      WHERE s.user_id=?''', (user_id,))
            return {"hasStrategies":"True","data": [dict(row) for row in c.fetchall()] }
        except Exception as e:
            return {"hasStrategies":'False', "data":[f"Error listing strategies: {str(e)}",]}



    def toggle_strategy(self, user_id, strategy_id, active):
        """Enable/disable a strategy"""
        try:
            c = self.conn.cursor()
            c.execute('''UPDATE strategies SET is_active=?
                      WHERE id=? AND user_id=?''',
                      (active, strategy_id, user_id))
            self.conn.commit()
            return c.rowcount > 0, "Updated" if c.rowcount else "Strategy not found"
        except Exception as e:
            return False, f"Error updating strategy: {str(e)}"

#####################################################

# --- User Authentication ---
def register_user(username, password):
    conn = sqlite3.connect('trading_system.db')
    try:
        conn.execute("INSERT INTO users (username, password) VALUES (?, ?)",
                     (username, password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        print("Username already exists")
        return False
    finally:
        conn.close()


def authenticate_user(username, password):
    conn = sqlite3.connect('trading_system.db')
    c = conn.cursor()
    c.execute("SELECT id FROM users WHERE username=? AND password=?", (username, password))
    user = c.fetchone()
    conn.close()
    return user[0] if user else None


def save_portfolio(user_id, portfolio_obj):
    conn = sqlite3.connect('trading_system.db')
    try:
        # Serialize the portfolio
        json_data = json.dumps(serialize_simulation(portfolio_obj), cls=PortfolioEncoder)

        if hasattr(portfolio_obj, 'db_id'):
            # Update existing portfolio
            conn.execute('''UPDATE portfolios SET name=?, data=? WHERE id=? AND user_id=?''',
                         (portfolio_obj.name, json_data, portfolio_obj.db_id, user_id))
        else:
            # Insert new portfolio
            cursor = conn.cursor()
            cursor.execute('''INSERT INTO portfolios (user_id, name, data) VALUES (?, ?, ?)''',
                           (user_id, portfolio_obj.name, json_data))
            portfolio_obj.db_id = cursor.lastrowid

            # Save images if this is a new portfolio
            for img_path in portfolio_obj.images + portfolio_obj.portfolio.get('performance_images', []):
                img_type = 'performance' if img_path in portfolio_obj.portfolio.get('performance_images',
                                                                                    []) else 'graph'
                conn.execute('''INSERT INTO simulation_images (simulation_id, image_path, image_type)
                              VALUES (?, ?, ?)''', (portfolio_obj.db_id, img_path, img_type))

        conn.commit()
        return True
    except Exception as e:
        print(f"Error saving portfolio: {str(e)}")
        return False
    finally:
        conn.close()


# Update your load_portfolio function
def load_portfolio(user_id, portfolio_id):
    conn = sqlite3.connect('trading_system.db')
    try:
        c = conn.cursor()

        # Load portfolio data
        c.execute('''SELECT id, name, data FROM portfolios 
                   WHERE id=? AND user_id=?''',
                  (portfolio_id, user_id))
        result = c.fetchone()

        if not result:
            return None

        db_id, name, data_str = result
        data = json.loads(data_str)

        # Reconstruct portfolio
        portfolio = Simulation(name, data['portfolio']['cash'])
        portfolio.db_id = db_id
        portfolio.portfolio.update(data['portfolio'])
        portfolio.logs = data.get('logs', [])
        portfolio.images = data.get('images', [])

        # Load associated images
        c.execute('''SELECT image_path, image_type FROM simulation_images
                   WHERE simulation_id=?''', (db_id,))
        images = c.fetchall()

        for img_path, img_type in images:
            if img_type == 'performance':
                if 'performance_images' not in portfolio.portfolio:
                    portfolio.portfolio['performance_images'] = []
                portfolio.portfolio['performance_images'].append(img_path)
            else:
                portfolio.images.append(img_path)

        return portfolio
    except Exception as e:
        print(f"Error loading portfolio: {str(e)}")
        return None
    finally:
        conn.close()


# --- Watchlist Storage ---
def save_watchlist(user_id, watchlist_obj):
    conn = sqlite3.connect('trading_system.db')
    try:
        conn.execute('''INSERT INTO watchlists (user_id, name, symbols)
                        VALUES (?, ?, ?)''',
                     (user_id, watchlist_obj.name, json.dumps(list(watchlist_obj.watchlist.keys()))))
        conn.commit()
        return True
    except Exception as e:
        print(f"Error saving watchlist: {e}")
        return False
    finally:
        conn.close()


def load_watchlist(user_id, watchlist_id):
    conn = sqlite3.connect('trading_system.db')
    try:
        c = conn.cursor()
        c.execute('''SELECT name, symbols FROM watchlists 
                     WHERE id=? AND user_id=?''', (watchlist_id, user_id))
        result = c.fetchone()

        if not result:
            return None

        watchlist = Watchlist(result[0])
        symbols = json.loads(result[1])
        for symbol in symbols:
            watchlist.add_to_watchlist(symbol)

        return watchlist
    except Exception as e:
        print(f"Error loading watchlist: {e}")
        return None
    finally:
        conn.close()


def get_user_portfolios(user_id):
    """Get ALL portfolios for a user in a nested structure"""
    conn = sqlite3.connect('trading_system.db')
    try:
        c = conn.cursor()
        c.execute('''SELECT id, name, data, created_at FROM portfolios
                   WHERE user_id=? ORDER BY created_at DESC''',
                  (user_id,))

        portfolios = []
        for row in c.fetchall():
            try:
                portfolio_data = json.loads(row[2])

                portfolio = {
                    'id': row[0],
                    'name': row[1],
                    'created_at': row[3],
                    'type': 'portfolio',
                    'data': portfolio_data,  # Full portfolio data
                    'cash': portfolio_data.get('portfolio', {}).get('cash', 0),
                    'holdings_count': len(portfolio_data.get('portfolio', {}).get('holdings', {})),
                    'transactions_count': len(portfolio_data.get('portfolio', {}).get('transactions', []))
                }

                portfolios.append(portfolio)

            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error processing portfolio {row[0]}: {str(e)}")
                portfolios.append({
                    'id': row[0],
                    'name': row[1],
                    'created_at': row[3],
                    'type': 'portfolio',
                    'error': f"Could not load portfolio data: {str(e)}"
                })

        return {
            'user_id': user_id,
            'count': len(portfolios),
            'portfolios': portfolios  # Nested under 'portfolios' key
        }
    finally:
        conn.close()


def get_user_watchlists(user_id):
    """Get ALL watchlists for a user in a nested structure"""
    conn = sqlite3.connect('trading_system.db')
    try:
        c = conn.cursor()
        c.execute('''SELECT id, name, symbols, created_at FROM watchlists
                   WHERE user_id=? ORDER BY created_at DESC''',
                  (user_id,))

        watchlists = []
        for row in c.fetchall():
            try:
                symbols = json.loads(row[2])

                watchlist = {
                    'id': row[0],
                    'name': row[1],
                    'created_at': row[3],
                    'type': 'watchlist',
                    'symbols': symbols,
                    'count': len(symbols)
                }

                watchlists.append(watchlist)

            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error processing watchlist {row[0]}: {str(e)}")
                watchlists.append({
                    'id': row[0],
                    'name': row[1],
                    'created_at': row[3],
                    'type': 'watchlist',
                    'error': f"Could not load watchlist data: {str(e)}"
                })

        return {
            'user_id': user_id,
            'count': len(watchlists),
            'watchlists': watchlists  # Nested under 'watchlists' key
        }
    finally:
        conn.close()


def get_portfolio_details(portfolio_id):
    """Get full details of a specific portfolio with additional metadata"""
    conn = sqlite3.connect('trading_system.db')
    try:
        c = conn.cursor()

        # Get basic portfolio info
        c.execute('''SELECT id, user_id, name, data, created_at 
                   FROM portfolios WHERE id=?''',
                  (portfolio_id,))
        result = c.fetchone()
        if not result:
            return None

        portfolio_id, user_id, name, data_str, created_at = result
        data = json.loads(data_str)

        # Get associated images
        c.execute('''SELECT image_path, image_type FROM simulation_images
                   WHERE simulation_id=? ORDER BY created_at DESC''',
                  (portfolio_id,))
        images = [{'path': row[0], 'type': row[1]} for row in c.fetchall()]

        # Structure the response
        response = {
            'id': portfolio_id,
            'user_id': user_id,
            'name': name,
            'created_at': created_at,
            'type': 'portfolio',
            'details': {
                'cash': data['portfolio']['cash'],
                'holdings': data['portfolio']['holdings'],
                'transactions': data['portfolio']['transactions'],
                'performance_images': data['portfolio'].get('performance_images', []),
                'logs': data.get('logs', []),
                'return': data['portfolio'].get('return', 0)
            },
            'images': images,
            'raw_data': data  # The complete stored data
        }

        return response
    finally:
        conn.close()


def get_watchlist_details(watchlist_id):
    """Get full details of a specific watchlist"""
    conn = sqlite3.connect('trading_system.db')
    try:
        c = conn.cursor()
        c.execute('''SELECT id, user_id, name, symbols, created_at 
                   FROM watchlists WHERE id=?''',
                  (watchlist_id,))
        result = c.fetchone()
        if not result:
            return None

        watchlist_id, user_id, name, symbols_str, created_at = result
        symbols = json.loads(symbols_str)

        # Get current prices for all symbols
        symbol_details = []
        for symbol in symbols:
            price = get_stock_price(symbol)
            symbol_details.append({
                'symbol': symbol,
                'current_price': price
            })

        response = {
            'id': watchlist_id,
            'user_id': user_id,
            'name': name,
            'created_at': created_at,
            'type': 'watchlist',
            'details': {
                'symbols': symbol_details,
                'count': len(symbols)
            },
            'raw_data': {
                'name': name,
                'symbols': symbols
            }
        }

        return response
    finally:
        conn.close()


def get_portfolio_images(portfolio_id):
    """Get all images associated with a portfolio"""
    conn = sqlite3.connect('trading_system.db')
    try:
        c = conn.cursor()
        c.execute('''SELECT id, image_path, image_type, created_at 
                   FROM simulation_images
                   WHERE simulation_id=? ORDER BY created_at DESC''',
                  (portfolio_id,))
        images = []
        for row in c.fetchall():
            images.append({
                'id': row[0],
                'path': row[1],
                'type': row[2],
                'created_at': row[3]
            })
        return images
    finally:
        conn.close()

    # Create user
    # register_user("john_doe", "secure123")

    # Authenticate
    # user_id = authenticate_user("kau", "secure123")

    # if user_id:
    # Create and save portfolio
    """
    my_portfolio = Simulation("Tech Portfolio", 100000)
    my_portfolio.buy_stock("TCS", 10)
    my_portfolio.buy_stock("INFY", 5)
    save_portfolio(user_id, my_portfolio)

    # Create and save watchlist
    tech_watchlist = Watchlist("Tech Stocks")
    tech_watchlist.add_to_watchlist("TCS", "Large cap")
    tech_watchlist.add_to_watchlist("INFY", "Mid cap")
    save_watchlist(user_id, tech_watchlist)

    simobject = Simulation("main1", 20000)

    results = simobject.run_backtest(
        strategy=simobject.bollinger_bands_strategy,
        symbol="TCS",
        start_date="2023-01-01",
        end_date="2023-12-31",
    )

    print(f"Strategy Return: {results['return'] * 100:.2f}%")

    for transaction in results['transactions']:
        print(transaction)

        simobject = Simulation("main1", 20000)

    results = simobject.run_backtest(
        strategy=simobject.bollinger_bands_strategy,
        symbol="TCS",
        start_date="2023-01-01",
        end_date="2023-12-31",
    )

    print(f"Strategy Return: {results['return'] * 100:.2f}%")

    for transaction in results['transactions']:
        print(transaction)
    save_portfolio(user_id, simobject )

    results = loaded_portfolio.run_backtest(
        strategy=loaded_portfolio.bollinger_bands_strategy,
        symbol="WIPRO",
        start_date="2023-01-01",
        end_date="2023-12-31",
    )

    print(f"Strategy Return: {results['return'] * 100:.2f}%")

    for transaction in results['transactions']:
        print(transaction)
    save_portfolio(user_id, loaded_portfolio)



    """


"""
    # Load and display portfolio
    loaded_portfolio = load_portfolio(user_id, 3)


if loaded_portfolio:
        loaded_portfolio.view_portfolio()

    # Load and display watchlist
loaded_watchlist = load_watchlist(user_id, 1)
if loaded_watchlist:
        loaded_watchlist.view_watchlist()


"""

"""
print(get_portfolio_images(14))
# Get all portfolios
portfolios_response = get_user_portfolios(4)
print(json.dumps(portfolios_response, indent=2))
print(f"User has {portfolios_response['count']} portfolios:")
for portfolio in portfolios_response['portfolios']:
    print(f"- {portfolio['name']} (ID: {portfolio['id']}) with {portfolio['holdings_count']} holdings")

# Get all watchlists
watchlists_response = get_user_watchlists(1)
print(json.dumps(watchlists_response, indent=2))
print(f"\nUser has {watchlists_response['count']} watchlists:")
for watchlist in watchlists_response['watchlists']:
    print(f"- {watchlist['name']} (ID: {watchlist['id']}) with {watchlist['count']} symbols")



portfolios_response = get_user_portfolios(6)
print(json.dumps(portfolios_response, indent=2))
sm=StrategyManager()
list_strategies1 = sm.list_strategies_json(2)
print(list_strategies1)

"""
