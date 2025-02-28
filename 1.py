from jugaad_data.nse import NSELive, stock_df
from datetime import datetime, date, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

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
    def __init__(self,name,cash):
        self.name=name
        self.timestamp=datetime.now()
        self.logs=[]
        self.portfolio = {
            'cash': cash,  # Initial cash balance
            'holdings': {},  # {symbol: {'quantity': qty, 'avg_price': price}}
            'transactions': []  # List of transactions
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

    def buy_and_hold(self,symbol, initial_investment, start_date):
        price = get_historical_price(symbol, start_date)
        if price:
            quantity = initial_investment // price
            self.add_historical_transaction(symbol, quantity, "BUY", f"{start_date} 09:15:00")

    def momentum_strategy(self,symbol, current_date, lookback_days=14, threshold=0.05):
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

    def bollinger_bands_strategy(self,symbol, current_date, window=20, num_std=2):
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

    def moving_average_crossover(self,symbol, start_date, short_window=50, long_window=200):
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

    def plot_backtest_results(self,symbol):
        """Plot price history with buy/sell signals"""
        if not self.portfolio['price_history']:
            print("No price history to plot")
            return

        # Prepare data
        dates = sorted(self.portfolio['price_history'].keys())
        prices = [self.portfolio['price_history'][date] for date in dates]

        # Convert dates to matplotlib format
        mpl_dates = [mdates.date2num(date) for date in dates]

        plt.figure(figsize=(14, 7))

        # Plot price history
        plt.plot(dates, prices, label='Price', color='royalblue', linewidth=2)

        # Plot buy/sell signals
        for transaction in self.portfolio['transactions']:
            trans_date = datetime.strptime(transaction['timestamp'], "%Y-%m-%d %H:%M:%S").date()
            trans_type = transaction['type']
            price = transaction['price']

            if trans_type == 'BUY':
                plt.scatter(trans_date, price, color='limegreen', marker='^', s=150,
                            edgecolors='black', zorder=3, label='Buy Signal')
            elif trans_type == 'SELL':
                plt.scatter(trans_date, price, color='crimson', marker='v', s=150,
                            edgecolors='black', zorder=3, label='Sell Signal')

                # Add P/L annotation if available
                if 'pl' in transaction:
                    pl = transaction['pl']
                    plt.annotate(f'₹{pl:.2f}\nP/L', (trans_date, price),
                                 textcoords="offset points", xytext=(0, 15),
                                 ha='center', fontsize=9, color='darkred')

        # Formatting
        plt.title(f"{symbol} Trading Performance\nStrategy Return: {self.portfolio['return'] * 100:.2f}%",
                  fontsize=14, pad=20)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price (₹)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)

        # Handle legend duplicates
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='upper left')

        # Format x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        plt.gcf().autofmt_xdate()

        plt.tight_layout()
        plt.show()

    def run_backtest(self,strategy, symbol, start_date, end_date):
        # Initialize portfolio
        initial_cash=self.portfolio["cash"]
        self.portfolio['price_history']= {}  # Added to store price history

        # Convert start_date and end_date to datetime.date objects
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d").date()

        # Simulate daily trading
        current_date = start_date
        while current_date <= end_date:
            # Skip weekends
            if current_date.weekday() < 5:
                # Get daily price and store in price history
                price = get_historical_price(symbol, current_date.strftime("%Y-%m-%d"))
                if price:
                    self.portfolio['price_history'][current_date] = price

                # Execute strategy
                strategy(symbol, current_date)

            current_date += timedelta(days=1)

        # Calculate performance metrics
        final_value = self.portfolio['cash'] + sum(
            get_historical_price(s, current_date.strftime("%Y-%m-%d")) * h['quantity']
            for s, h in self.portfolio['holdings'].items()
        )

        # Add return to portfolio for plotting
        self.portfolio['return'] = (final_value - initial_cash) / initial_cash

        # Generate graph
        self.plot_backtest_results(symbol)

        return {
            'return': self.portfolio['return'],
            'transactions': self.portfolio['transactions'],
            'final_portfolio': self.portfolio
        }



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
        self.name=name
        self.watchlist={}

    def add_to_watchlist(self,symbol):
        """Add a stock to the watchlist."""
        if symbol in self.watchlist:
            print(f"{symbol} is already in the watchlist.")
            return

        price = get_stock_price(symbol)
        if price is not None:
            self.watchlist[symbol] = {'added_on': date.today().strftime('%Y-%m-%d'), 'last_price': price}
            print(f"Added {symbol} to watchlist at ₹{price:.2f}.")
        else:
            print(f"Failed to add {symbol} to watchlist.")

    def remove_from_watchlist(self,symbol):
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
    #lstm_prediction = lstm_predictor(symbol, lookback=60)

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
    #print(
    #    f"LSTM Predicted Price (next day): ₹{lstm_prediction:.2f}" if lstm_prediction else "LSTM prediction unavailable.")

    print("\n=== Final Recommendation ===")
    final_recommendation = generate_final_recommendation(
        buy_and_hold_recommendation,
        momentum_recommendation,
        bollinger_recommendation,
        ma_crossover_recommendation,
        arima_prediction,
        #lstm_prediction
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
