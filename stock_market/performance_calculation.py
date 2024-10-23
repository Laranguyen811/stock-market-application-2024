
class PerformanceCalculator:
    '''
        Class of performance calculators.
    '''
    def __init__(self,performance_calculations):
        self.performance_calculations = {}  # Adding a private attribute performance_calculations
        self.prices = []

    def calculate_mas(self,prices:list,period:int):
        '''
        Takes prices and period and calculates the moving average periods.
        Inputs:
            prices(list): A list of prices of stocks.
            period(int): An integer of period
        Returns:
            float: A float number of the simple moving average
            list: A list of the exponential moving averages
        '''

        simple_ma = sum(self.prices[:period])/period  # Calculating the simple moving average
        multiplier = 2 / (period + 1)  # 2 is the smoothing, the higher the smoothing is, the more influence the recent observations have on exponential moving average
        exponential_ma = [simple_ma*(multiplier)]
        for price in self.prices[:period]:
            exponential_ma.append((price - exponential_ma[-1]) * multiplier + exponential_ma[-1])
        return simple_ma, exponential_ma
    def calculate_rsi(self,prices: list,period=14):
        ''' Takes prices and the 14-day period and returns the Relative Strength Index (RSI) thresholds,a momentum oscillator measuring the speed and change of price movements.
        Inputs:
            prices(list): A list of prices of stocks.
            period(int): An integer of the time period.
        Returns:
             float: A float number of the Relative Strength Index (RSI) threshold.
        '''

        deltas = [self.prices[i] - self.prices[i-1] for i in range(1,len(prices))] # Calculating the movements
        gains = [delta if delta > 0 else 0 for delta in deltas]  # Assigning upward movements to gains as a list
        losses = [-delta if delta < 0 else 0 for delta in deltas]  # Assigning downward movements to losses as a list
        average_gain = sum(gains)/len(gains)  # Calculating the average upward movement
        average_losses = sum(losses)/len(losses)  # Calculating the average downward movement
        rs = average_gain/average_losses  # Calculating the average strength
        rsi = 100 - (100/(1 + rs))  # Calculating the RSI
        return rsi

    def calculate_bollinger_bands(self,prices:list, period=20, num_std_dev=2):
        '''
        Takes a list of prices and calculates the Bollinger Bands.
        Inputs:
            prices(list): A list of stock prices
            period(int): An integer of the duration of the period. Defaults to 20.
            num_std_dev(int): An integer of the number of standard deviations. Defaults to 2.
        Returns:
            float: A float number of lower bollinger band
            float: A float number of upper bollinger band
            float: a float number of simple movement average
        '''
        sma = self.calculate_mas(prices,period)[0]  # Calculating simple moving average
        std_dev = (sum([(price - sma) ** 2 for price in prices[-period:]]) / period) ** 0.5  # Calculating the standard deviation of the period
        assert isinstance(std_dev, float)  # Adding an assertion to ensure standard deviation is a float number
        upper_band = sma + (num_std_dev * std_dev)  # Calculating the upper bollinger band
        lower_band = sma - (num_std_dev * std_dev)  # Calculating the lower bollinger band
        return upper_band, lower_band, sma

    def calculate_stop_loss(self,entry_price,stop_loss_percentage):
        '''
        Takes an entry price and a stop loss percentage and calculates stop loss (A stop loss is a limit set for a trader. If the price is above the limit, the trade will automatically stop).
        Inputs:
            entry_price(float): A float number of entry price.
            stop_loss_percentage(float): A float number of stop loss percentage
        Returns:
            float: A float number of stop loss
        '''
        return (1- stop_loss_percentage) * entry_price

    def calculate_take_profit(self, entry_price, take_profit_percentage):
        """ Takes an entry price and a take profit percentage and calculates take profit( A take profit order is used by a trader to close a position when a price of a security reaches a desired profit).
        Inputs:
            entry_price(float): A float number of an entry price.
            take_profit_percentage(float): A float number of a take profit percentage.
        Returns:
            float: A float number of take profit:
        """
        return (1 + take_profit_percentage) * entry_price






