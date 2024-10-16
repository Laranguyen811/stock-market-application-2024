
class PerformanceCalculator:
    '''
    Class of performance calculators.
    '''
    def __init__(self,performance_calculations):
        self.performance_calculations = {}
        self.prices = []

    def calculate_sma(self,prices:list,period:int):
        ''' Takes prices and period and calculates the moving average periods.
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







