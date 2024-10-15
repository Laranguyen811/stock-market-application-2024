class PerformanceCalculator:
    '''
    Class of performance calculators.
    '''
    def __init__(self,performance_calculator):
        self.performance_calculator = performance_calculator

    def calculate_sma(self,prices,period):
        ''' Takes prices and period and calculates the moving average periods.
        Inputs:
            prices(list): A list of prices of stocks.
            period(int): An integer of period
        Returns:
            float: A float number of the simple moving average
            float: A float number of the exponential moving average
        '''

        simple_ma = sum(prices[:period])/period  # Calculating the simple moving average
        multiplier = 2 / (period + 1)
        exponential_ma = [simple_ma*(multiplier)]
