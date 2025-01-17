import pandas as pd
from unicodedata import category

from matplotlib.galleries.examples.text_labels_and_annotations.fonts_demo import weight


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

    def calculate_position_size(self,account_balance,risk_per_trade,stop_loss_amount):
        '''
        Calculates the position size (the number of units a trader/investor invests in a specific security to control risks and maximise returns) of a stock based on account balance, risk per trade and stop loss amount.
        Inputs:
            account_balance(float): A float number of account balance
            risk_per_trade(float): A float number of percentage risk per trade
            stop_loss_amount(float): A float number of stop loss amount
        Returns:
             float: A float number of position size
        '''
        return (account_balance * risk_per_trade) / stop_loss_amount

    def moving_average_crossover(self,prices, short_period, long_period):
        '''
        Calculates the moving average crossover (a popular technical analysis to identify trends in prices using simple moving average, crossing over between short terms and long terms) of a stock based on short and long periods.
        Inputs:
            prices(list): A list of stock prices
            short_period(int): An integer of a short period
            long_period(int): An integer of a long period
        Returns:
            string: A string of "Buy", "Sell" or "Hold" depending on the short moving average and long moving average
        '''
        short_ma = self.calculate_mas(self.prices,short_period)[0]
        long_ma = self.calculate_mas(self.prices,long_period)[0]
        if short_ma > long_ma: return "Buy"
        elif short_ma < long_ma: return "Sell"
        return "Hold"

    def calculate_allocation_ratios(self,total_capital, allocations):
        '''
        Calculates the allocation ratio using total capital and allocations.
        Inputs:
            total_capital(float): A float number of total capital.
            allocations(dict): A dictionary of allocations for asset and ratio
        Returns:
            float: A float of the allocation ratio
        '''
        return {asset: total_capital * ratio for asset, ratio in allocations.items()}

    def rebalance_portfolio(self,portfolio, target_ratios):
        '''
        Takes a portfolio and target ratios and rebalances.
        Inputs:
            portfolio(dictionary): A dictionary of portfolio values
            target_ratios(dictionary): A dictionary of target ratios for each security
        Returns:
            dict: A dictionary of asset and the amount to rebalance to for each security
        '''
        total_value = sum(portfolio.values())
        return {asset: total_value * ratio for asset, ratio in target_ratios.items()}

    def calculate_max_drawdown(self,portfolio_values):
        '''
        Takes portfolio values and calculates the maximum drawdown (the biggest asset price drop from a peak to a trough, key indicator of downside risk over a period).
        Inputs:
            portfolio_values(list): A list of portfolio values.
        Returns:
             float: A float number of maximum drawdown
        '''
        peak = portfolio_values[0]
        max_drawdown = 0
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        return max_drawdown

    def calculate_risk_reward_ratio(self,potential_profit,potential_loss):
        '''
        Takes potential profit and loss and returns the risk reward ratio (the ratio between potential profit and potential loss).
        Inputs:
            potential_profit(float): A float number representing the potential profit.
            potential_loss(float): A float number representing the potential loss.
        Returns:
            float: A float number representing the risk reward ratio.
        '''
        return potential_loss/potential_profit

    def calculate_volatility(self,prices, period):
        '''
        Takes prices and period and calculates the volatility threshold (if reaches or exceeds will trigger action or decision) of the stock.
        Inputs:
            prices(list): A list of prices of a stock.
            period(int): An integer of a period
        Returns:
             float: A float number containing the volatility of the stock.
        '''
        returns = [prices[i]/ prices[i-1] - 1 for i in range(1,len(prices))]
        return (sum([(r - sum(returns) / len(returns)) ** 2 for r in returns]))

    def calculate_lookback_period(self,prices, period):
        '''
        Takes prices and period and calculates the lookback period (the period for which data is analysed for calculations or decisions).
        Inputs:
            prices(list): A list of prices.
            period(int): An integer of a period.
        Returns:
             integer: A integer representing the lookback period.
        '''
        return prices[-period:]

    def calculate_weighted_carbon_footprint_score(self, score, weight):
        '''
         Takes the score of each carbon footprint category and the weight corresponding to each category and returns the carbon footprint weighted score.
         Inputs:
            score(float): A float number of the score for each carbon footprint category
            weight(float): A float number of the weight of each carbon footprint category
         Returns:
               float: A float number representing the weighted carbon footprint score
        '''
        weighted_carbon_footprint_score = score * weight
        return weighted_carbon_footprint_score

    def calculate_carbon_footprint_score(self, carbon_footprint_data: dict, weights:dict):
        '''
        Takes a carbon footprint data dictionary and a dictionary of weights corresponding to each carbon footprint category and returns the total carbon footprint score.
        Inputs:
            carbon_footprint_data(dict): A dictionary of carbon footprint data
            weights(dict): A dictionary of weights for each category
        Returns:
            float: A float number of the total carbon footprint score
        '''
        total_carbon_footprint_score = sum(self.calculate_weighted_carbon_footprint_score(carbon_footprint_data[category], weights[category]) for category in carbon_footprint_data)
        return total_carbon_footprint_score

    def calculate_carbon_footprint(self,activity_data,emission_factor):
        '''
            Calculates the carbon footprint by measuring green house gas emissions produced directly and indirectly by a company.
            Inputs:
                activity_data(dictionary): A dictionary of activity data
                emission_factor(float): A float number of emission factor
            Returns:
                float: A float number of green house gas emissions
        '''
        ghg_emissions = activity_data * emission_factor
        return ghg_emissions
    def calculate_weighted_renewable_energy_score(self,score,weight):
        '''
        Takes a score of each renewable energy category and a weight corresponding to each renewable energy category and returns the weighted score.
        Inputs:
            score(float): A float number of the score for each renewable energy category
            weight(float): A float number of the weight of each renewable energy category
        Returns:
             float: A float number representing the weighted renewable energy score
        '''
        weighted_renewable_energy_score = score * weight
        return weighted_renewable_energy_score
    def calculate_renewable_energy_score(self,renewable_energy_data: dict, weights:dict):
        '''
        Takes the renewable energy data dictionary and the dictionary of weights corresponding to the renewable energy dictionary and returns the total renewable energy score.
        Inputs:
            renewable_energy_data(dict): A dictionary of renewable energy data
            weights(dict): A dictionary of weights for each category
        Returns:
             float: A float number representing the total renewable energy score
        '''
        total_reneweable_energy_score = sum(self.calculate_weighted_renewable_energy_score(renewable_energy_data[category],weights[category]) for category in renewable_energy_data)
        return total_reneweable_energy_score

    def calculate_energy_consumption(self,energy_data: dict ) -> pd.DataFrame:
        ''' Takes the energy consumption data and returns the total energy consumption for an organisation.
        Inputs:
            energy_data(dict): A dictionary of energy consumption data
        Returns:
            pd.DataFrame: A DataFrame containing the total energy consumption of an organisation
        '''

        # Create a DataFrame for the total energy consumption data
        df = pd.DataFrame(energy_data)
        df['Total Energy Consumption (kWh'] = df.sum(axis=1)

        # Calculating annual total energy consumption
        annual_total_consumption = df['Total Energy Consumption (kWh)'].sum()

        # Adding annual total to the DataFrame
        df.loc['Annual Total'] = df.sum()
        df.at['Annual Total', 'Total Energy Consumption (kWh)'] = annual_total_consumption  # Locating the annual total consumption

        return df

    def calculate_water_usage(self,water_usage_data: dict) -> pd.DataFrame:
        '''
        Takes the water usage data and returns the total water usage for an organisation.
        Inputs:
            water_usage_data(dict): A dictionary containing water sources and values as lists of monthly usage
        Returns:
            pd.DataFrame: A DataFrame containing the total water usage for each month
        '''
        water_usage_df = pd.DataFrame(water_usage_data)
        # Calculating total water usage for each month
        water_usage_df['Total Water Usage (cubic meters)'] = water_usage_df.sum(axis=1)

        # Calculating annual total water usage
        annual_total_water_usage = water_usage_df['Total Water Usage (cubic meters)'].sum()

        # Adding annual total for the DataFrame
        water_usage_df.loc['Annual Total'] = water_usage_df.sum()
        water_usage_df.at['Annual Total', 'Total Water Usage (cubic meters)'] = annual_total_water_usage

        return water_usage_df



    def calculate_weighted_waste_score(self,quantity,score):
        '''
        Takes the quantity and score and calculates the weighted score for each waste category.
        Inputs:
            quantity(float): A float number representing the quantity of waste category
            score(float): A float number representing the weighted score
        Returns:
            float: A float number representing the weighted score
        '''
        weighted_score = quantity * score
        return weighted_score

    def calculate_waste_management(self,waste_data: dict) -> pd.DataFrame:
        '''
        Takes the waste data and returns the total waste management score for an organisation.
        Inputs:
            waste_data(dict): A dictionary containing waste data
        Returns:
            pd.DataFrame: A DataFrame containing the total waste management for an organisation
        '''
        total_quantity = sum(quantity for quantity, _ in waste_data.values())  # ignoring the scores
        total_weighted_score = sum(self.calculate_weighted_waste_score(quantity, score) for quantity, score in waste_data.values())
        waste_management_score = total_weighted_score / total_quantity if total_quantity else 0
        return waste_management_score

    def weighted_labor_score(self,score,weight):
        '''
        Takes the score of each category and the corresponding weight to calculate the weighted labor score.
        Inputs:
            score(float): A float number representing the score for each labor pratice
            weight(float): A float number representing the weight of each labor pratice
        Returns:
            float: A float number representing the weighted labor score
        '''
        labor_weighted_score = score * weight
        return labor_weighted_score


    def calculate_labor_practices_score(self,labor_data: dict, weights: dict):
        '''
        Takes the labor data, score and weight and returns the total labor practices score for an organisation.
        Inputs:
            labor_data(dict): A dictionary containing labor practice data
            weights(dict): A dictionary containing weights corresponding to categories of labor practices
        Returns:
            float: A float number of total labor practices score
        '''
        total_labor_practices_score = sum(self.calculated_weighted_labor_score(labor_data[category], weights[category]) for category in labor_data)
        return total_labor_practices_score

    def calculate_di_weighted_score(self,score, weight):
        '''
        Takes the score of each category and the corresponding weight to calculate the total diversity and inclusion score.
        Inputs:
            score(float): A float number representing the score for each D&I category
            weight(float): A float number representing the weight of each D&I category
        Returns:
            float: A float number representing the total diversity and inclusion score
        '''
        d_and_i_score = score * weight
        return d_and_i_score

    def calculate_di_score(self,di_data: dict,weights: dict):
        '''
        Takes the diversity and inclusion score for each category and the corresponding weight for each category to calculate the total diversity and inclusion score.
        Inputs:
            di_data(dict): A dictionary containing a diversity and inclusion score for each category
            weights(dict): A dictionary containing a weight for each category
        Returns:
            float: A float number representing the total diversity and inclusion score
        '''
        total_di_score = sum(self.calculate_diversity_and_inclusion_weighted_score(di_data[category],weights[category]) for category in di_data)
        return total_di_score

    def calculate_weighted_ce_score(self,score,weight):
        '''
        Takes the score of each community engagement category and the weight for each and returns the weighted score.
        Inputs:
            score(float): A float number representing the score for each community engagement category
            weight(float): A float number representing the weight of each community engagement category
        Returns:
            float: A float number representing the weighted score
        '''
        weighted_ce_score = score * weight
        return weighted_ce_score

    def calculate_ce_score(self,ce_data: dict,weights: dict):
        '''
        Takes a dictionary of community engagement data and the weights corresponding to community engagement categories and returns the total community engagement score.
        Inputs:
            ce_data(dict): A dictionary containing community engagement data
            weights(dict): A dictionary containing a weight for each community engagement category
        Returns:
            float: A float number representing the total community engagement score
        '''
        ce_score = sum(self.calculate_weighted_ce_score(ce_data[category],weights[category]) for category in ce_data)
        return ce_score

    def calculate_weighted_bd_score(self,score,weight):
        '''
        Takes the board diversity score for each category and the weight for each and returns the weighted board diversity score.
        Inputs:
            score(float): A float number representing the score for each board diversity category
            weight(float): A float number representing the weight of each board diversity category
        Returns:
            float: A float number representing the weighted board diversity score
        '''
        weighted_bd_score = score * weight
        return weighted_bd_score

    def calculate_bd_score(self,bd_data: dict,weights: dict):
        '''
        Takes a dictionary of board diversity data and the weights corresponding to board diversity categories and returns the total board diversity score.
        Inputs:
            bd_data(dict): A dictionary containing board diversity data
            weights(dict): A dictionary containing a weight for each board diversity category
        Returns:
            float: A float number representing the total board diversity score
        '''
        total_db_score = sum(self.calculate_weighted_bd_score(bd_data[category],weights[category]) for category in bd_data)
        return total_db_score

    def calculate_weighted_ep_score(self,score,weight):
        '''
        Takes the score of each executive pay category and the weight for each and returns the weighted score.
        Inputs:
            score(float): A float number representing the score for each executive pay category
            weight(float): A float number representing the weight of each executive pay category
        Returns:
            float: A float number representing the weighted score
        '''
        weighted_ep_score = score * weight
        return weighted_ep_score

    def calculate_ep_score(self,ep_data: dict,weights: dict):
        '''
        Takes a dictionary containing the executive pay score for each category and the dictionary of weights corresponding to the categories and returns the total score.
        Inputs:
            ep_data(dict): A dictionary of executive pay data.
            weights(dict): A dictionary containing a weight for each executive pay category
        Returns:
            float: A float number representing the weighted score
        '''
        total_ep_score = sum(self.calculate_weighted_ep_score(ep_data[category],weights[category]) for category in ep_data)
        return total_ep_score

    def calculate_weighted_ap_score(self,score,weight):
        '''
        Takes the score of anti-corruption policy score for each category and the weight for each and returns the weighted score.
        Inputs:
            score(float): A float number of the score for each anti-corruption policies category
            weight(float): A float number of the weight for each anti-corruption policies category
        Returns:
            float: A float number representing the weighted anti-corruption policies score
        '''
        weighted_ap_score = score * weight
        return weighted_ap_score

    def calculate_ap_score(self,ap_data: dict,weights: dict):
        '''
        Takes a dictionary of anti-corruption data and the dictionary of weights for anti-corruption categories and returns the total score.
        Inputs:
            ap_data(dict): A dictionary of anti-corruption policies data
            weights(dict): A dictionary of weights corresponding to anti-corruption policies categories
        Returns:
            float: A float number representing the total anti-corruption policies score
        '''
        total_ap_score = sum(self.calculate_weighted_ap_score(ap_data[category],weights[category]) for category in ap_data)
        return total_ap_score

    def calculate_weighted_environmental_score(self,score,weight):
        '''
        Takes the score of each environmental category and the weight for each and returns the weighted environmental score.
        Inputs:
            score(float): A float number representing the score for each environmental category
            weight(float): A float number representing the weight of each environmental category
        Returns:
            float: A float number representing the weighted environmental score
        '''
        weighted_environmental_score = score * weight
        return weighted_environmental_score

    def calculate_environmental_score(self,environmental_data: dict,weights: dict):
        '''
        Takes the environmental data dictionary and the dictionary of weights corresponding to environmental categories and returns the total environmental score.
        Inputs:
            environmental_data(dict): A dictionary containing environmental data
            weights(dict): A dictionary containing a weight for each environmental category
        Returns:
             float: A float number representing the total environmental score
        '''
        total_environmental_score = sum([self.calculate_carbon_footprint_score.total_carbon_footprint_score + self.calculate_renewable_energy_score.total_renewable_energy_score + self.calculate_waste_management.total_waste_management_score])
        return total_environmental_score

    def calculate_social_score(self,environmental_data: dict,weights: dict):
        '''
        Takes the environmental data dictionary and the dictionary of weights corresponding to environmental categories and returns the total social score.
        Inputs:
            environmental_data(dict): A dictionary containing environmental data
            weights(dict): A dictionary containing a weight for each environmental category
        Returns:
            float: A float number representing the total social score
        '''
        total_social_score = sum([self.calculate_labor_practices_score.total_labor_practices_score + self.calculate_di_score.total_di_score + self.calculate_ce_score.total_ce_score])
        return total_social_score

    def calculate_governance_score(self,governance_data: dict,weights: dict):
        '''
        Takes the governance data dictionary and the dictionary of weights corresponding to governance categories and returns the total governance score.
        Inputs:
            governance_data(dict): A dictionary containing governance data
            weights(dict): A dictionary containing weights for each governance category
        Returns:
            float: A float number representing the total governance score
        '''
        total_governance_score = sum([self.calculate_ap_score.total_ap_score + self.calculate_bd_score.total_bd_score + self.calculate_ep_score.total_ep_score])
        return total_governance_score


