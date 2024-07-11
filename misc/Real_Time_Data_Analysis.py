from stock_data_analysis import *
import pandas_datareader as pdr
from datetime import datetime, timedelta

real_time_msft = 'MSFT'
real_time_goog = 'GOOG'
real_time_amzn = 'AMZN'
real_time_aapl = 'AAPL'
real_time_sap = 'SAP'
real_time_meta = 'META'
real_time_005930_ks ='005930.KS'
real_time_intc = 'INTC'
real_time_ibm = 'IBM',
real_time_orcl = 'ORCL'
real_time_baba = 'BABA'
real_time_tcehy = 'TCEHY'
real_time_nvda = 'NVDA'
real_time_tsm = 'TSM',
real_time_nflx = 'NFLX'
real_time_tsla = 'TSLA'
real_time_crm = 'CRM'
real_time_adbe = 'ADBE'
real_time_pypl = 'PYPL'

real_time_data = {real_time_msft, real_time_goog, real_time_amzn, real_time_aapl,real_time_sap, real_time_meta, real_time_005930_ks, real_time_intc, real_time_ibm
                  , real_time_orcl, real_time_baba, real_time_tcehy, real_time_nvda, real_time_tsm, real_time_nflx, real_time_tsla, real_time_crm, real_time_adbe, real_time_pypl
                  }

class VolatilityTest:
    def __init__(self,stock_data):
        self.stock_data = stock_data
        self.real_time_data = real_time_data
    def historical_average_true_range(self):
        """ Takes stock data and returns Average True Range (ATR) and its graph
        Inputs:
        stock_data (DataFrame): a DataFrame of stock data and its corresponding stock name
        Return:
        string: a string containing the Average True Range
        string: a string containing the average ATR
        graph: a line plot of the Average True Range of each stock
        """
        for i, (name, data) in enumerate(self.stock_data.items()):
            data['HL'] = data['High'] - data['Low']
            data['AHL'] = abs(data['High'] - data['Close'].shift()) #calculate the
            # absolute value of high value less close value of the previous day
            data['ALC'] = abs(data['Low'] - data['Close'].shift()) #calculate the low value
            #less the close value of the previous day
            data['TR'] = data[['HL','AHL','ALC']].max(axis=1)

            #Calculate the Average True Range
            data['ATR'] = data['TR'].rolling(window=14).mean() #function rolling() used
            # for creating a rolling window calculation over a specific period

            #Calculate the average ATR of each stock

            average_ATR = np.mean(data['ATR'])
            volatility_threshold = average_ATR * 3
            print(f"The ATR for {name} is {data['ATR']}")
            print (f"The average ATR for {name} is {average_ATR}")
            #Create a line plot of ATR of each stock
            plt.plot(data['ATR'])
            plt.title(f"ATR for {name}")
            plt.tight_layout()
            plt.show()
            plt.close('all')

    def real_time_volatility_analysis(self):
        for j,(real_name,real_data) in enumerate(self.real_time_data.items()):
            end_date = datetime.now() - timedelta(1)
            start_date = end_date - timedelta(15)
            pdr.get_data_yahoo(real_name, start_date, end_date)







VolatilityTest(stock_data).average_true_range()

end_date = datetime.now() - timedelta(1)

start_date = end_date - timedelta(15)

pdr.get_data_yahoo(real_time_msft, start_date, end_date)

high_volatility_threshold = average_ATR * 3