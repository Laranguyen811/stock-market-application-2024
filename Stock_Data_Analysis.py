from Stock_Market_Application import *
import matplotlib.pyplot as plt
import pandas as pd

#Create a dictionary to store all the stock dataframes and its name
stock_data = {'MSFT':data_msft,'GOOG': data_goog,'AMZN': data_amzn,'AAPL': data_aapl,'SAP':data_sap,'META':data_meta,'005930_KS':data_005930_ks,'INTC':data_intc,
              'IBM':data_ibm, 'ORCL':data_orcl, 'BABA':data_baba,'TCEHY':data_tcehy,'NVDA': data_nvda,'TSM':data_tsm,'NFLX': data_nflx,'TSLA':data_tsla,'CRM':data_crm,
              'ADBE':data_adbe,'PYPL':data_pypl}
'''a dictionary can store a dataframe'''

# A function to analyse data structures of stock data
def analyse_stock_data(stock_data):
    for i, (name,data) in enumerate(stock_data.items()):
        '''taking 2 lists of stock data and stock names and returning an iterator aggregating
        elements from each list. Also print the info, shape (numbers of columns and rows), data type and the table of statistical descriptions.
        of each stock.  
        Inputs:
        stock_data (dataframes): list of stock dataframes
        stock_names (string): list of stock names
        Returns:
        string: stock name
        string: information about the stock
        string: numbers of columns and rows
        string: data types of each column in the stock dataframe
        string: stats of the stock dataframe 
        '''
        print(f'{name}')
        print("\nInfo:")
        print(data.info())
        print("\nShape:")
        print(data.shape)
        print("\nData Types:")
        print(data.dtypes)
        print("\nDescribe:")
        print(data.describe())
        print(data.duplicated().sum())
        print("\n Number of duplicated values: " + str(data.duplicated().sum()))


analyse_stock_data(stock_data)
#Convert the DataFrame to an Agate table
#table = ag.Table.from_object(data_msft)
#print(table.column_names)

#new_table = table.select(['column_name'])

#Checking outliers in our stock data using the method of Interquartile Range(IQR).
#The IQR is the range between the first quartile (25th percentile) and the third quartile (75th percentile) of the data.
#Any point falling below Q1 - 1.5IQR or above Q3 + 1.5IQR is considered an outlier.

for i,(name,data) in enumerate(stock_data.items()):
    Q1 = data['Close'].quantile(0.25)
    Q3 = data['Close'].quantile(0.75)
    IQR = Q3 - Q1
    #Define bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    #Detect outliers
    data_outliers = data[(data['Close'] < lower_bound) | (data['Close'] > upper_bound)]
    if data_outliers.empty:
        print(f"The outliers of {name} is none")
    else:
        print(f"The outliers of {name} is {data_outliers['Close']}")
        print(f"Number of outliers: {data_outliers.shape[0]}")
    #Create boxplot
    plt.boxplot(data['Close'])
    plt.title (f"Box Plot of {name}")
    plt.show()
















