from Stock_Market_Application import *
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import pylab
import scipy.stats as stats
from statsmodels.tsa.seasonal import seasonal_decompose
from datetime import datetime
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
#Create a dictionary to store all the stock dataframes and its name

stock_data = {'MSFT':data_msft,'GOOG': data_goog,'AMZN': data_amzn,'AAPL': data_aapl,'SAP':data_sap,'META':data_meta,'005930_KS':data_005930_ks,'INTC':data_intc,
              'IBM':data_ibm, 'ORCL':data_orcl, 'BABA':data_baba,'TCEHY':data_tcehy,'NVDA': data_nvda,'TSM':data_tsm,'NFLX': data_nflx,'TSLA':data_tsla,'CRM':data_crm,
              'ADBE':data_adbe,'PYPL':data_pypl}
'''a dictionary can store a dataframe'''

# A function to analyse data structures of stock data
def analyse_stock_data(stock_data):
    for i, (name,data) in enumerate(stock_data.items()):
        '''Takes a dictionary of stock data and returning an iterator aggregating
        elements from each list. Also print the info, shape (numbers of columns and rows), data type and the table of statistical descriptions.
        of each stock.  
        Inputs:
        stock_data (dictionary): A dictionary of stock data and their corresponding names
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

#A dictionary to store all the stock data with reset DateTimeindex indices
stock_reset_index = {}
#A function to reset DateTimeIndex for each dataframe
def reset_index_stock_data(stock_data):
    '''Takes a dictionary of stock data and resets the index of time.
    Input:
    stock_data(dictionary): a dictionary containing the stock data and its name
    Returns:
    stock_reset_index(dictionary): Dictionary with stock names as keys and stock data as values with reset Date indices
    '''
    for i,(name,data) in enumerate(stock_data.items()):
        data_reset = data.reset_index()
        stock_reset_index[name + '_reset'] = data_reset
    print(stock_reset_index)

reset_index_stock_data(stock_data)

#Pair plots for stock market data
for i,(name,data) in enumerate(stock_reset_index.items()):
    plt.figure()
    sns.pairplot(data)
    plt.title(name)
    plt.show()
plt.close('all')

#Principal Component Analysis for stock market data to identify dominant patterns and understand relationships
def principal_analysis(stock_data):
    '''
     Takes a dictionary of stock data and returns values of explained variance ratios for
     principal components, scatter plots of principal components and dataframes of loadings
     Input:
     stock_data(dictionary): a dictionary containing the stock data and their names

     Returns:
     string: explained variance ratios for principal components of stock data
     dataframe: loadings of principal components of stock data
     graph: scatter plots of principal components of stock data
            '''
    for i,(name,data) in enumerate(stock_data.items()):
        scaler = StandardScaler() #scaling our data with standard scaler
        scaled_data = scaler.fit_transform(data)

        n_components = 6 #specifying the number of dimensions we want to keep
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(scaled_data)
        #Convert to DataFrame
        component_names = [f"PC{j+1}" for j in range(principal_components.shape[1])]
        principal_df = pd.DataFrame(principal_components, columns=component_names)
        print(principal_df.head())
        #Check how much variance each principal component explains
        explained_variance = pca.explained_variance_ratio_
        print(f"Explained variance ratio for {name}: {explained_variance}")
        print(f"{name} has a maximum explained variance of PC1: \n {explained_variance[0]}")
        print(f"{name} has a second highest explained variance of PC2: \n {explained_variance[1]}")
        print(f"The rest has little impact on {name}")


        #Calculate the correlations/covariance between the original features and PCA-scaled units
        loadings = pd.DataFrame(
            pca.components_.T, #transpose the matrix of loadings
            columns = component_names,# the columns a re the principal components
            index = data.columns, #the rows are the original features
        )
        print(loadings)
        for col in loadings.columns:
            for feature in loadings.index:
                loading = loadings.loc[feature,col] #Obtaining the loading of each principal component
                #corresponding to each feature
                if loading>0.3:
                    print(f"{name}'s {col} may have a substantially positive relationship with {feature}")
                elif loading < -0.3:
                    print(f"{name}'s {col} may have a substantially inverse relationship with {feature}")
                else:
                    print(f"{name}'s {col} may have little relationship with {feature}")

        def plot_variance(pca):
            '''Takes PCA components and their corresponding explained variance
             and returns their plots of explained variance.
            Input:
            pca components: a string of pca components of all features in stock market data
            explained_variance: a list of explained variance for each pca component
            Returns:
            plot: a list of plots of explained variance for each pca
            '''
            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(6, 8))
            n = pca.n_components
            grid = np.arange(1, n + 1)

            # Explained variance
            axs[0].bar(grid, explained_variance)
            axs[0].set(
                xlabel="Component", title=" Explained Variance ",
                ylim=(0.0, 1.0)

            )
            # Cumulative Variance
            cv = np.cumsum(explained_variance)
            axs[1].plot(np.r_[0, grid], np.r_[0, cv], "o-")
            axs[1].set(
                xlabel="Component", title=" Cumulative Variance ",
                ylim=(0.0, 1.0)
            )
            # Set up figure
            plt.title(f"{name}",loc="left")
            fig.set(figwidth=8, dpi=100)
            plt.show()
            plt.close('all')
            return axs

        plot_variance(pca)
principal_analysis(stock_data)
#Visualise the reduced dimension data
def scatter_plot(stock_data):
    '''Takes stock_data and returns a scatter plot of principal components
    Input:
    stock_data (DataFrame): Dataframe with stock data and their names
    Return:
    DataFrame: a DataFrame of principal components
    graph: a scatter plot of principal components
    '''
    for i, (name, data) in enumerate(stock_data.items()):
        scaler = StandardScaler()  # scaling our data with standard scaler
        scaled_data = scaler.fit_transform(data)
        n_components = 6  # specifying the number of dimensions we want to keep
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(scaled_data)
        # Convert to DataFrame
        component_names = [f"PC{j + 1}" for j in range(principal_components.shape[1])]
        principal_df = pd.DataFrame(principal_components, columns=component_names)
        principal_df.head()
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=principal_df)  # creating a scatter plot for the principal components of all the data points for a company
        plt.title(f"Principal Components for {name}")
        plt.show()
    plt.close('all')
scatter_plot(stock_data)

#Probability Plot to determine whether our data follow a specific distribution.
cols = ['Open', 'High', 'Low', 'Adj Close', 'Close', 'Volume']
def detect_normal_distribution(stock_data):
    for i, (name,data) in enumerate(stock_data.items()):
        fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(10, 8))
        axs = axs.flatten() #flatten (meaning to transform into a one-dimensional array)
        # the axes array to easily iterate over it
        for ax, col in zip(axs,cols): #using zip function to iterate over the axes and column names
            #simultaneously
            stats.probplot(data[col], dist='norm', plot=ax) #Q-Q Plot
            ax.set_title(f"Probability Plot of {col} for {name}")
        plt.tight_layout()
        plt.show()
#Create a dictionary to store stock names and their dataframes
detect_normal_distribution(stock_data)




#Line Plot for stock data
trace_names = ['Open', 'High', 'Low', 'Adj Close', 'Close']
for i,(name,reset_data) in enumerate(stock_reset_index.items()):
    fig = go.Figure()
    for trace_name in trace_names:
        fig.add_trace(go.Scatter(x=reset_data['Date'], y=reset_data[trace_name],mode='lines',name='Open'))
    #fig.add_trace(go.Scatter(x=reset_data['Date'], y=reset_data['Volume'], mode='lines', name='Volume'))
    fig.update_layout(title=f"Stock Price for {name}")
    fig.show()

#Statistical Analysis to understand the data distribution
def seasonal_decomposition(stock_data):
    '''Takes a dictionary of stock_data and returning decomposition seasonality of the stock data
    Input:
    stock_data (dict): Dictionary of stock data and their names

    Returns:
    graph: Histogram of stock data and their names
    '''

    for i, (name,data) in enumerate(stock_data.items()):
        fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(10, 8))
        axs = axs.flatten()  # flatten (meaning to transform into a one-dimensional array)
        for (ax,col) in zip(axs,cols):
            decomposition = seasonal_decompose(data[col],model= 'additive', period=1) #2 types of models: additive (seasonality
            #and irregularities don't change as much when trend increases, multiplicative (seasonality and irregular variations increase in amplitude when trend increases
            #period: the number of observations in a cycle
            trend = decomposition.trend
            seasonal = decomposition.seasonal
            residual = decomposition.resid
          #Plot the original data, trend, seasonality and residuals
            plt.subplot(411)
            plt.plot(data[col],label='Original')
            plt.legend(loc='best')
            plt.subplot(412)
            plt.plot(trend,label='Trend')
            plt.legend(loc='best')
            plt.subplot(413)
            plt.plot(seasonal,label='Seasonality')
            plt.legend(loc='best')
            plt.subplot(414)
            plt.plot(residual,label='Residuals')
            plt.legend(loc='best')
            plt.tight_layout()
            plt.show()
    plt.close('all')
          #shapiro_test = stats.shapiro(data[col]) #Shapiro-Wilk Test
          #print(f"{name}'s Shapiro Test for {col}: {shapiro_test[0]}, p-value:{shapiro_test[1]}")
          #if shapiro_test[1] < 0.05:
          #  print(f"Based on Shapiro test, {name}'s {col} may not be a normal distribution.")
          #else:
          #  print (f"Based on Shapiro test, there is not enough evidence to suggest that {name}'s {col} may not be a normal distribution.")

           # kolmo_test = stats.kstest(data[col], 'norm') #Kolmogorov-Smirnov test
          #  print (f"{name}'s KS test for {col}: {kolmo_test.statistic}, p-value:{kolmo_test.pvalue}")
          #  if kolmo_test.pvalue < 0.05:
           #     print (f"Based on KS test, {name}'s {col} may not be a normal distribution.")
           # else:
            #    print(f"Based on KS test, there is not enough statistic evidence to suggest that {name}'s {col} may not be a normal distribution.")
seasonal_decomposition(stock_data)

#Henze-Zirkler's test to assess the multivariate normality of a dataset
from pingouin import multivariate_normality
for i,(name,data) in enumerate(stock_data.items()):
    #Compute test statistic and p-value for Henze-Zirkler's test
    results = multivariate_normality(data,alpha=.05)
    if results[2] == False:
        print(f"{name}'s results for Mardia's test: {results} and we do not have evidence that there might be multivariate normality in the stock data.")
    else:
        print(f"{name}'s results for Mardia's test: {results} and we have evidence that there might be multivariate normality in the stock data.")

#Mardia's test to assess the multivariate normality
from scipy.stats import skew,kurtosis, chi2
def mardia_test(stock_data):
    '''Takes stock data and returns '''
    for i,(name,data) in enumerate(stock_data.items()):
    #Compute skewness and kurtosis for each variable
        mardia_skewness = skew(data,axis=0) #skewness of the data
        mardia_kurtosis = kurtosis(data,axis=0) #kurtosis of the data

        # Compute Mardia's multivariate skewness and kurtosis
        n = data.shape[0] #number of observations
        m = data.shape[1] #number of features

        skewness_multivariate = (n/ ((n-1) * (n-3))) * np.sum(mardia_skewness**2) # calculate the Mardia's multivariate skewness: the number of observations divided the multiplication
        #the first and fourth moments and multiplied by the sum of mardia skewness squared
        kurtosis_multivariate = (1 - m) * np.sum(mardia_kurtosis) # calculate the Mardia's multivariate kurtosis: subtract 1 by the number of features, multiplied by the sum of mardia
        #kurtosis

        #Compute test statistic
        test_statistic = skewness_multivariate**2 + (kurtosis_multivariate - m*(m+2))**2 #calculate test statistic by squaring the Mardia's multivariate skewness
        # then add the squared Mardia's multivariate kurtosis subtracting the number of features times the number of features plus 2


        #Compute the p-value
        mardia_p = 1 - chi2.cdf(test_statistic,df=m*(m+1)*(m+2)/6) #compute the p-value for the test statistic using chi-squared distribution
        # with degrees of freedom m*(m+1)*(m+2)/6

        if mardia_p < 0.05:
            print(f"{name}'s results for Mardia's test is {mardia_p} and we do not have evidence that there might be multivariate normality in its stock data")
        else:
            print(f"{name}'s results for Mardia's test is {mardia_p} and we have evidence that there might be multivariate normality in its stock data")

mardia_test(stock_data)

#Investigating outliers in our stock data using the method of Interquartile Range(IQR) since all stock data are not normally
# distributed.
# It is a common technique used for stock market because it gives us insights into the spread of stock prices
# over a specific period. It is resistant to outliers, making it robust.
#An outlier: an extremely high or low data point relative to the nearest data points and
#the rest of the neighboring co-existing vals in a dataset.
#The IQR is the range between the first quartile (25th percentile) and the third quartile (75th percentile) of the data.
#Any point falling below Q1 - 1.5IQR or above Q3 + 1.5IQR is considered an outlier.

for i,(name,data) in enumerate(stock_data.items()):
    '''Takes stock data and returns and box plots outliers of them.
    Input:
    stock_data: a dictionary of stock data
    Returns:
    plot: a box plot of the stock data
    Series: a series of outliers for each stock
    '''

    Q1 = data['Close'].quantile(0.25)
    Q3 = data['Close'].quantile(0.75)
    IQR = Q3 - Q1
    #Define bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    #Detect outliers
    data_outliers = data[(data['Close'] < lower_bound) | (data['Close'] > upper_bound)]
    if data_outliers.empty:
        print(f"The outliers of {name} are none")
    else:
        print(f"The outliers of {name} are {data_outliers['Close']}")
        print(f"Number of outliers: {data_outliers.shape[0]}")
    #Create boxplot
    plt.boxplot(data['Close'])
    plt.title (f"Box Plot of {name}")
    plt.show()
    plt.close('all')

 #Heat maps to show the correlation between different features
 for i,(name,data) in enumerate(stock_data.items()):
     #Calculate the correlation matrix
     feat_corr = data.corr()

     #Create the heat map of the correlation matrix
     plt.figure(figsize=(10,8))
     sns.heatmap(corr, cmap='coolwarm',annot=True)
     plt.title (f" The correlation matrix for {name}")
     plt.show()






















