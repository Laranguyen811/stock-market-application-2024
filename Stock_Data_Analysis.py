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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from scipy.stats import chi2,f
from pingouin import multivariate_normality
from scipy.spatial.distance import mahalanobis
#Create a dictionary to store all the stock dataframes and its name

stock_data = {'MSFT':data_msft,'GOOG': data_goog,'AMZN': data_amzn,'AAPL': data_aapl,'SAP':data_sap,'META':data_meta,'005930_KS':data_005930_ks,'INTC':data_intc,
              'IBM':data_ibm, 'ORCL':data_orcl, 'BABA':data_baba,'TCEHY':data_tcehy,'NVDA': data_nvda,'TSM':data_tsm,'NFLX': data_nflx,'TSLA':data_tsla,'CRM':data_crm,
              'ADBE':data_adbe,'PYPL':data_pypl}
'''a dictionary can store a dataframe'''

# A function to analyse data structures of stock data
def analyse_stock_data(stock_data):
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
    for i, (name,data) in enumerate(stock_data.items()):
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
    '''
    Takes a dictionary of stock data and resets the index of time.
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

#Heat maps to show the correlation between different features
for i,(name,data) in enumerate(stock_data.items()):
    #Calculate the correlation matrix
    feat_corr = data.corr()

    #Create the heat map of the correlation matrix
    plt.figure(figsize=(10,8))
    sns.heatmap(feat_corr, cmap='coolwarm',annot=True)
    plt.title (f" The correlation matrix for {name}")
    plt.show()

#Calculate correlation coefficients
def calculate_corr_coeffs(stock_data):
    '''
    Takes stock data and calculates the correlation coefficients between two different features.
    Input:
    stock_data (Dictionary):a dictionary of stock data and their corresponding names
    Returns:
    string: a string of correlation coefficients between the features for each stock data
    '''
    for i,(name,data) in enumerate(stock_data.items()):
        corr_matrix = data.corr()

    #Calculate the correlation coefficients (off-diagonal elements)
        print(f" Correlation coefficient for {name} : \n {corr_matrix}")
        for col in corr_matrix.columns:
            for row in corr_matrix.index:
                corr_coeff= corr_matrix.loc[col,row]
                if corr_coeff > 0.8:
                    print(f"We have evidence that there might be a strong positive correlation between {name}'s {col} and {row} with the correlation coefficient of {corr_coeff}.")
                elif 0.6 <= corr_coeff <= 0.8:
                    print(f"We have evidence that there might be a moderate positive correlation between {name}'s {col} and {row} with the correlation coefficient of {corr_coeff}.")
                elif 0.4 <= corr_coeff <0.6:
                    print(f"We have evidence that there might be a weak positive correlation between {name}'s {col} and {row} with the correlation coefficient of {corr_coeff}.")
                elif 0 <= corr_coeff < 0.4:
                    print(f"We have evidence that there might be a very weak positive correlation between {name}'s {col} and {row} with the correlation coefficient of {corr_coeff}.")
                elif -0.4 <= corr_coeff <0:
                    print(f"We have evidence that there might be a very weak negative correlation between {name}'s {col} and {row} with the correlation coefficient of {corr_coeff}.")
                elif -0.6< corr_coeff <= -0.4:
                    print(f"We have evidence that there might be a weak negative correlation between {name}'s {col} and {row} with the correlation of {corr_coeff}.")
                elif -0.6 <= corr_coeff <= -0.8:
                    print(f"We have evidence that there might be a moderate weak positive correlation between {name}'s {col} and {row} with the correlation coefficient of {corr_coeff}")
                else:
                    print(f"We have evidence that there might be a strong negative correlation between {name}'s {col} and {row} with the correlation of {corr_coeff }.")

calculate_corr_coeffs(stock_data)


#Principal Component Analysis for stock market data to identify dominant patterns and understand relationships

#Visualise the reduced dimension data
def scatter_plot(stock_data):
    '''
    Takes stock_data and returns a scatter plot of principal components
    Input:
    stock_data (DataFrame): Dataframe with stock data and their names
    Returns:
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
def detect_normal_distribution(stock_data):
    ''' Takes a DataFrame of stock data and return probability plot (Q-Q plot) and histogram for each stock.
    Input:
    stock_data(Dataframe): a Dataframe of stock data and their respective names
    Returns:
    graph: a Probability plot of stock data
    graph: a histogram of stock data
    '''
    for i, (name,data) in enumerate(stock_data.items()):
        cols = data.columns
        fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(10, 8))
        axs = axs.flatten() #flatten (meaning to transform into a one-dimensional array)
        # the axes array to easily iterate over it
        for ax, col in zip(axs,cols): #using zip function to iterate over the axes and column names
            #simultaneously
            stats.probplot(data[col], dist='norm', plot=ax) #Q-Q Plot
            ax.set_title(f"Probability Plot of {col} for {name}")
        plt.tight_layout()
        plt.show()
    plt.close('all')
#Create a dictionary to store stock names and their dataframes
detect_normal_distribution(stock_data)


def histogram_plot(stock_data):
    """ Takes a DataFrame of stock data and return histogram of each stock
    Input:
    stock_data(Dataframe): a Dataframe of stock and its corresponding name

    Returns:
        graph: A histogram of stock data
    """
    for i, (name,data) in enumerate(stock_data.items()):
        cols = data.columns
        fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(8,6))
        axs = axs.flatten()
        for ax, col in zip(axs,cols):
            ax.hist(data[col], bins=6)
            ax.set_title(f"Histogram of {col} for {name}")
        plt.tight_layout()
        plt.show()
    plt.close('all')

histogram_plot(stock_data)


#def plot_distribution(stock_data):
  #  '''
  #  Takes a DataFrame of stock data and return probability plot (Q-Q plot) and histogram for each stock.
  #  Input:
  #  stock_data(Dataframe): a Dataframe of stock data and their respective names
  #  Returns:
  #  graph: a Probability plot of stock data
  #  graph: a histogram of stock data
   # '''
    #for name, data in stock_data.items():
     #   cols = data.columns
      #  fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(10, 8))
       # axs = axs.flatten()  # flatten (meaning to transform into a one-dimensional array)
       # for ax, col in zip(axs,cols):

            # Q-Q plot
        #    stats.probplot(data[col], dist='norm', plot=axs[0])
         #   ax[0].set_title(f"Probability Plot of {col} for {name}")

            # Histogram
          #  ax[1].hist(data[col], bins=6, color='skyblue', edgecolor='black')
           # ax[1].set_title(f"Histogram of {col} for {name}")

           # plt.tight_layout()
           # plt.show()
    #plt.close('all')

# Create a dictionary to store stock names and their dataframes
#plot_distribution(stock_data)
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
    graph: line plots of seasonal decomposition of stock data and their names
    '''

    for i, (name,data) in enumerate(stock_data.items()):
        fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(12,10))
        axs = axs.flatten()  # flatten (meaning to transform into a one-dimensional array)
        for j,(ax,col) in enumerate(zip(axs,cols)):
            decomposition = seasonal_decompose(data[col],model= 'additive', period=1) #2 types of models: additive (seasonality
            #and irregularities don't change as much when trend increases, multiplicative (seasonality and irregular variations increase in amplitude when trend increases
            #period: the number of observations in a cycle, choose yearly (1) to analyse the long-term trend
            trend = decomposition.trend
            seasonal = decomposition.seasonal
            residual = decomposition.resid
          #Plot the original data, trend, seasonality and residuals
            ax.plot(data[col],label='Original', linewidth=2, linestyle='-')
            ax.plot(trend,label='Trend', linewidth=1, linestyle='--')
            ax.plot(seasonal,label='Seasonality',linewidth=2, linestyle='-')
            ax.plot(residual,label='Residuals',linewidth=1, linestyle='--')

            ax.legend(loc='best')
            ax.set_title(f"Seasonal decomposition of {name}'s {col}")
        plt.tight_layout()
        plt.show()
    plt.close('all')
seasonal_decomposition(stock_data)

#Henze-Zirkler's test to assess the multivariate normality of a dataset
def henze_zirkler_test(stock_data):
    ''' Takes stock data and returns Henze-Zirkler's test statistic.
    Input:
    stock_data (dictionary): a dictionary of stock data and its corresponding values
    Returns:
    string: a string of Henze-Zirkler's test statistic for stock data
    '''
    for i,(name,data) in enumerate(stock_data.items()):
        #Compute test statistic and p-value for Henze-Zirkler's test
        results = multivariate_normality(data,alpha=.05) #compute the nonnegative function distance measuring the distance between
        #the empirical characteristic function of the data and the character function of multivariate normal distribution.
        #Process: measuring the mean, variance and smoothness of the data. Then, we lognormalised the mean and covariance, and estimate the p-value.
        if results[2] == False:
            print(f"{name}'s results for Mardia's test: {results} and we do not have evidence that there might be multivariate normality in the stock data.")
        else:
            print(f"{name}'s results for Mardia's test: {results} and we have evidence that there might be multivariate normality in the stock data.")

#Mardia's test to assess the multivariate normality
def mardia_test(stock_data):
    '''Takes stock data and returns the p-values of Mardia's test statistic
    Input:
    stock_data (dictionary): A dictionary of stock data and its corresponding values
    Returns:
    string: a string of p-values of Mardia's test statistic for each stock data
    '''
    for i,(name,data) in enumerate(stock_data.items()):
    #Compute skewness and kurtosis for each variable
        mardia_skewness = skew(data,axis=0) #skewness of the data
    #b_{1,p} = \frac{1}{n^2} \sum_{i=1}^{n} \sum_{j=1}^{n} [(x_i - \bar{x})^T S^{-1} (x_j - \bar{x})]^3
    #for x_i and x_j data points, we measure the distance between them and with itself.x bar is the
    # mean (average) of all the data points. The double summation means that for each data point x_i, we are comparing with data point
    #x_j, including itself.
    #S(-1) is the inverse of the covariance matrix (the covariance matrix measure how much of each of dimensions/variances
    # vary from the mean with respect to each other). We use the inverse of the covariance matrix, or the precision matrix, to hold other variables
    #constant while measuring the relationship of two variables in questions, by normalising the data (essentially removing any correlation and variance
    #between two variables.It is crucial in multivariate analysis/statistics.
    #(x_i - \bar{x})^T: the transpose matrix of the difference between the data point x_i and the mean, since we need a row vector to multiply
    #with the precision matrix.
    #(x_j - \bar{x})^3: cubing the difference between x_j and the mean to estimate the skewness
        mardia_kurtosis = kurtosis(data,axis=0) #kurtosis of the data
    #b_{2,p} = \frac{1}{n} \sum_{i=1}^{n} [(x_i - \bar{x})^T S^{-1} (x_i - \bar{x})]^2
    #We only calculate for data point x_i since we only measure the tailedness of the data by looking at the distance from a data point to the mean.
    # Therefore, we square the difference between x_i and the mean to calculate the kurtosis.
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



class Outliers: #using class to define how objects should behave (type). An instance is an object with that type
    def __init__(self,stock_data): #calling this method when we want to initialise the class
        #using self inside a class definition to reference the instance object we have created
        self.stock_data = stock_data

    # Investigating outliers in our stock data using the method of Interquartile Range(IQR) since all stock data are not normally
    # distributed.
    # It is a common technique used for stock market because it gives us insights into the spread of stock prices
    # over a specific period. It is resistant to outliers, making it robust.
    # An outlier: an extremely high or low data point relative to the nearest data points and
    # the rest of the neighboring co-existing vals in a dataset.
    # The IQR is the range between the first quartile (25th percentile) and the third quartile (75th percentile) of the data.
    # Any point falling below Q1 - 1.5IQR or above Q3 + 1.5IQR is considered an outlier.
    def iqr_method(self):
        '''Takes stock data and returns and box plots outliers of them.
           Input:
           stock_data: a dictionary of stock data
           Returns:
           plot: a box plot of the stock data
           Series: a series of outliers for each stock
        '''
        for name,data in self.stock_data.items():
            fig, axes = plt.subplots(3, 2, figsize=(10, 8))
            axes = axes.flatten()
            for ax,col in zip(axes,data.columns):
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                # Define bounds for outliers
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
            # Detect outliers
                data_outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
                lower_outliers = data_outliers[col][data_outliers[col] < lower_bound] #filtering operations to select outliers in the lower bounds
                upper_outliers = data_outliers[col][data_outliers[col] > upper_bound] #filtering operations to select outliers in the upper bounds
                min_outlier = data_outliers[col].min()
                max_outlier = data_outliers[col].max()
                min_row = data_outliers.loc[data_outliers[col] == min_outlier] #selecting row with the minimum outlier for the column
                max_row = data_outliers.loc[data_outliers[col] == max_outlier] #selecting row with maximum outlier for the column
                if data_outliers.empty:
                    print(f"The outliers of {name}'s {col} are none")
                else:
                    print(f"The outliers of {name}'s {col} are \n {data_outliers[col]}")
                    print(f"Number of outliers: {data_outliers.shape[0]}")
                    print(f"Upper bound outliers of {name}'s {col} are: {upper_outliers}")
                    print (f"Lower bound outliers of {name}'s {col} are: {lower_outliers}")
                    print (f"The minimum outlier of {name}'s {col} is: \n {min_row}")
                    print (f"The maximum outlier of {name}'s {col} is: \n {max_row}")
                    ax.plot(data_outliers.index, data_outliers)
                    ax.set_title(f"Outliers of {name}'s {col}")
            plt.tight_layout()
            plt.show()
            plt.close('all')

    def box_plot(self):
        ''' Takes stock data and returns and box plots
      #  Input:
      #  stock_data (Dictionary): a dictionary of stock data
      #  Returns:
       # plot: a box plot of the stock
      #  '''
        fig, axes = plt.subplots(3, 2, figsize=(10, 8))
        axes = axes.flatten()
        for name,data in self.stock_data.items():
            fig, axes = plt.subplots(3, 2, figsize=(10, 8))
            axes = axes.flatten()
            for ax, col in zip(axes,data):
                sns.boxplot(data=data[col],ax=ax)
                ax.set_title(f"Box Plot of {name}'s {col}")
            plt.tight_layout()
            plt.show()
        plt.close('all')

    def mahalanobis_distance(self):
        ''' Takes stock data and returns mahalanobis distance for stock prices of a specific date
    stock_data (Dictionary): a dictionary of stock data and their respective names
    Returns:
    DataFrame: a DataFrame of mahalanobis distance
    '''
        for name,data in self.stock_data.items():

            #Calculate the mean
            mu = np.mean(data,axis=0)
            sigma = np.cov(data.T) #Singular Matrix since there is multicollinearity, The matrix does not have an inverse

            # Add a small constant to the diagonal elements of the covariance matrix to solve the problem (ridge regression)
            sigma += np.eye(sigma.shape[0]) * 1e-6

            # Calculate the covariance matrix of the distribution
            p = 6  # number of variables
            alpha = 0.05  # value of alpha (significance level)
            n = data.shape[0]  # number of observations
            alpha_level = alpha / n  # the alpha level with Bonferroni correction to control the family-wise error rate while performing
            # multiple tests, hence, minimise Type I errors (false positive) for a smaller sample data
            dfn = p - 1  # degree of freedom for the numerator
            dfd = n - dfn  # degree of freedom for the denominator
            f_critical_value = f.ppf(1 - (alpha_level), dfn, dfd)  # calculate the critical value of f-distribution
            chi_squared_critical = stats.chi2.ppf(1 - (alpha_level), data)
            cut_off = (p * (n - 1) ** 2) * f_critical_value / (n * (n - p - 1 + p * f_critical_value))
            for row in data.index:
                point = data.loc[row]
                mahalanobis_dist = mahalanobis(point,mu,np.linalg.inv(sigma))
                if mahalanobis_dist > cut_off:
                    print(f"We have evidence that {name} might have potential outliers of \n {point} since the mahalanobis distance is {mahalanobis_dist}")
                else:
                    pass;



Outliers(stock_data).iqr_method()
Outliers(stock_data).box_plot()
Outliers(stock_data).mahalanobis_distance()

#PCA without outliers:
class PrincipalComponent:
    def __init__(self,stock_data):
        self.stock_data = stock_data
    def principal_analysis(self):
        '''Takes a dictionary of stock data and returns values of explained variance ratios for
        principal components, scatter plots of principal components and dataframes of loadings
        Input:
        stock_data(dictionary): a dictionary containing the stock data and their names

     Returns:
     string: explained variance ratios for principal components of stock data
     dataframe: loadings of principal components of stock data
     graph: scatter plots of principal components of stock data
    '''
        for name,data in self.stock_data.items():
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

    def scatter_plot(self):
        '''
        Takes stock_data and returns a scatter plot of principal components
        Input:
        stock_data (DataFrame): Dataframe with stock data and their names
        Returns:
        DataFrame: a DataFrame of principal components
        graph: a scatter plot of principal components
        '''
        for name, data in self.stock_data.items():
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
PrincipalComponent(stock_data).principal_analysis()
PrincipalComponent(stock_data).scatter_plot()



def regression_analysis(stock_reset_index):

    for col in stock_reset_index.items():




#class TimeSeriesAnalysis:
#   def __init__(self,stock_data):
 #      self.stock_data = stock_data
  #  def simple_moving_average(self):

























