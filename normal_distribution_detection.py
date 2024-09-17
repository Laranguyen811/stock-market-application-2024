class DetectNormalDistribution:
    def __init__(self,stock_data):
        self.stock_data = stock_data
    def prob_plot(self):
        ''' Takes a DataFrame of stock data and return probability plot (Q-Q plot) and histogram for each stock.
        Input:
        stock_data(Dataframe): a Dataframe of stock data and their respective names
        Returns:
        graph: a Probability plot of stock data
        '''
        for name,data in self.stock_data.items():
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


    def histogram_plot(self):
        """ Takes a DataFrame of stock data and return histogram of each stock
        Input:
        stock_data(Dataframe): a Dataframe of stock and its corresponding name

        Returns:
        graph: A histogram of stock data
        """
        for i, (name,data) in enumerate(self.stock_data.items()):
            cols = data.columns
            fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(8,6))
            axs = axs.flatten()
            for ax, col in zip(axs,cols):
                ax.hist(data[col], bins=50) #using square-root rule to decide the number of bins
                ax.set_title(f"Histogram of {col} for {name}")
            plt.tight_layout()
            plt.show()
        plt.close('all')

    def kolmogorov_smirnov_test(self):
        """ Takes a DataFrame of stock data and return Kolmogorov-Smirnov test.
            Input:
            stock_data(Dataframe): a Dataframe of stock and its corresponding names
            Returns:
            string: A Kolmogorov-Smirnov test result of stock data
        """
        for i, (name,data) in enumerate(self.stock_data.items()):
            cols = data.columns
            for col in cols:
                d, p_value = stats.kstest((data[col] - np.mean(data[col])) / np.std(data[col],ddof=1), 'norm')
                if p_value < 0.05:
                    print(f"We have no evidence that {name}'s {col} might be normally distributed since the p-value is {p_value}.")
                if p_value >= 0.05:
                    print(f"We have evidence that {name}'s {col} might be normally distributed since the p-value is {p_value}")


    def anderson_darling_test(self):
        """ Takes a DataFrame of stock data and returns the Anderson-Darling test result.

        Input:
            stock_data(Dataframe): a Dataframe of stock and its corresponding names

        Returns:
            string: the Anderson-Darling test result of the stock data
        """
        for i, (name,data) in enumerate(self.stock_data.items()):
            cols = data.columns
            for col in cols:
                anderson_darling_result = stats.anderson(data[col], dist='norm')
                anderson_stats = anderson_darling_result.statistic
                print(f"The Anderson-Darling test statistics of {name}'s {col}: {anderson_stats}")
                for critical_value,significance_level in zip(anderson_darling_result.critical_values, anderson_darling_result.significance_level):
                    if anderson_stats > critical_value:

                        print(f"We have no evidence that {name}'s {col} might be normally distributed since the critical value is {critical_value} at {significance_level}%.")
                    if anderson_stats <= critical_value:
                        print(f"We have evidence that {name}'s{col} might be normally distributed since the critical value is {critical_value} at {significance_level}%.")

DetectNormalDistribution(stock_data).prob_plot()
DetectNormalDistribution(stock_data).histogram_plot()
DetectNormalDistribution(stock_data).kolmogorov_smirnov_test()
DetectNormalDistribution(stock_data).anderson_darling_test()