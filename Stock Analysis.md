***Missing values***
- MSFT: no missing values
- AMZN: no missing values
- APPL: no missing values
- SAP: no missing values
- META: no missing values
- OO5930_KS (Samsung): no missing values
- INTC: no missing values
- IBM: no missing values
- ORCL: no missing values
- BABA: no missing values
- TCEHY:no missing values
- NVDA: no missing values
- TSM ( Taiwan Semiconductor Manufacturing Company Limited): no missing values
- NFLX: no missing values
- TSLA: no missing values
- CRM: no missing values
- ADBE: no missing values
- PYPL: no missing values
*** Duplicated values ***
- MSFT: none
- GOOG: none
- AMZN: none
- APPL: none
- SAP: none
- META: none
- 005930_KS: none
- INTC: none
- IBM: none
- ORCL: none
- BABA: none
- TCEHY: none
- NVDA: none
- TSM: none
- NFLX: none
- TSLA: none
- CRM: none
- ADBE: none
- PYPL: none

***Outliers***
- The formula of detecting outliers using the IQR depends on statistic dispersion and the characteristics of a box plot. 
We know lower and upper bounds as the "outlier gates"
- It is based on the properties of the normal distribution and the desire to classify as outliers any extreme data points, typically far from the median.
a) Q3 situates at approx 0.675 SD from the median for a normal distribution.
b) The IQR (Q3 - Q1) represents approx 1.35 SD.
c) Adding 1.5 * IQR to Q3 ( or subtracting it from Q1) is therefore approx equivalent to defining outliers as points
that are more than 2.7 SD from the median. 
d) In a normal distribution, approx 0.7% of data falls more than 2.7 SD from the mean. So, this will classify only the most
extreme 0.7% of the data as outliers. 
e) In different fields, we may use different multipliers instead of 1.5, depending on how inclusive or exclusive they want
the outlier definition to be. 
- Outliers:
MSFT: none
GOOG: none
AMZN: none 
AAPL: none
SAP: none
META: 16
005930_KS: none
IBM: 82
ORCL: 49
BABA: none
TCEHY: 15
NVDA: 143
TSM: none
NFLX: none
TSLA: none
CRM: none
ADBE: none
PYPL: 270
Outliers in the 'Close' prices may occur for various reasons: market volatility, significant news or events, earning reports (investors
use earnings reports to gauge a company's future profitability and adjust their investment decisions accordingly earnings per share is a common
ratio investors use to assess a company's profitability, P/E (price to earnings) ratio => calculated by dividing the share price of a company
by its EPS, a high P/E ration compared to others in the same industry may indicate that the company is overvalued, while a low P/E could indidate that
the company is undervalued, earnings yield is an inverse of the P/E ratio), changes in market sentiments, data entry errors.
Highly unlikely in this case that there will be data entry errors. 
Researchers often view outliers as 'data problems', and they tend to overlook the fact that outliers can be substantively interesting
and studied as unique phenomena that could lead to novel theoretical insights. There is, therefore, a need for a better understanding
and clear guidelines regarding the following 3 issues: a) How to define them. b) how to identify them, c) how to handle them. 
How to best handle outliers:
https://journals.sagepub.com/stoken/default+domain/10.1177/1094428112470848/full

https://www.tandfonline.com/doi/pdf/10.1080/23322039.2022.2066762

- Meta: outliers from the period of 2021-07-26 till 2021-09-09 =>  The Dow Jones Industrial Average also experienced some changes during this period
(could be due to economic indicators, geopolitical events, and changes in investor sentiment) => maybe Covid-19, more people relying on online communication?
Amid ongoing concerns about its struggles to adequately protect data and limit hate speech, 
misinformation and other disreputable content, the world’s largest social network confronted a flood of issues this year,
beginning with the Capitol insurrection and its subsequent decision to indefinitely suspend then-President Trump. 
Troves of documents later leaked by former Facebook employee turned whistleblower Frances Haugen revealed more damaging information about the impact of the company’s platforms on young users’ mental health. 
Finally, Facebook announced that it was rebranding itself 
as Meta to reflect a focus on the metaverse, a virtual reality space where users interact with each other amid a computer-generated environment.
But the outliers occurred before its rebranding. 
Why outliers during this period? => natural variation or something else? (need more investigation)
- IBM: outliers occurred from 2014-01-07 till 2020-03-23. 2014-2017: Bull Market Continues: The stock market, as represented by indices like the Dow Jones Industrial Average (DJIA) and the S&P/ASX 200
generally trended upwards during this period. => corresponding to IBM's upper bound outliers.
2018: Increased Volatility: The year 2018 saw increased market volatility, with the DJIA and other indices experiencing several notable dips. 
This was due in part to concerns about global trade tensions and changes in monetary policy
2019: Market Recovery: Despite the volatility in 2018, the market bounced back in 2019, with the DJIA and other indices reaching new all-time highs
2020: COVID-19 Impact: The global outbreak of the COVID-19 pandemic in early 2020 had a significant impact on the stock market. 
In March 2020, the DJIA and other global indices experienced dramatic drops, marking the end of the bull market that had begun in 2009 => corresponding to the lower bound outliers from 2020-03-16 till 2020-03-23 


***Correlation***
- For all the stocks, Open, High, Low, Close and Adj Close are all correlated to each other (1). 
There are no correlations between Open, High, Low, Close, Adj Close with Volume. 

***Q-Q Plots***
- None of the stock data follow a normal distribution

*** Linear relationships***
- Seaborn pair plots show that in all companies, "Open", "High","Low",Close" and "Adj Close" all show 
linear relationships with each other. No strong relationships between "Volume" and the rest of other features exhibited (need 
to calculate correlation coefficients to make sure)