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
  'Volume': measuring trading activity rather than price-related variables 
      - MSFT: none for 'Open','High','Low','Close' and 'Adj Close'. But there are 145 outliers for 'Volume' from 2013 to 2023. High trading volume since
  MSFT has been actively traded on various stock exchanges and has had a status as a blue-chip tech company within a large market capitalisation. Factors affecting this:
  earnings reports, product announcements, market sentiment, news and events. Microsoft's trading vol has grown significantly over the years => transforming from a software company to a
  diversified tech giant => surging in investor interest => higher vol in trading. High vol trading ensures liquidity (ease with which traders and investors can buy and sell shares without
  significantly affecting the stock price) and market depth (crucial for traders and investors). However, according to my PCA => vol doesn't affect the overall data as much as price-related variables (especially with "Open") => we 
  might have evidence for this based on our outlier analysis => trading events & news, market sentiment and liquidity, algorithmic trading and high-frequency traders, earnings reports and dividends, 
  technical breakouts and breakdowns (technical traders watch for breakouts(prices moving above a resistence level) or breakdowns(price falling below a support level)), market orders and limit orders, 
  institutional activity (large institutional investors can significantly impact vol)
  The minimum outlier of MSFT's Volume is: 53336500 on 2016-07-19
  The maximum outlier of MSFT's Volume is: 202522400 on 2014-09-19
        - GOOG: none for 'Open', 'High','Low', 'Close' and 'Adj Close'. However, there are 186 outliers in Volume. High trading vol since GOOG is a tech giant, market leadership, it has positive earnings 
  reports, innovation and product launches, acquisition and strategic moves, market sentiment and news, algorithmic trading and institutional activity, liquidity and market depth, technical analysis and trends,
  Google's impact on the economy, investor behavior and sentiment. However, according to my PCA => vol doesn't affect the overall data as much as price-related variables (especially with "Open") => we 
  might have evidence for this based on our outlier analysis. Outliers in 'Volume' but not in others => trading events and news, market sentiment and liquidity, algorithmic trading and high-frequency traders,
  earnings reports and dividends, contextual investigation (extreme values in a very meaningful way).
  The minimum outlier of GOOG's Volume is: 61210000 on 2016-04-21
  The maximum outlier of GOOG's Volume is: 223486554 on 2014-01-31
          - AMZN: none for 'Open', 'High','Low', 'Close' and 'Adj Close'. However, there are 137 outliers in'Volume'(measuring trading activity rather than price-related variables). High trading vol since AMZN draws the active participation of investors in buying and selling 
  AMZN shares, growth, diversification, market dominance => sustained trading activity. However, according to my PCA => vol doesn't affect the overall data as much as price-related variables (especially with "Open") => we 
  might have evidence for this based on our outlier analysis. Outliers in 'Volume' but not in others => earnings reports, product launches, market sentiment shifts, liquidity and market depth, algorithmic trading & institutional activity, contextual investigation. 
 The minimum outlier of AMZN's Volume is: 151522000 on 2018-12-07
 The maximum outlier of AMZN's Volume is: 477122000 on 2015-01-30
        - AAPL: none for 'Open', 'High','Low', 'Close' and 'Adj Close'. However, there are 127 outliers in 'Volume'. => High trading vol ensures liquidity => News & events, earnings reports, and market sentiment drive.  
However, according to my PCA => vol doesn't affect the overall data as much as price-related variables (especially with "Open"). Outliers in 'Volume' but not in others => earnings reports, product launches, market 
sentiment shifts, liquidity and market depth, algorithmic trading and institutional activity, technical analysis & breakouts, contextual investigation (meaningful extreme values)
The minimum outlier of AAPL's Volume is: 297141200 on 2015-09-18
The maximum outlier of AAPL's Volume is: 1065523200 on 2014-01-28

          - SAP: none for 'Open', 'High', 'Low', 'Close', 'Adj Close'. There are 123 outliers for 'Volume'. High trading vol due to market news and events, price movements, trend confirmation, algorithmic trading, innovation(?), customer loyalty(?),
skilled workforce(?), acquisitions(?),advertising (?), price reductions (?) 
          The minimum outlier of SAP's Volume is: 1948500 on 2018-01-29 
          The maximum outlier of SAP's Volume is: 11290200 on 2020-10-26 
            - META: there are 12 outliers in 'Open', 12 outliers in 'High', 14 in 'Low', 16 in 'Close', 16 in 'Adj Close' and 174 in 'Volume'. High 'Open','High','Low', 'Close' and 'Adj Close' prices and  high 'Volume ' trading => due to user engagement, analyst ratings, major change in management, algorithmic trading, market trends and investor sentiment. 
             Outliers in price-related variables have Date very similar to each other.
           More outliers in 'Volume' than in any other variables.
           Outliers exist due to market volatility, trading volume, corporate actions, technical factors (including algorithmic trading) 
            The minimum outlier of META's Open is: 374.559998 on 2021-07-28
            The maximum outlier of META's Open is: 381.679993 on 2021-09-13
            The minimum outlier of META's High is: 377.549988 on 2021-07-28
            The maximum outlier of META's High is: 384.329987 on 2021-09-01
            The minimum outlier of META's Low is: 367.670013 on 2021-09-15
            The maximum outlier of META's Low is: 378.809998 on 2021-09-01
            The minimum outlier of META's Close is: 
            
005930_KS: none for 'Open','High','Low', 'Close' and 'Adj Close'. There are 132 outliers in 'Volume'. High trading volume perhaps due to new trends and large shareholder interest, recent news,
earnings reports, or changes in the industry. Outliers occur due to block sales (institutional investors), earnings reports and forecasts, market news, overall market trends 
INTC: none for 'Open', 'High', 'Low', 'Close' and 'Adj Close'. There are 120 outliers in 'Volume'. High trading volume perhaps due to market capitalisation, liquidity, company news and events,
industry trends, investor sentiment (towards tech industry and stock market). Outliers exist due to news and announcements, market trends, investor sentiment, institutional trading 
IBM: 80 for 'Open' (74 in upper bound (from 21/01/2014 to 06/10/2014), 6 in lower bound(from 16/03/2020 to 24/03/2020)), 68 for 'High'(66 in ), 90 for 'Low', 82 for 'Close', 45 in 'Adj Close' and 171 in 'Volume'. 
Stock prices of IBM have been volatile because of business performance and earnings reports (especially if differ from analysts' expectations), industry trends and market conditions (transitioning to cloud computing => improve its ability to compete),
economic factors (for example, changes in interest rates), investor sentiment (news, rumor and market psychology), high trading volume, corporate actions (decisions made by IBM => mergers and acquisitions, divestitures or changes in strategic direction). High 
trading volume due to: financial results, market capitalisation, investor interest, index funds, stock price movement. Outliers exist due to: corporate actions, financial results, market news and events, high trading vol, incomplete orders
ORCL: 64 for 'Open' (all in upper bound from 2023-13-06 till 04-12-2023),51 in 'High' (all in upper bound from 2023-06-12 till 2023-12-01), 71 in 'Low' (all from 2023-06-12 till 2023-12-11), 
49 in 'Close' (all in upper bound from 2023-06-12 till 2023-12-01), 17 in 'Adj Close'(all in upper bound from 2023-06-14 till 2023-09-11), 123 in 'Volume' (all from 2013-12-19 till 2023-12-15). 
High stock prices and trading volume due to strong financial performance, growth in cloud business, positive investor sentiment, high trading vol, market trends. High trading vol due to strong financial performance,
positive investor sentiment, market trends, earnings reports, corporate actions. Outliers exist because earnings reports, corporate actions, market news and events,
high trading volume, investor sentiments
BABA: none for 'Open', 'High', 'Low', 'Close' and 'Adj Close'. There are 126 outliers in 'Volume'. A history of high trading volume due to a high level of activity and 
interest in Alibaba's stock, institutional buying (volume, and its relationship with price, is one of the primary considerations an investor should bear in mind. For a price increase
to be genuinely meaningful, it is vital that the stock being bought in heavy volume since it is a crucial indicator of institutional buying), strong financial performance, large user base, market sentiment.  
Outliers in 'Volume' due to: institutional buying, earnings reports, company news, market sentiment, company restructuring
TCEHY: 14 in 'Open'(all in upper bound), 15 in 'High' (all in upper bound), 13 in 'Low' (all in upper bound), 15 in 'Close' (all in upper bound),
13 in 'Adj Close' (all in upper bound), 74 in 'Volume' (all in upper bound). A history of high ‘Open’, ‘High’, ‘Low’, ‘Close’, ‘Adj Close’ and ‘Volume’ in its stock prices due to strong financial performance,
institutional buying, market sentiment, company news, after-hours trading. Outliers due to: market sentiment, company news, regulatory changes (?), 
financial performance (?)
NVDA: 143 in 'Open'(all in upper bound), 143 in 'High' (all in upper bound), 143 in 'Low' (all in upper bound), 143 in 'Close' (all in upper bound),
143 in 'Adj Close'(all in upper bound), 112 in 'Volume' (all in upper bound). Outliers in 'Open','High','Low', 'Close' and 'Adj Close' are all around the same period and with seemingly similar values.
(from 2023-05-25 to 2023-12-18). History of high stock prices and trading volume due to: strong financial performance, institutional buying, market sentiment, company news, after-hours trading. Outliers due to financial
performance, market sentiment, company news, after-hours trading. Outliers in price-related variables due to: investment in AI chips, strong financial performance, Analysts' price target updates => investor sentiment, trillion-dollar valuation.
Outliers in Volume due to: product launches and innovations => investor interest and trading activity, earnings announcements, market news and events (regulations, competitor actions, macroeconomic events), analyst ratings  

TSM: none for 'Open', 'High','Low','Close', 'Adj Close', 102 for Volume(all upperbound). High trading volume due to: earnings announcements, market news and events （changes in regulations, competition actions, or macroeconomic events). 
Outliers due to: product launches and innovations (3-nanometer tech in 2024, 5-nanometer and 7-nanometer techs in the same period, innovations powering AI with silicon leadership, expansion in Arizona in 2024), earning announcements (Consolidated revenue: NT$625.53 billion (2022 Q4 Earnings Results: approximately $19.93 billion, Net profit margin: 47.3%, Shipments of 5-nanometer accounted for 32% of total wafer revenue; 7-nanometer accounted for 22%;
2023 Q4 Earnings Results:Net Revenue: $19.62 billion), market news and events (trade policies including US-China trade war, environmental regulations,inflation peaked in US and Europe in 2023, economic global growth forecasted to remain well below its historical avg over the next 2 yrs; 
competitor innovations: competitors made significant product launches and innovations => Intel's AI Everywhere Launch, Intel Innovation 2023 (Xeon processors),AI Everywhere Launch, Samsung's Galaxy Unpacked 2023,Galaxy Unpacked 2023, IAA 2023: Samsung highlighted its latest automotive semiconductor innovations, including the advanced Exynos Auto V920 automotive SoC and ISOCELL 1H1 with CornerPixel™ technology vision sensors for advanced driver visibility, CES 2023 Innovation Awards: Samsung won 46 CES 2023 Innovation Awards from the Consumer Technology Association), 
analyst ratings (in 2024 Barclays maintained an Overweight rating + Barclays maintained an Overweight rating + Needham reiterated a Strong Buy rating + 
Susquehanna maintained a Positive rating, in 2023  Needham reiterated a Strong Buy rating + Needham maintained a Strong Buy rating + Susquehanna maintained a Buy rating, in 2022 On July 14, 2023, Susquehanna maintained a Buy rating + Susquehanna maintained a Buy rating)
NFLX: none for Open, High, Low, Close, and Adj Close. 133 for Volume (all upper bound). High trading volume due to: earnings announcements (NFLX’s revenue was $33.723B For the twelve months ending December 31, 2023,
For the twelve months ending December 31, 2023, NFLX’s annual revenue for 2023 was $33.723B,
NFLX’s annual revenue for 2022 was $31.616B,NFLX’s annual revenue for 2021 was $29.698B,  
NFLX’s net profit margin for the three months ending December 31, 2023 was 16.04%
The net profit margin for NFLX stock as of December 31, 2023 was 16.04%,
The net profit margin of 18.42% is greater than its 5-year average of 13.28%), 
product launches and innovations (Expansion into New Markets almost all countries worldwide by 2016,introduction of Download Feature in 2016,
Investment in Original Content,introduction of Mobile-Only Plan in 2019, introduction of Top 10 Lists in 2020,
introduction of Top 10 Lists in 2020)
, market news and events (the repeal net neutrality in 2017 in US,Data Protection and Privacy Laws in Europe in 2017, Rise of Competing Streaming Services: The launch of several major streaming services, including Disney+ in 2019, Apple TV+ in 2019, and HBO Max in 2020;
content wars,COVID-19 Pandemic, economic fluctuations (inflation and labor markets) analyst ratings. Outliers due to: earnings announcements, analyst ratings, news events, 
market trends (investments in tech stock), stock splits (actions taken by company to divide stock into multiple shares) and dividends (dividend paid out as normal if stock split after the record date, if not per-share
price will be adjusted). 
=>  it’s worth noting that in 2014, Netflix added a record 13 million new members, with 4.33 million coming aboard in Q4. However, 2014 was a significant year for the tech industry with several high-profile buyouts,
the emergence of smartwatches and other wearables, and controversies and security disasters involving companies like Apple, Google, Uber, and Sony. These events could have influenced the stock market and trading volumes of tech companies throughout the year.
In 2015, Netflix had a significant year with a record addition of new members. The company added a record 13 million new members, with 4.33 million coming aboard in Q41. This was the year known as the Year of the Netflix Original2. Netflix had been quietly building teams and buying out certain contracts to gain exclusive TV series for its users。
 2015 was a significant year with several high-profile events. Apple, Facebook, and Google dominated the tech headlines, along with self-driving car technology, security concerns, women in technology, and gadgets like the Apple Watch, iPhone 6S, drones, and 360-view spherical cameras.
Global Expansion: On January 6, 2016, Netflix launched its service globally, simultaneously bringing its Internet TV network to more than 130 new countries around the world.
Original Content: Netflix continued to invest heavily in original content. The company debuted 15 new original series in 2016. As for the tech industry, 2016 was a year of significant events:

Samsung’s Galaxy Note7 Crisis: Samsung’s Galaxy Note7 smartphones started exploding due to battery issues, leading to a global recall56.
Facebook’s Fake News Controversy: Facebook became embroiled in a controversy over the spread of fake news on its platform56.
Advancements in Self-Driving Cars: Self-driving car technology made significant strides, with companies like Google, Tesla, and Uber investing heavily in this area.
In 2017, Netflix had a significant year with several notable events:

Global Expansion: On January 6, 2017, Netflix launched its service globally, simultaneously bringing its Internet TV network to more than 130 new countries around the world.
Original Content: Netflix continued to invest heavily in original content.
As for the tech industry, 2017 was a year of significant events:

Samsung’s Galaxy Note7 Crisis: Samsung’s Galaxy Note7 smartphones started exploding due to battery issues, leading to a global recall3.
Facebook’s Fake News Controversy: Facebook became embroiled in a controversy over the spread of fake news on its platform3.
Advancements in Self-Driving Cars: Self-driving car technology made significant strides, with companies like Google, Tesla, and Uber investing heavily in this area.


TSLA: none for 'Open','High', 'Low', 'Close', 'Adj Close' and 176 for Volume (all in upper bound). High trading volume due to: earnings reports, product announcements, 
executive news (Elon Musk), macro economic news (electric vehicle subsidies, changes in trade policy, macroecon indicators： Supply Chain Issues， economic upheavals after 2020, price cuts of up to 20% in Europe and US in 2023, global expansions in 2022 (especially in China)), analyst ratings. 
2013: Tesla had already launched the Model S in 2012, and in 2013, the company continued to ramp up production and deliveries of this model.
2014: Tesla announced plans for its “Gigafactory” in Nevada, which would produce batteries for its vehicles.
2015: Tesla launched the Model X, its luxury electric SUV.
2016: Tesla unveiled the Model 3, its more affordable electric sedan.
2017: Tesla started deliveries of the Model 3.
2018: Tesla announced the new Roadster and the Semi.
2019: Tesla launched the Model Y, a compact SUV.
2020: Tesla announced the Cybertruck, its electric pickup truck.
2021: Tesla began deliveries of the Model Y.
2022: Tesla announced that it would finally deliver its promised Cybertruck, Roadster, and Semi.
2023: the first industrial deployment of an acid-free lithium refining route. This process eliminates the use of hazardous reagents and byproducts in favor of more inert options.

Outliers due to: 2014: Tesla announced plans for its “Gigafactory” in Nevada,
2015: Tesla launched the Model X, its luxury electric SUV
2016: Tesla unveiled the Model 3, its more affordable electric sedan
2017: Tesla started deliveries of the Model 3
2018: Tesla announced the new Roadster and the Semi
2019: Tesla launched the Model Y, a compact SUV
2020: Tesla announced the Cybertruck, its electric pickup truck
2021: Tesla began deliveries of the Model Y
2022: Tesla announced that it would finally deliver its promised Cybertruck, Roadster, and Semi1.
2023: Tesla broke ground on a new lithium refinery in Texas.
Regulatory News: Changes in government regulations or policies can have a significant impact on Tesla’s business. For example, changes in electric vehicle subsidies, emission standards, or trade policies can affect Tesla’s sales and profitability, leading to movements in the stock price and trading volume
Regulatory News:

In 2023, Tesla’s regulatory credits generated an additional $1.79 billion in revenue1. The money Tesla makes due to other carmakers needing assistance to reach emissions standards continues to accumulate1.
Tesla faced multiple challenges from regulators over various issues, ranging from steering wheels falling off to software updates affecting braking methods
Market Conditions:

In 2023, Tesla shares plunged more than 14% on growing worries about weakening demand and logistical problems that have hampered deliveries3.
Tesla’s 2022 financial data impressed on every metric, but executives took a more cautious approach toward 2023, while Elon Musk continued to hint at a “deep recession” coming
Technological Innovations:

Tesla’s biggest claim to fame is the electrification of its fleet5. The cornerstone of Tesla’s success lies in its battery technology5.
Tesla’s Autopilot feature is another technological marvel5. With continued software updates, Tesla is inching closer to achieving its goal of full self-driving capabilities.
Investor Sentiment:

As of when this article was written, Tesla had a total market capitalization of $547 billion, making it the seventh most valuable company in the world6.
Tesla short-sellers watched their bets against the electric carmaker implode to the tune of $12.2 billion in 2023.
CRM: none for 'Open', 'High', 'Low', 'Close', 'Adj Close' and 169 for Volume (all upper bound). High trading volume due to: 
 earnings reports, changes in analyst ratings, news events, and overall market trends. 
uld have affected Salesforce’s (CRM) trading volume:

2014: Salesforce announced the launch of its new analytics cloud, which was the first cloud platform designed for Wave, the Salesforce Analytics Cloud1.
2015: Salesforce launched the IoT Cloud, powered by Thunder1.
2016: Salesforce attempted to acquire LinkedIn, but was outbid by Microsoft1.
2017: Salesforce announced a partnership with IBM to leverage artificial intelligence for enhanced customer service1.
2018: Salesforce acquired MuleSoft, a platform for building application networks1.
2019: Salesforce completed its acquisition of Tableau, a leading analytics platform1.
2020: Salesforce announced Work.com to help businesses and communities safely reopen during the COVID-19 pandemic1.
2021: Salesforce acquired Slack, a business communication platform1.
2022: Salesforce launched new innovations across the Customer 360 Platform to make it easier for IT and business users to build apps and experiences1.
2023: Salesforce announced strong fourth quarter and full fiscal 2023 results.

Cloud Adoption: The adoption of cloud services has been growing steadily. In 2023, it was reported that 94% of companies were using cloud services。
Digital Transformation: Digital transformation has been a key trend during this period. In 2023, it was reported that while 89% of large companies globally had a digital and AI transformation underway, they had only captured 31% of the expected revenue lift and 25% of expected cost savings from the effort
AI and ML in Business: The use of AI and ML in business has been growing. In 2023, it was reported that one-third of survey respondents said their organizations were using generative AI tools regularly in at least one business function。
Salesforce Acquisitions: Salesforce made several significant acquisitions during this period. Some of the notable ones include the acquisition of MuleSoft in 2018, Tableau in 2019, and Slack in 2021. In 2023, they also acquired Airkit, an AI-based agent that can interact directly with customers, and Spiff, a commission management platform。
Economic Conditions: The economic conditions during this period also had a significant impact. For example, in 2023, the Australian economy experienced a period of below-trend growth, and GDP per capita declined6. These conditions could have influenced the trading volume of stocks, including Salesforce’s CRM stock。 
Outliers due to: earnings reports, product announcements, analyst ratings, macroeconomic news, digital transformation, cloud adoption, AI and ML in business.

Macroeconomic events:
2014: The global cloud market started to mature, and more businesses began to adopt cloud-based solutions1. As a leading provider of cloud-based CRM solutions, Salesforce likely benefited from this trend.
2015: Digital transformation became a key focus for many businesses, leading to increased demand for Salesforce’s services1.
2016: The adoption of AI and machine learning in business processes began to grow, potentially benefiting Salesforce due to their investments in these areas1.
2017: Regulatory changes in data privacy, such as the introduction of GDPR in Europe, could have impacted Salesforce’s operations and, in turn, its stock volume1.
2018: Economic conditions, such as the US-China trade war, could have influenced investor sentiment and trading volumes1.
2019: The COVID-19 pandemic began, leading to increased demand for cloud services and digital transformation solutions as businesses shifted to remote work1.
2020: The economic impact of the COVID-19 pandemic could have affected investor sentiment and trading volumes1.
2021: Salesforce acquired Slack, a significant event that likely impacted its stock volume1.
2022: Salesforce launched new innovations across the Customer 360 Platform, potentially leading to increased trading volume1.
2023: Salesforce announced strong fourth quarter and full fiscal 2023 results

ADBE: none for 'Open', 'High', 'Low', 'Close', 'Adj Close', 140 for 'Volume' (all upper bound). History of high trading volume:
earnings reports, changes in analyst ratings, news events, product announcements, changes in stock prices (buy low sell high, price volatility,stop orders and limit orders, psychological factors),   and overall market trends. 
Adobe MAX 2023: Adobe unveiled the latest version of Adobe Creative Cloud at Adobe MAX in Los Angeles, with more than 100 new features across Photoshop, Illustrator, Premiere Pro, and beyond, including new generative AI features powered by three new foundational Adobe Firefly models for images, vectors, and design1.
Adobe Firefly: Adobe introduced the next generation of Firefly’s image generation capabilities with a massively improved model. Available now in beta in the Adobe Firefly web app, Adobe Firefly Image 2 Model significantly advances creator control and image quality1.
Adobe Workfront Enhancements: Adobe Workfront introduced several enhancements in the Fourth Quarter 2023 release. These enhancements included the option to move your organization to monthly releases, the availability of the $$USER wildcard in calculated custom fields and external lookup fields on the new form designer, and the addition of value options from an external API to a custom form.
Adobe announced new features at no additional cost for members who are already on the Premium plan.
Adobe introduced a new credit-based model across all Creative Cloud subscription plans.
Adobe announced the availability of Adobe Firefly and Adobe Express.
Adobe introduced the new request signatures experience for Acrobat for teams license users.

In the past three months, 7 analysts have released ratings for Adobe, presenting a wide array of perspectives from bullish to bearish.https://markets.businessinsider.com/news/stocks/key-takeaways-from-adobe-analyst-ratings-1033151058
Ratings for Adobe were provided by 13 analysts in the past three months, showcasing a mix of bullish and bearish perspectives.
The current estimate for Q2 2024 is $4.39, which is the same as it was 1 month ago and 3 months ago.
The current estimate for Q3 2024 is $4.48, which is the same as it was 1 month ago but slightly lower than it was 3 months ago ($4.51).
The current estimate for FY 2024 is $18.03, which is slightly higher than it was 1 month ago ($18.01) but the same as it was 3 months ago.
The current estimate for FY 2025 is $20.36, which is slightly lower than it was 1 month ago ($20.35) but the same as it was 3 months ago. 

PYPL: 271 for 'Open'(upper bound),270 for 'High' (upper bound), 269 for 'Low'(upper bound), 270 for 'Close' (upper bound), 270 for 'Close'(upper bound),
270 for 'Adj Close' (upper bound), 109 for 'Volume' (upper bound). Price-related stock variables have outliers from roughly 06/08/2020 till
19/11/2021. Volume-related outliers dated from 17/05/2015 till 15/12/2023. A history of high trading volume due to:  

2013-2018:

PayPal continued to enhance its product portfolio and expand its global presence.
2019-2023:

PayPal introduced new features and services, which might have attracted investor interest1.
In the fourth quarter of 2023, PayPal’s total payment volume had grown by 14.7 percent compared to the same quarter one year before.
PayPal reported solid fourth quarter results in 2023, with revenue increasing 9% to $8.0 billion.
For the full year 2023, PayPal’s revenue increased 8% to $29.8 billion.
PayPal’s GAAP EPS increased 84% to $3.84 and non-GAAP EPS increased 24% to $5.10 in 2023. 

History of high stock prices for PYPL due to:

Digital Payments Growth: More people started using their digital wallets, which was great news for PayPal. 
The company’s peer-to-peer payment service, Venmo, was a key catalyst behind the solid growth in its total payment volume

Product Enhancements and Acquisitions: Over the course of the year, PayPal continued its strategy to expand its capabilities and brand ubiquity by introducing new services1. 
In January of 2020, PayPal completed the $4 billion acquisition of Honey Science, providing consumers online shopping tools that save them money at checkout1. Later in the year, PayPal rolled out its contactless QR code technology, taking a big step to expand its addressable market to in-store payments.

Strong Financial Performance: PayPal started the year posting a modest increase in revenue of 13% year over year, excluding currency changes during the first quarter. 
But revenue growth accelerated to 25% in each of the last two quarters, with growth in total payment volume reaching a robust 38% in the third quarter. 
PayPal’s growing user base of 361 million active customer accounts has made it a digital payments juggernaut.

Market Share: In terms of market share, PayPal began the COVID-19 pandemic with more than 50% of the market share in global digital payments and that figure increased in its wake.

Outliers in trading volume due to:
Earnings Reports: companies often experience a spike in trading volume around the time they release their earnings reports.
Product Launches or major announcements: PayPal introduced new features and services during this period
Market trends: the shift towards digital payments and the increased use of online shopping could have contributed to increased interest in PayPal. 
Investor sentiment: changes in investor sentiment, driven by factors such as changes in the economic outlook, can lead to spikes in trading volume.
Strategic Investments and Acquisitions: can lead to increased in trading activity since they could potentially impact PYPL's future growth and profitability


Outliers in the 'Close' prices may occur for various reasons: market volatility, significant news or events, earning reports (investors
use earnings reports to gauge a company's future profitability and adjust their investment decisions accordingly earnings per share is a common
ratio investors use to assess a company's profitability, P/E (price to earnings) ratio => calculated by dividing the share price of a company
by its EPS, a high P/E ration compared to others in the same industry may indicate that the company is overvalued, while a low P/E could indicate that
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

  - Using Mahalanobis distance: only identified outliers for TCEY for stock prices of the date: 2021-02-18 00:00:00, the mahalanobis distance is 44.19225647535039.
  => These outliers are unlikely the results of errors in recording, coding or data collection since I obtained the data using API. We move to the second step of identifying potential interesting outliers 
          - On the 18th Feb 2021, TenCent raised 8.3 billion in the biggest offshore syndicated loan in Asia for a Chinese firm in 2016 https://www.bloomberg.com/news/articles/2021-02-19/wechat-owner-tencent-raises-its-biggest-loan-of-8-3-billion
          => might affect its stock prices positively since it used the debt for corporate purposes. A debt can be advantageous if: debt for growth(expanding operations,entering new markets,investing in research and development)
          debt for efficiency(efficiency improvements => upgrading equipment or eliminating interest expenses from debt), debt for debt (paying down more expensive debt), positive market perception (if the market perceives this as a positive move).
            The impact depend on various factors including the financial health of the company, the terms of loan and its use.
  ***Correlation***
- For all the stocks, Open, High, Low, Close and Adj Close are all correlated to each other. 
There are no correlations between Open, High, Low, Close, Adj Close with Volume. 

***Q-Q Plots***
- None of the stock data follow a normal distribution based on visual inspection of probability plots, histograms, Kolmogorov-smirnov test and Anderson-Darling test

*** Linear relationships***
- Seaborn pair plots show that in all companies, "Open", "High","Low",Close" and "Adj Close" all show 
positive correlations with each other. However, for INTC and IBM, the scatter plots between "Adj Close" and "Open", "High","Low",Close" 
do not form diagonal lines perhaps due to corporate actions. 
. No strong relationships between "Volume" and the rest of other features exhibited (need 
to calculate correlation coefficients to make sure)
- Correlation:
    Matrices with corr coeffs:
    - MSFT: strong positive correlation amongst "Open", "High","Low",Close" and "Adj Close". Very weak negative correlation between "Volume" and the rest
    of the features 
    - GOOG:  strong positive correlation amongst "Open", "High","Low",Close" and "Adj Close". Very weak negative correlation between "Volume" and the rest
    - AMZN: strong positive correlation amongst "Open", "High","Low",Close" and "Adj Close". Very weak negative correlation between "Volume" and the rest
    - AAPL: strong positive correlation amongst "Open", "High","Low",Close" and "Adj Close". Weak negative correlation between "Volume" and the rest
    - SAP: strong positive correlation amongst "Open", "High","Low",Close" and "Adj Close". Very weak negative correlation between "Volume" and the rest
    - META: strong positive correlation amongst "Open", "High","Low",Close" and "Adj Close". Very weak negative correlation between "Volume" and the rest
    - 005930_KS: strong positive correlation amongst "Open", "High","Low",Close" and "Adj Close". Very weak positive correlation between "Volume" and the rest
    - INTC: strong positive correlation amongst "Open", "High","Low",Close" and "Adj Close". Very weak negative correlation between "Volume" and the rest
    - IBM: strong positive correlation amongst "Open", "High","Low" and "Close". Weak positive correlation between "Adj Close" and "Open"/ "High"/"Low"/"Close". 
    Very weak negative correlation between "Volume" and the rest. 
    - ORCL: strong positive correlation amongst "Open", "High","Low",Close" and "Adj Close". Very weak negative correlation between "Volume" and the rest
    - BABA: strong positive correlation amongst "Open", "High","Low",Close" and "Adj Close". Very weak negative correlation between "Volume" and the rest
    - TCEHY: strong positive correlation amongst "Open", "High","Low",Close" and "Adj Close". Weak positive correlation between "Volume" and the rest
    - NVDA: strong positive correlation amongst "Open", "High","Low",Close" and "Adj Close". Very weak positive correlation between "Volume" and the rest
    - TSM: strong positive correlation amongst "Open", "High","Low",Close" and "Adj Close". Very weak positive correlation between "Volume" and the rest
    - NFLX: strong positive correlation amongst "Open", "High","Low",Close" and "Adj Close". Weak negative correlation between "Volume" and the rest
    - TSLA: strong positive correlation amongst "Open", "High","Low",Close" and "Adj Close". Very weak negative correlation between "Volume" and the rest
    - CRM: strong positive correlation amongst "Open", "High","Low",Close" and "Adj Close". Very weak positive correlation between "Volume" and the rest
    - ADBE: strong positive correlation amongst "Open", "High","Low",Close" and "Adj Close". Very weak negative correlation between "Volume" and the rest
    - PYPL: strong positive correlation amongst "Open", "High","Low",Close" and "Adj Close". Very weak negative correlation between "Volume" and the rest
    Corr Heat Map:
  - For all the stocks, Open, High, Low, Close and Adj Close are all correlated to each other. 
  There are no correlations between Open, High, Low, Close, Adj Close with Volume. 
***Principal Component Analysis***
Price-related components have more influence in our PCA than the volume-related component because: price variability（stock prices can vary significantly over time and
can lead to a higher degree of variance in price-related data that PCA would capture in the first few components), volume stability (trading volumes, while fluctuating, 
may not exhibit as much variance as prices. Therefore, they may not contribute as much to the total variance in our dataset, causing in lower-ranked principal components),
market impacts (price changes often have a more direct impact on market participants than volume changes. Therefore, price-related components might capture more of the market dynamics that PCA is trying to model)
- MSFT: - High cumulative variance => Principal components explain a lot about the data
        - PC1 may have the highest explained variance => highly influential (variance indicating relationships with price). PC2 has the 
        second highest explained variance (variance indicating relationship with Volume)
        - The rest of PCs have little impact. However, PC3 may a highly inverse relationship with 
        'Open' and a moderately positive relationship with Adj Close. PC4 may a highly inverse 
        relationship with 'Low' and a highly positive relationship with 'High'. PC5 may a highly
        inverse relationship with 'Adj Close' and a moderately positive relationship with 'Close'.
        PC6 may a moderately positive relationship with 'Close'
- Google: - High cumulative variance 
       - PC1 has the highest explained variance => highly influential (variance indicating relationships with price). PC2 has the 
        second highest explained variance (variance indicating relationship with Volume)
        - The rest of PCs have little impact. PC3 may a highly inverse relationship with 
        'Open'. PC4 may a highly inverse relationship with 'Low' and a highly positive relationship with 'High'.
        PC5 may a moderately positive relationship with 'Open', a moderately inverse relationship with 'High' 
       and 'Low'. PC6 may have a highly inverse relationship with 'Close' and 'Adj Close'
- AMZN: - High cumulative variance
        - PC1 has the highest explained variance => highly influential (variance indicating relationships with price). PC2 has the 
        second highest explained variance (variance indicating relationship with Volume)
        - The rest of PCs have little impact. PC3 may have a highly inverse relationship with 
        'Open'. PC4 may have a highly inverse relationship with 'Low' and a highly positive relationship with 'High'.
        PC5 may have a moderately positive relationship with 'Open' and a moderately inverse relationship with 'High' 
       and 'High'. PC6 may have a highly inverse relationship with 'Close' and 'Adj Close'
- AAPL: - High cumulative variance
        - PC1 has the highest explained variance => highly influential (variance indicating relationships with price). PC2 has the 
        second highest explained variance (variance indicating relationship with Volume)
        - The rest of PCs have little impact. PC3 may have a highly inverse relationship with 
        'Open' and a moderately positive relationship with 'Adj Close'. PC4 may have a highly inverse relationship with 'Low' and a highly positive relationship with 'High'.
        PC5 may have a moderately inverse relationship with 'Open' and 'Adj Close'. PC6 may have a highly positive relationship with 'Close'
  - SAP: - High cumulative variance
          - PC1 has the highest explained variance => highly influential (variance indicating relationships with price). PC2 has the 
          second highest explained variance (variance indicating relationship with Volume)
          - The rest of PCs have little impact. PC3 may have a highly inverse relationship with 'Adj Close' . PC4 may have a highly inverse relationship with 'Close' 
          and a highly positive relationship with 'Open'.
          PC5 may have a highly inverse relationship. PC6 may have a highly positive relationship with 'Close',a substantially positive relationship with 'Open',
          a substantially inverse relationship with 'High' and a substantially inverse relationship with 'Low'. 
  - META: - high cumulative variance
        - PC1 has the highest explained variance => highly influential (variance indicating relationships with price)。 PC2 has the 
          second highest explained variance (variance indicating relationship with Volume)
        - The rest of PCs have little impact. However, PC3 may have a substantially positive relationship with 'Open', a substantially inverse relationship with 'Close',
        and a substantially inverse relationship with 'Adj Close'. PC4 may have a substantially inverse relationship with 'High' and a substantially positive relationship with 'Low'. 
        PC4 may have a substantially inverse relationship with 'High' and a substantially positive relationship with 'Low'. PC5 may have a substantially positive relationship with 'Open',
        a substantially inverse relationship with 'High' and a substantially inverse relationship with 'Low'. PC6 may have a substantially inverse relationship with 'Close' and a substantially positive relationship with 'Adj Close'.
  -005930_KS: - high cumulative variance
        - PC1 has the highest explained variance => highly influential (variance indicating relationships with price)。 PC2 has the 
          second highest explained variance (variance indicating relationship with Volume)
        - The rest of PCs have little impact.However,PC3 may have a substantially positive relationship with 'Adj Close' and a substantially inverse relationship with 'Open'.
        PC4 may have a substantially inverse relationship with 'Open' and a substantially positive relationship with 'Close'. PC5 may have a substantially positive relationship with 'High'
        and a substantially inverse relationship with 'Low'.PC6 may have a substantially inverse relationship with 'Open', a substantially positive relationship with High,
        a substantially positive relationship with Low and a substantially inverse relationship with Close. 
  - INTC: - High cumulative variance
        - PC1 has the highest explained variance => highly influential (variance indicating relationships with price)。 PC2 has the 
          second highest explained variance (variance indicating relationship with Volume). PC3 has the third highest explained variance (variance indicating relationship with Adj Close)
        - The rest of PCs have little impact. However, PC3 may have a substantially inverse relationship with 'Adj Close', a substantially inverse relationship with 'Open',
        and a substantially positive relationship with 'Close'. PC5 may have a substantially positive relationship with 'High' and a substantially inverse relationship with 'Low'. 
        PC6 may have a substantially positive relationship with 'Open', a substantially inverse relationship with 'High',a substantially inverse relationship with 'Low', 
        and a substantially positive relationship with 'Close'.
  - IBM: - High cumulative variance
        - PC1 has the highest explained variance => highly influential (variance indicating relationships with price)。 PC2 has the 
          second highest explained variance (variance indicating relationship with Volume)
        - The rest may have little impact on IBM. However, PC3 may have a substantially positive relationship with 'Adj Close'. PC4 may have a substantially positive relationship with 'Open',
        and a substantially inverse relationship with 'Close'. PC5 may have a substantially positive relationship with 'High' and a substantially inverse relationship with 'Low'.
        PC6 may have a substantially positive relationship with 'Open',a substantially inverse relationship with 'High', a substantially inverse relationship with 'Low',
        and a substantially positive relationship with 'Close'. 
- ORCL: - High cumulative variance
        - PC1 has the highest explained variance => highly influential (variance indicating relationships with price)。 PC2 has the 
          second highest explained variance (variance indicating relationship with Volume)
        - The rest may have little impact. However, PC3 may have a substantially positive relationship with 'Open' and a substantially inverse relationship with 'Adj Close'.
        PC4 may have a substantially positive relationship with 'Open',a substantially inverse relationship with 'Low',a substantially inverse relationship with 'Close',
        and a substantially positive relationship with 'Adj Close'. PC5 may have a substantially positive relationship with 'High' and a substantially inverse relationship with 'Low'. 
        PC6 may have a substantially inverse relationship with 'Open',a substantially positive relationship with 'High',a substantially positive relationship with 'Low',
        and a substantially inverse relationship with 'Close'. 
- BABA: - high cumulative variance
        - PC1 has the highest explained variance => highly influential (variance indicating relationships with price)。 PC2 has the 
          second highest explained variance (variance indicating relationship with Volume)
        - The rest may have little impact. However, PC3 may have a substantially inverse relationship with 'Open', a substantially positive relationship with 'Close',
        and a substantially positive relationship with 'Adj Close'. PC4 may have a substantially inverse relationship with 'High' and a substantially positive relationship with 'Low'.
        PC5 may have a substantially positive relationship with 'Open',a substantially inverse relationship with 'High', and a substantially inverse relationship with 'Low'. 
        PC6 may have a substantially inverse relationship with 'Close' and a substantially positive relationship with 'Adj Close'. 
- TCEHY: - high cumulative variance
        - PC1 has the highest explained variance => highly influential (variance indicating relationships with price)。 PC2 has the 
          second highest explained variance (variance indicating relationship with Volume). 
        - The rest may have little impact. PC3 may have a substantially inverse relationship with 'Open' and a substantially positive relationship with 'Adj Close'.
        PC4 may have a substantially positive relationship with 'High' and a substantially inverse relationship with 'Low'. PC5 may have a substantially inverse relationship with 'Open'
        and a substantially positive relationship with 'Close'. PC6 may have a substantially positive relationship with 'Open', a substantially inverse relationship with 'High'
        and a substantially positive relationship with 'Close'. 
- NVDA: - high cumulative variance
        - PC1 has the highest explained variance => highly influential (variance indicating relationships with price)。 PC2 has the 
          second highest explained variance (variance indicating relationship with Volume). 
        - The rest has little impact. However, PC3 may have a substantially inverse relationship with 'Open', a substantially positive relationship with 'Close',
        and a substantially positive relationship with 'Adj Close'.PC4 may have a substantially positive relationship with 'High' and a substantially inverse relationship with 'Low'. 
        PC5 may have a substantially inverse relationship with 'Open',a substantially positive relationship with 'High', and a substantially positive relationship with 'Low'.  
        PC6 may have a substantially positive relationship with 'Close' and a substantially inverse relationship with 'Adj Close'. 
- TSM: - high cumulative variance
        - PC1 has the highest explained variance => highly influential (variance indicating relationships with price)。 PC2 has the 
          second highest explained variance (variance indicating relationship with Volume). 
        - The rest has little impact. However, PC3 may have a substantially positive relationship with 'Open' and a substantially inverse relationship with 'Adj Close'. 
        PC4 may have a substantially positive relationship with 'Open' and a substantially inverse relationship with 'Close'. PC5 may have a substantially positive relationship with 'High'
        a substantially inverse relationship with 'Low'. PC6 may have a substantially inverse relationship with 'Open', a substantially positive relationship with 'High', a substantially positive relationship with 'Low'
        and a substantially inverse relationship with 'Close'. 
- NFLX: - high cumulative variance
        - PC1 has the highest explained variance => highly influential (variance indicating relationships with price)。 PC2 has the 
          second highest explained variance (variance indicating relationship with Volume). 
        - The rest has little impact. PC3 may have a substantially inverse relationship with 'Open', a substantially positive relationship with 'Close',
        a substantially positive relationship with 'Adj Close'. PC4 may have a substantially positive relationship with 'High' and a substantially inverse relationship with 'Low'.
        PC5 may have a substantially positive relationship with 'Open', a substantially inverse relationship with 'High' and a substantially inverse relationship with 'Low'. 
        PC6 may have a substantially inverse relationship with 'Close' and a substantially positive relationship with 'Adj Close'. 
- TSLA: - high cumulative variance
        - PC1 has the highest explained variance => highly influential (variance indicating relationships with price)。 PC2 has the 
          second highest explained variance (variance indicating relationship with Volume). 
        - The rest has little impact. PC3 may have a substantially positive relationship with 'Open', a substantially inverse relationship with 'Close',
        and a substantially inverse relationship with 'Adj Close'. PC4 may have a substantially positive relationship with 'High' and a substantially inverse relationship with 'Low'.
        PC5 may have a substantially inverse relationship with 'Open', a substantially positive relationship with 'High' and a substantially positive relationship with 'Low'.
        PC6 may have a substantially positive relationship with 'Close' and a substantially inverse relationship with 'Adj Close'.
- CRM: - high cumulative variance
        - PC1 has the highest explained variance => highly influential (variance indicating relationships with price)。 PC2 has the 
          second highest explained variance (variance indicating relationship with Volume). 
        - The rest has little impact.PC3 may have a substantially positive relationship with 'Open', a substantially inverse relationship with 'Close',
        and a substantially inverse relationship with 'Adj Close'. PC4 may have a substantially positive relationship with 'High' and a substantially inverse relationship with 'Low'.
        PC5 has a substantially inverse relationship with 'Open', a substantially positive relationship with 'High' and a substantially positive relationship with 'Low'.  
        PC6 has a substantially positive relationship with 'Close'and a substantially inverse relationship with 'Adj Close'.  
  - ADBE: - high cumulative variance
         - PC1 has the highest explained variance => highly influential (variance indicating relationships with price)。 PC2 has the 
second highest explained variance (variance indicating relationship with Volume). 
         - The rest has little impact. PC3 has a substantially inverse relationship with 'Open' and  a substantially positive relationship with 'Adj Close'.
        PC4 has a substantially positive relationship with 'High' and a substantially inverse relationship with 'Low'. PC5 may have a substantially positive relationship with 'Open',
        a substantially inverse relationship with 'High' and a substantially inverse relationship with 'Low'. PC6 may have a substantially positive relationship with 'Close' 
        and a substantially inverse relationship with 'Adj Close'. 
  - PYPL: - high cumulative variance
        - PC1 has the highest explained variance => highly influential (variance indicating relationships with price)。 PC2 has the 
second highest explained variance (variance indicating relationship with Volume). 
        - The rest has little impact. PC3 may have a substantially positive relationship with 'Open', a substantially inverse relationship with 'Close',
        and a substantially inverse relationship with 'Adj Close'. PC4 may have a substantially positive relationship with 'High' and a substantially inverse relationship with 'Low'.
        PC5 may have a substantially positive relationship with 'Open', a substantially inverse relationship with 'High' and a substantially inverse relationship with 'Low'.
        PC6 may have a substantially positive relationship with 'Close' and a substantially inverse relationship with 'Adj Close'. 

***Multivariate Normality***
- There might not be the presence of multivariate normality in all the stock data based on Mardia's test and Henze-Zirkler's test
***Seasonal Decomposition***
- There might not be any seasonality in all stock data. Need to check more. 
***Trend***
Visual inspection
- MSFT: clear upward trend
- GOOG: clear upward trend
- AMZN: clear upward trend
- APPL: clear upward trend
- SAP: slight upward trend, not clear, plateauing
- META: clear upward trend, slight dip at the end of 2022
- 005830_KS: clear upward trend
- INTC: unclear trend, up from 2014 till 2018, plateauing from 2018 till 2021, downward from then on
- IBM: plateauing
- ORCL: clear upward trend 
- BABA: unclear trend
- TCEHY: unclear trend
- NVDA: clear upward trend
- TSM: clear upward trend
- NFLX: clear upward trend
- TSLA: clear upward trend
- CRM: clear upward trend
- ADBE: clear upward trend
- PYPL: unclear trend, upward from 2016 till mid-2021, downward from then on
***Volatility***

  