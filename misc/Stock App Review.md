
23/12/2023
**Things I learned today**
- A list can store DataFrames
- It is generally a good practice to use lists or dictionaries to store variables since they give us more 
functionality and flexibility
- We can create a for loop to go through all of the stock market
DataFrames, reset Date indices to be columns and create new variables
- The enumerate function goes through the list and tracking its iteration

4/1/2024
**Things I learned today**
- Refactoring: process of restructuring existing computer code without changing its external behavior. 
Goal: to improve the design, structure,and/or implementation of the software (its non-functional attributes) while preserving its
functionality 
Benefits: improve code readability, and reduced complexity (can enhance the source code's maintainability, creating simpler, cleaner and more
expressive internal architecture or object model to improve extensibility, discover and fix hidden or dormant bugs or 
vulnerabilities in the system since it simplifies the underlying project and diminishes unnecessary levels of complexity). 
Refactoring is 1 of the primary means of repaying technical debt
- Error message: TypeError: unhashable type :'DataFrame' normally happens when we try to use a pandas DataFrame (which is
mutable, hence unhashable) +. using DataFrame as a dictionary ky or in a set, inccorect usage of .loc or iloc, passing DataFrame to a function
expecting a hashable type

*** Things I did well ***
- Figuring out a way to write a function to perform some analysis on multiple dataframes 

*** Things I did not do well ***
- Wrote a function to perform some analysis on multiple dataframes that did not work

*** Things I will do better next time ***
- Ask Bing for some idea
- Understand why it did not work 

9/1/2023
***Things I learned today:***
- zip (): a built-in function taking 2 or more iterables and returns an iterator 
aggregating elements from each iterable. The resulting iterator contains tuples, where
the i-th tuple contains the i-th element from each of the input iterables. 
- plotly.graph_objs: a module in plotly providing a set of classes for creating and manipulating figures. 
We call these "graph objects" and represent parts of a figure. 
- No def needed when creating a plot. Simply no need to. 
- Plotly is a data visualisation library allowing you to create interactive plots and 
charts in Python. Advantages: interactive visualisations, ease of use, customisation, compatibility,
community support. In general, if your data set is small and simple, and you only need to communicate a few key insights, a static visualization might be the best choice. 
However, if your data set is large and complex, and you need to allow the user to explore the data in more detail, an interactive visualization might be a better choice.
- A Scatter object: used to create line charts, scatter charts, text charts and bubble charts. 
***Things I did well today:***
- Understanding how to write a function to take 2 or more iterables and return a iterator aggregating
elements from each iterable using zip. The function takes 2 lists from stock data and stock names to print information, shape, data types and statistical descriptions. 
- Understanding how to write a function to reset index of DateTimeIndex to a column of stock data using enumerate. 
- Learning to use Plotly to create an interactive plot since it has advantages of interactive
visualisations, ease of use, customisation, compatibility and community support. When the dataset is complex and large, we can use it.

***Things I didn't do well today:***
- When visualising stock data Price vs. Time, there were multiple plots created, not just 1 plot needed.
- Didn't write the code completely off my head

*** Things I will do better next time: ***
- Chunking code in memory and recreate it 
- Looking into why go.Scatter created multiple plots instead of one 

12/1/2024
***Things I learned today***
- Building a Dash application requires 2 parts: the first one is the layout of the app, describing what the app looks like, 
the 2nd part describe the interactivity of the app
***Things I did well today ***
- Reconstructing the code to reset all of DateTimeIndex for all stock data from memory
- Learning how to build a Dash app
***Things I did not do well today ***
- Unable to build the Dash app using dcc.Graph (deprecated)
***Things I will do better next time ***
- Figure out how to build a Dash app using dcc.Graph, perhaps checking with Amin or JetBrain support team 

15/01/2024
***Things I learned today ***
- In Dash @callback, 'my-id' and 'value' correspond to the unique identifier and current value of the component, which needing lookup
- In @callback, 'inputs' and 'outputs' are arguments of callback decorator. Whenever an input changes, the function the callback decorator wraps around will change automatically
- To detect outliers in stock data, we can use the method of Interquartile Range (IQR). It is the range between the first quartile (25 percentile) and the second quartile (75 percentile).
Any data points fall below the Q1 - 1.5*IQR and above the Q2 + 1.5*IQR will be outliers.
***Things I did well today***
- Figuring out how to build an interactive Dash visualisation app
- Figuring out how to conduct Exploratory Data Analysis
- Detecting outliers using IQR method

*** Things I did not do well today ***
- Not getting the dash app to work
- Taking too much time with the dash app

*** Things I will do better next time ***
- Only spend 1 hour on Dash app. If not working, make notes to ask
- Spend more time on data analysis 

19/01/2024
***Things I learned today***
- if plt.show() outside the loop, only show the box plot last iteration (pypl). If inside, show all the box plots
- Reasons for outliers in stock market data: market volatility, significant news or events, earnings reports, changes in 
market sentiments, and data entry errors
- Outliers, though often viewed as 'data problems' we need to fix, may offer interesting explanations that can lead to 
theoretical insights since we can study them as unique phenomena. Therefore, we need to develop clear guidelines and a better understanding
in 3 areas: a) how to define them, b) how to detect them, c) how to handle them. 

***Things I did well today***
- Writing clearer, concise and higher quality code to replace the old one
- Getting better at understanding the stock market data
- Looking into outliers, reasons for their existence and how to handle them

***Things I did not do well today***
- Not understanding how to handle outliers in stock market data yet
- Talking to Hayden while doing deep work

***Things I will do better next time***
- Asking Hayden to go out for a walk while doing deep work or simply moving to the library
- Better grasp how to handle outliers in stock market data


22/01/2024
*** Things I learned today ***
- Considerations about outliers: outliers can lead to important changes in parameter estimates when researchers 
use statistical methods relying on maximum likelihood estimator and how we deal with outliers
may lead us to falsely accept or reject hypotheses; we must use caution since in most cases, deleting 
outliers helps us support our hypotheses and opting a course of action post hoc that is certain to increase
our likelihood of finding what we want to find is a dangerous practice.
- The decisions researchers make about how to define, identify, and handle outliers have
crucial implications, because such decisions change substantive conclusions.
- 14 outlier definitions: single construct outlier, error outlier, interesting outlier, discrepancy outlier,
model fit outlier, prediction outlier, influential meta-analysis effect size outlier, influential meta-analysis
sample size outliers, influential meta-analysis effect and sample size outlier, cluster analysis outlier, influential
time series additive outlier, influential series innovation outliers, influential level shift outlier,
influential temporary changes outlier

***Things I did well today ***
- Reading and scanning papers about outliers
- Digging deeper into outliers

*** Things I did not do well today ***
- Taking more time to read. Should take about 3 - 4 hours to read

*** Things I will do better next time ***
- Speeding up my reading, perhaps can skim some information

24/01/2024
***Things I learned today***
- outlier identification techniques: box plot, stem and leaf plot, schematic plot analysis,
standard deviation analysis, percentage analysis, scatter plot, q-q plot, p-p plot, standardised residual,
studentised deleted residual, euclidean distance, mahalanobis distance, K-clustering,
2- or 30 dimesional plots of the original and PCA, autocorrection function plot, time plot, extreme studentised deviate,
Hosmer and Lemeshow goodness-of-fit test, Leverage value, centred leveraged value, deletion standardised multivariate residual,
Cook's Dt, Modified Cook's Dt, Generalised Cook's Dt,difference in fits (standardised), 
difference in beta (standardised), Chi-squared difference test, single parameter influence, average squared 
deviation technique, sample-adjusted meta-analytic deviancy, conduct anaysis with and without outliers, nearest neighbor techniques,
nonparametric methods, semiparametric methods, iterative outlier identification procedure, independent component analysis

***Things I did well today***
- Understanding outlier detection techniques in detail

***Things I did not do well today***
- Slow speed in reading the paper

***Things I will do better next time***
- Using abbrev, shortening sentences, copying 

***Things I learned today***
- Standardised residual: calculated by dividing the ith observation's residual value by a std deviation term
- Studentised residual: measuring both the outlyingness of the observation regarding its standardised residual
value and the outlyingness of the observation on the predicted value. 
- Standardised deleted residual: the predicted value for the focal observation is calculated without the observation itself
- Euclidean distance: length of the line segment between 2 specified points in different dimension space, can be calculated
from the Cartesian coords of the points using the Pythagorean theorem. 
- Mahalanobis distance: similar to Euclidean distance, the length of the line segment between a data point and the centroid
of the remaining cases, where the centroid is the point created at the intersection of the means of all the prediction vars.
A large Mahalanobis distance may mean the corresponding observation is an outlier. 
- K-clustering: yield different candidate subsets that then have to be evaluated by 1 or more multiple-case diagnostics
- 2 - or 3-dimensional plots of the original and the principal component vars: produced as a result of a principal 
component analysis. A isolated data point may indicate a potential outlier.
- autocorrelation plot: computing autocorrelations for data valyes at varying time lags. 
- time plot: relationship between a certain variable and time.
- extreme studentised deviate: different btwn a var's mean and query value, / a std value
- Hosmer and Lemeshow goodness-of-fit test: a Pearson chi-square stat from a table of observed and expected freq.

***29/01/2024***
***Things I did well today***
- Identifying outliers and factors might cause them
- Looking at different methods and techniques to detect outliers

***Things I did not do well today***
- Taking some time to read the paper

*** Things I will do better next time***
- Give myself more time to read

***31/01/2024***

***Things I did well today***
- To calculate regression coefficients:
1. Fit a linear regression model to our data.
2. Calculate the regression coefficients using:
The coefficient of X is given by the formula a = n(∑xy)−(∑x)(∑y) / n(∑x2)−(∑x)2
The constant term is given by the formula b = [(∑y)(∑x^2)−(∑x)(∑xy)] / [n(∑x2)−(∑x)2]
3. Lastly, we can input the regression coefficients in the equation Y = aX + b to obtain
the predicted vals of Y. 
- Cook's Di: assessing the influence a data point i has on all regression coefficients as a whole, 
calculated by deleting the observation from the dataset and then recalculating the regression coefficients. 
Di = [(ri^2)/(p*MSE)]/[(hii)/(1-hii)^2], where ri denotes the ith residual, p is the number of coefficients in the 
regression model, MSE is the mean sqrd error, hii is the ith leverage value
- Modified Cook's Di: using standardised deleted residuals rather than standardised residuals
Di* = (Di)/(p*MSE) * (hii)(1-hii)^2, where Di: Cook's distance, p is the number of coefficients in the regression model,
MSE is the mean squared error, hii is the ith leverage val.
Taking into acc the number of independent vars in the model. 
- Generalised Cook's Di: applied to structural equation modelling to assess the influence a data point has on
the parameter estimates.
Di^* = ri^2/(p * MSE *hii), where rii denotes the ith residual, p is the number of coeffs in the regression model,
MSE is the mean squared error, hii is the ith leverage val.
- Structural equation modelling: method used to analyse the relationships between the observed and latent vars. Combining
factor analysis and multiple regression analysis. We create a model representing how various aspects of some phenomenon
thought to causally connect to one another. Often contain postulated causal connections among some latent vars. Additional
causal connections link those latent vars to observed vars. 
- latent vars: vars we cannot directly observe or measure, but inferred from other observable vars that can be directly observed
or measured via a mathematical model, representing underlying concepts or construct we cannot directly measure.  
- Difference in fits, standardised (DFFITSi): assessing the influence a data point i has on all regression coeffs as a whole,
Diff between Cook's Di and this is that they produce info on different scales. A diagnostic measure to detect influential 
observations, measuring the diff in the predicted val for a point, obtained when that point is left out of the regression.
DFFITS is the Studentised DFFIT where we can achieve studentisation by dividing the estimated std of the fit at that point.
DFFITSi = (^yi - ^y(i))/(sqrt(MSEi x hii)), where ^yi denotes the predicted val for the ith observation, 
^y(i) the predicted val for the ith obs when we exclude it from the regression, MSE(i) the mean sqrd error when the 
ith obs is excluded from the regression, hii the ith leverage value 
Obs with DFFITS val greater than 2x the sqr root of the no of parameters / the no of obs are considered to be influential 

- Difference in beta, standardised (DFBETASij): assessing if the inclusion of a case i leads to an increase or decrease in a single regression coeff
j i.e. a slope or intercept. Measuring the difference in the estimated regression coeff for a predictor car when that obs is included or excluded 
from the regression model. 
DFBETAS(ij) = [beta_i - beta(i(-j)]/[sqrt(MSE(-i) x hii))]

- Chi-squared difference test: allowing a researcher conducting SEM to assess the diff in the model
fit between 2 models, 1 with the outlier and the other without 
To conduct a chi-squared difference test:
1. Define a null hypo and the alternative hypo
2. Calculate the expected frequencies for each category under the null hypo
3. Need to calculate the chi-squared test stat using this:
chi^2 = sum{i=1}^{k}\frac(O_i = E_i)^2}{E_i}
where:
O_i is the observed frequency for the category i
E_i is the expected frequency for category i
k is the number of categories
4. Need to compare the calculated chi-squared test stat to the critical value of the 
chi-squared distribution with k-1 degrees of freedom. If the test sta > 
the critical val => reject the null hypothesis to conclude that there is a sig diff 
between the observed and expected frequencies of the categorical var.  
- Single parameter influence: used in SEM to assess the effect of an outlier on a specific parameter 
estimate, as opposed to the overall influence of an outlier on all para estimates
Identifying which parameters have the greatest impact on the model's output. We can calculate
using various methods: DIFFITS, DFBETAS, and Cook's distance. 
A common rule of thumb: observations with a DIFFITS val greater than 2 x the sqr ropt of the no of 
parameters/the no of obs considered to be influential. Also, points with a large Cook's distance are considered
influential. 
- Avg squared deviation technique: when conducting multilevel modeling, explores the effect each group
has on the fixed &/or random parameters, allowing for the identification of higher-level prediction outliers. 
Measuring the variability of a set of data points. 
Calculated by finding the diff between each data point & the mean if the data set, squaring the differences, adding 
them together, and dividing by the no of data pts - 1. 
s^2 = (sum_{i_1}^{n}({x_i - \bar{x})^2/{n-1}
where:
s^2: the avg sqrd deviation 
s_i is the ith data pt
x_i: the ith data pt
\bar{x} is the mean of the data set
n : the no of data pts

A large avg sqrd deviation: indicates the data pts are more spread out, while a small avg sqrd deviation conveys data points are tightly clustered around 
the mean. 


- ***Things I did well today***
- Reading in-depth about outlier identification techniques

***Things I did not do well today***
- Woke up a bit late to study
***Things I will do better next time***
- Have an alarm ready

***05/02/2024***
***Things I learned today***
- Sample-adjusted meta-analytic deviancy (SAMD): in metaanalysis, it takes the diff between the val of each primary-level
effect suze estunate and the mean sample-weighted coeff  computed without that effect size in the analysis, then alternates 
the diff val based in the sample size of the primary-level study. Can use SAMD to detect outliers. Helps us to identify
studies whose effect sizes deviates significantly from the overall trend. Calculates external studentised residuals for each study.
By considering the sample size, SAMD gives us a more robust way to detect outliers and helps researchers identify studies that might be driving 
unusual patterns in the meta-analysis. 
- Conduct analysis with or without outliers: when results differ across the 2, we confirm that the data pt is indeed an outlier. Why:
sensitivity check, transparency, understanding impact.
How: document, describe approach, report the diffs
- Nearest Neighbor techniques: calculate the closest val to the query val using various types of distance metrics. 
Techniques: KNN, optimised NN, 1-NN, 2-NN, NN with reduced features, dragon method, PAM (partitioning around medoids),
CLARANS (clustering large apps based on randomised search) and graph connectivity method.

- Nonparametric methods: fitting a smoothed curved without making any constraining assumptions about the data. Lack of 
linear relationships indicates the presence of outliers. Not relying on specific assumptions about the parameters of the
data distribution. Often used when the assumptions of the parametric methods are violated. 
Charateristics: distribution-free, function on samples (defining stats as functions on samples dependency on specific parameters). 
Can use them for descriptive stats, statistical inference, modelling financial time series. 
- Semiparametric methods: combining the speed and complexity of parametric methods with the flexibility of nonparametric methods
to understand local clusters or kernels rather than a single global distribution model. We identify outliers as lying in regions of 
low density. Striking the balance of flexibility and structures.  
Components of semiparametric models: 
parametric components (representing a finite-dimensional vector) & nonparametric components (representing a infinite-dimensional vector)
Advantages: flexibility, robustness, interpretability/ 
Challenges: estimation, curse of dimensionality
ApplicationL biostats, econs, env sciences
- Iterative outlier identification procedure: allowing for the estimate of the residual std to identify
data pts sensitive to the estimation procedure used for a time series analysis. 
How:
Identify potential outliers
Remove or adjust outliers
Recompute stats
Repeat step 1-3
- Independent component analysis: separate independent component by maximising the 
statistical independence among them. We identify the separate independent components 
as outliers. 
Objective: separate a multivariate signal into additive subcomponents, 
assuming at most subcomponent is Gaussian (normal), and the subcomponents are statistically independent
from each other. 
Goal: unmix and decompose signal into its original independent sources
Key assumptions: source signals are independent of each other. Values in each source signal
have non-Gaussian distribution

Application: cocktail party problem (separate overlapping speech signals from 
multiple speakers), blind source separation (extracting individual sources from their
mixture), biomedical signal processing (separating EEG signals, fMRI data), image processing
(texture analysis, face recognition). financial data (separating market signals)
Mathematical approach: attempt to find a linear transformation of the data such that 
the xformed components are independent as possible. works well when the statistical 
independence assumption holds.
ICA goes beyond PCA by seeking statistically independent components

A good way to disentangle complex mixture and reveal hidden structures in data

***Things I did well today***
- Understanding outlier detection methods more in-depth
- Be more on-time 
- More focus
- Identifying the Iterative outlier identification procedure as a suitable procedure for this project
***Things I did not do well today***
- Syntax error with using ' ' instead of " " after f
- Not completing the analysis

***Things I will do better next time***
- using f " "
- figuring out more ways to conduct analysis to make it more complete


*** 7/2/2024***
***Things I learned today***
- Correct val: correcting a data pt to its proper val.
- Remove outlier: eliminate the data pt from the analysis.
- Study the outlier in detail: conduct follow-up work to study the case
as a unique phenomenon of interest. Analyse outlier, study its impact
- Keep: acknowledge the presence of an outlier, but do nothing prior to the
analysis
- Report findings with and without outliers: report substantive results with 
and without, include any explanation for any diff in the results. Be transparent.
- Winsorisation: transforming extreme vals to a specified percentile of the data
- Truncation: setting observed vals within a certain range, anything outside we will
remove
- Transformation: applying a deterministic mathematical function to each val.
- Modification: changing an outlier to another val, less extreme one.
- Least absolute deviation: similar to ordinary least squares (method to estimate the 
unknown parameters in regression). Better than LSD when errors follow non-Gaussian distribution with longer tails. 
- Least trimmed squares: ordering the squared residual for each case from the highest to the lowest,
then trim or remove the highest val. 
- M-estimation: a class of robust techniques reducing the effect of influential outliers
by replacing the squared residuals by another func of them. 
- Bayesian stats: obtaining parameter estimates by weighing prior info and the observed data at hand
- 2-stage robust procedure: use Mahalanobis distance to assign weights to each data pt, extreme cases
are downweighted. completed via a 2-stage process.

***Things I did well today***
- Learning more about how to handle outliers: correct val, remove outlier. study the outlier in detail,
keep, report findings with or without the outlier, winsorisation, truncation, transformation, modification, 
least absolute deviation, least trimmed squares, M-estimation, Bayesian stats, 2-stage robust procedure

***Things I did not do well today***
- Not writing any code

***Things I will do better next time***
- Writing some code: will do some PCA maybe? 

***12/02/2024***
***Things I learned today***
- Direct robust method using iteratively reweighted least squares: using Mahalanobis distance to
assign weights to each data pt. We complete the assignment of weights via using an iteratively reweighted least 
squares algo. IRLS also commonly used for robust regression, especially when dealing with outliers or heavy-tailed 
error distributions. IRLS aim to mitigate the impact of outliers by downweighting their influence during parameter 
estimation. IRLS updates the regression coeffs by reweighting the observations based on their residuals: initialisation, 
weight calculation, re-estimation, converge check, final estimates. 
- generalised estimating equations (GEE) methods: estimating the variances and covariances in the random part of the 
multilevel model directly from the residuals. Useful for analysing correlated data, allowing to acc for dependencies and estimate pop-up effects
- Multilevel models: statistical techniques used to analyse data with a hierarchical or nest structure. Level 1 model: describe indv-level relationships. 
Level-2 model: describe group-level relationships
- Boostrapping methods: estimate parameters of a model and their standard errors from the sample, without reference to a theoretical sampling distribution. 
App: statistical inference, regression models, ML
- Non-parametric methods: does not assume the data distributed in any particular way
- Unweighted meta-analysis: obtaining meta-analytic stats not giving more weight to primary level studies with large sample sizes. How: each 
study contributes equally to the overall effect estimate.
- Generalised M-estimation: a class of robust techniques reducing the effect of outliers by replacing the squared residuals by another func of the residuals.
- 3 shortcomings authors identify: 1st => it is common for organisational science researchers to either vague or not transparent on how outliers are defined
and how a particular outlier identification method chosen and used, 2nd => we identify outliers in one way but then used another outlier identification technique not 
congruent with their adopted outlier definition, 3rd => the authors found little discussion, let alone recommendations, on the subject of studying outliers interesting 
and worthy of further examination. 
- A pervasive view: outliers are problems that we should fix
- 2 principles: we should describe choices and procedures regarding the treatment outliers that we have implemented, and we should clearly and explicitly acknowledge the type
of outlier where they are interested and use an identification technique congruent with the outlier definition

***Things I did well today***
- Learning about: direct robust method using iteratively reweighted least squares, generalising estimating equations methods, boostrapping methods, non-parametric method, unweighted 
meta-analysis, generalised M-estimation, 3 shortcomings, 2 principles
***Things I did not do well today***
- Not defining an outlier yet***
*** Things I will do better next time***
- Defining outliers

***13/02/2024***
***Things I learned today***
- 3 types of outliers: error outliers, interesting outliers and influential outliers
- 2 steps of detecting error outliers: 1st step => locating outlying obs, 2nd => separately 
investigating whether the outlyingness of such data pts was caused by errors
- 2 categories of techniques to identify error outliers: single construct and multiple construct

***Things I did well today***
- Learned about types of outliers, steps of detecting error outliers and 2 categories to 
identify error outliers

***Things I did not do well today***
- Did not study for the whole 3 hours due to tiredness and tummyache

***Things I will do better next time***
- Anticipate beforehand to fix my schedule

***16/02/2024***
***Things I did well today***
- Single construct techniques: refer to the measurement of constructs expected
to have a single underlying dimension. 2 steps: conceptualisation and operationalisation. 
The recommendation is using visual tools first then follow up with quantitative approaches. Can use 
recommended cutoffs. 
- Multiple construct techniques: use them when we believe a concept or construct
to have multiple dimensions, using multiple measures or tests to capture the different
dimensions of a construct. 2 steps are similar to single construct techniques. Use
recommended cutoffs when applicable. 
- Construct: abstract concepts not directly observable
- Researchers should keep diaries, logs or journals during data collection to use 
retrospectively to decide if something unusual happened with some particular case that 
they can no longer trace after the fact. 
- The 2nd step in the process of understanding the possible presence of outliers is 
examining interesting outliers. Do not automatically treat any outlying data pts as harmful. 
- Interesting outliers: accurate data pts, identified as outlying obs ( potential error outliers)
but not confirmed as actual error outliers. These cases may contain potentially valuable or unexpected knowledge
- Identifying interesting outliers involve 2 steps: identifying potentially interesting outliers and
identifying which outliers are indeed interesting. 
- Pursuing potential interesting outliers likely will include the examination of 
a great deal of error outliers going undetected as errors.

***Things I did well today***
- Focusing on my study of detecting and handling error outliers and interesting outliers

***Things I did not do well today***
- Talking to Hayden during deep work

***Things I will do better next time***
- Informing Hayden about my deep work session

***20/02/2024***
***Things I learned today***
- We address influential outliers differently from error outliers and interesting outliers depending 
on particular statistical techniques. 
- 2 types of influential outliers: model fit (presence changes the fit of the model) and prediction (change the parameter estimates of the model).
- 2 step process to identify model fit outliers: identifying data pts most likely to have influence
on the fit of the model since they deviate markedly from other cases in the dataset, investigating 
these cases to understand if they genuinely have influence on the model fit. 
- 3 techniques to assess the presence of pred outliers in regression: DIFFITS, DFBETAS, Cook's Di
- 3 courses of action to handle model fit and pred outliers: respecification (adding other terms to the regression equation,]
deletion and robust approaches (involving non-OLS standard). We should always report results with 
and without the technique.
- the process of identifying model fit outliers in SEM is similar to regression
- In regression, there are 2 types of pred outliers: global (impact all parameter estimates of the model) 
and specific (impacting 1 parameter estimate)
- We should use gCDi stat to detect global pred outliers while not neglecting the detection
of specific pred outliers (using the standardised change in the jth parameter resulting 
from the deletion of the obs i)
- Handling influential outliers in SEM is similar to in regression. Recommended the use of deletion
or robust approaches
- In multilevel modelling, the main goal of an analysis is assessing the size of the variance
components and the sources of such variances. 
- The recommendation is a top-down approach in identifying model fit outliers in multilevel modelling,
beginning at the highest level of analysis. The researcher then can decide whether a group of obs
affects the model fit b/c of a) the group itself and/or b) a particular data pt(s) in the group
- In multilevel modelling, we use a top-down approach. The recommendation is calculating 
the avg sqrd deviation, then the researcher can compare Cj vals against one another using an index plot. 

***Things I did well today***
- completing correlation heat maps
- Learning about: influential outliers and how to handle them in regression, SEM and multilevel modelling

***Things I did not do well today***
- Not finishing the paper on identifying and handling outliers

***Things I will do better next time***
- Finishing the paper

***22/2/2024***
***Things I learned today***
- Handling influential outliers in multilevel modeling are similar to those used in regression and SEM. But, unlike regression,
researchers need to 1st decide the level where any additional predictor(s) are to be added in the multilevel modelling context
- Options for robust techniques in multilevel modeling include: generalised estimating equations (GEE) and bootstrapping methods

***Things I did well today***
- Finishing the paper
- Code to visualise Q-Q plots 

***Things I did not do well today***
- Not reading other papers about outliers 
- Not doing time series analysis

***Things I will do better next time***
- Reading more papers about outliers
- Doing time series analysis

***27/02/2024***
***Things I learned today***
- ValueError: an exception raised when a function receives an argument of the correct type but an inappropriate value. 
- AttributeError: an exception raised when an attribute reference or assignment fails, normally occurs when
trying to access or modify an attribute or method that doesn't exist for a specific object or class 
- In Python, flattening an array means transforming a multi-dimensional array to a one-dimensional array to make it 
easier to iterate over
- We can ask Copilot to make our code more concise

***Things I did well today***
- Coding Q-Q Plot
- Coding distribution tests

***Things I did not do well today***
- Not being able to code box plots 

***Things I will do better next time***
- Coding box plots for all columns

***29/02/2024***
***Things I learned today***
- https://www.machinelearningplus.com/plots/matplotlib-plotting-tutorial/
- KeyError: occuring when trying to access a key that isn't in a dictionary or dictionary-like object
- TypeError: occuring when types don't match when performing Python operations
- to draw a line plot using matplotlib, ordinals have to meet the requirements

***Things I did well today***
- Trying to use matplotlib to draw line plots
- Using plotly to draw line plots 

***Things I did not do well today***
- Encountering problems with matplotlib

***Things I will do better next time***
- Fixing matplotlib

***5/3/2024***
***Things I learned today***
- If the error "Permission denied" => no access to Python packaging tools => need to add the variable to PATH. Also need to create a different
venv to test
- decomposition models can be additive or multiplicative. Additive: when seasonality and irregular variations don't change as much as trends changes.
Multiplicative: when seasonality and irregular variations increase in amplitude as trend changes
- When different features have different scales: consider adding another y-axis or normalise the data

***Things I did well today***
- Fixing matplotlib

***Things I did not do well today***
- Not finishing with visualising all features, including Volume with plotly
- Not finishing with seasonal.decompose yet 
***Things I will do better next time***
- Visualising all features with a line plot
- Finishing with seasonal.decompose

***7/3/2024***
***Things I learned today***
- Multivariate normality: when a dataset has multiple variables and these variables have data points
that are normally-distributed together. To assess whether multivariate normality exist: visual inspection (scatter plot
for pair vars to see if there are signs of elliptical patterns), PCA (checking variance), Mardias's test
- PCA: identify trends and relationships, help with dimensionality reduction and feature extraction

***Things I did well today***
- Drawing scatter plots for variables to see their relationships
- Checking for multivariate normality 
- Performing PCA

***Things I did not do well today***
- Spending a lot of time with Shapiro-Wilk test programming with stock data when not knowing whether it is right for them
- Not finishing multivariate normality check

***Things I will do better next time***
- Checking the business problem at hand
- Finishing multivariate normality check and understanding why multivariate normality

***14/03/2024***
***Things I learned today***
- explained variance: the proportion of total variability accounted for by a component/factor. The higher, the more influence it has
on the data 
- loadings: weights of components, indicating the covariance/relationships between original features and PCA-scaled units
- to understand PCA: check explained variance and loadings, relating these back to specific domain knowledge. To engineer new features, the 
higher explained variance the better

***Things I did well today***
- finishing code for PCA

***Things I did not do well today***
- Not saving what I found using PCA
- Not grasping why multivariate normality

***Things I will do better next time***
- Continuing with PCA
- Understanding why multivariate normality

***18/03/2024***
***Things I learned today***
- programming to print out relationships and patterns in PCA 
- Rule of thumb: if loading >0.3 => substantial relationship with feature. 
- df.loc [row,col] => locating the actual item. df.index = row, df.columns = col
- Why multivariate normality: regression analysis, PCA, ML and algorithms, factor analysis, hotelling's t-squared test (comparing means of multivariate data),
quality control and process monitoring, portfolio theory and risk management, robustness and sensitivity, graphic representations
***Things I did well today:***
- Coding to print out relationships between explained variance and the dataset, loadings and feature
- Getting an understanding why multivariate 
***Things I did not do well today:***
- wrong for loop for loadings, why print out the same results 3 times?

***Things I will do better next time:***
- Check my understand of how computer thinks and for loop

***20/03/2024***
***Things I learned today***
- seasonal decomposition => additive (trend and irregularities remain stable as time increases),
multiplicative (trend and irregularities change as time increases)
- high cumulative variance: variables explain the data, components retain most of the info in the original variables. low cumulative variance: the data varies sporadically 
=> variables cannot explain the data, total variance spread out across many components

***Things I did well today***
- Digging deeper into PCA
- Starting seasonal decomposition
***Things I did not do well***
- Taking some time to conduct PCA
- Not looking at how computers think and for loop
***Things I will do better next time***
- Checking what I will do better first 
- Spending less time on PCA

***22/3/2024***
***Things I learned today***
- To test for multivariate normality, PCA is not a reliable measure. We need to employ Mardia's test or Henze-Zirkler's test.
- Henze-Zirkler's test is based on a nonnegative functional distance measuring the distance between 2 distributions. Key
points: test statistic (approximately distributed as lognormal), distance measure(measuring the distance between the characteristic function of a multivariate normality
and the empirical characteristic distribution), consistency: Henze-Zirkler is a consistent test, as the sample size increases, the power of Henze-Zirkler to test
if the null hypothesis is false approaches 1. 
- Mardia's test: based on 1st and 3rd moments of the data distribution, (skewness and kurtosis) => based on chi-squared distribution (widely used for inferential statistics).
Skewness test and kurtosis test 

***Things I did well today***
- Conducting Henze-Zirkler's test and Mardia's test for multivariate normality
- Fixing up the code for CPA

***Things I did not do well today***
- PCA code is not concise enough
- Not fully grasping Mardia's test and chi-squared distribution
- Not conducting analysis for seasonal decomposition 
***Things I will do better next time***
- Shortening PCA code
- Grasping Mardia's test, Henze-Zirkler's test and chi-squared distribution

***Things I learned today***
- Chi-squared distribution: calculated by the sum of the squared of normal random variables, the shape is determined by 
degree of freedom. Used on various statistical tests.
- Period of seasonal decomposition depends. For stock data, could be yearly(1) or quarterly(4)
- Mardia's test: calculate 
- Henze-Zirkler's test

***Things I did well today***
- Finishing seasonal decomposition and analysis of it
- Finishing correlation heatmaps

***Things I did not do well today***
- Not conducting other testing for outliers
- Not finishing up CPA without outliers

***Things I will do better next time***
- Conducting other tests for outliers
- Finishing up CPA without outliers

***Things I learned today***
- Mardia's test: measuring the distance between x_i and x_j. The inverse of covariance matrix is the precision matrix (keeping the other variable constant 
while calculating the relationships between 2 other variables)
- Adjusted Close: the price of Close, but changed due to corporate actions

***Things I did well today***
- Learning about the maths behind Mardia's test in-depth. 
- Calculating the correlation coefficients and analyse correlations between features

***Things I did not do well today***
- Not finishing CPA without outliers
- Not conducting other tests for outliers yet

30/3/2024
***Things I learned today***
- Calculating mahalanobis distance
***Things I did well today***
- Writing the function to calculate mahalanobis distance
***Things I did not well today***
- Encountering ValueError while calculating mahalanobis distance
***Things I did well today***
- Fixing the error

4/1/2024
***Things I learned today***
- Bonferroni correction: fixing the alpha level for multiple tests since it reduces type I errors 
- chi-squared distribution: a special case of gamma distribution, used for hypothesis testing. critical value based on alpha level and degree of freedom (no of vars -1 )
- F-distribution: for 2 chi-squared distribution.Critical value based on alpha level & degree of freedom for numerator and denominator 
- The data has multicollinearity, when calculating sigma, need to add a jitter (ridge regression) to have an inverse since the sigma is a singular matrix 
with no inverse (why?)
***Things I did well today***
- Finishing calculating mahalanobis
- Understanding more about data distribution

***Things I did not do well today***
- Not fully understanding why a singular matrix doesn't have a inverse
- No PCA without outliers
***Things I will do better next time***
- Understanding more about inverse matrix
- PCA without outliers

***3/4/2024***
***Things I learned today***
- https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/full/10.1002/cem.2692
- self: reference to the class
- __init__: initialise a class
- class: dictate behaviors of objects 
  - Inverse matrix: a matrix when multiplied with our matrix will produce a matrix with 1 along the diagonal. To calculate an inverse matrix
        - A determinant: calculate the determinant of the matrix
        - An adjucate matrix: find cofactors (sign factor multiplied by minor) and replace all the elements of the original matrix with the corresponding cofactors
        - An inverse matrix: 1/det(A) * adj(A)
***Things I did well today***
- Writing class for better and more maintainable code 
- Understanding more about inverse matrices
***Things I did not do well today***
- Not finishing handling outliers
***Things I will do better next time***
- Finishing handling outliers

***5/4/2024***

***Things I learned today***
- Debt can be positive for a company if debt for growth, debt for efficiency, debt for debt,tax benefits and positive market perception. Risks: increased debt, interest expenses, cash flow, risk perception, opportunity cost

***Things I did well today***
- Understanding that outliers detected are most likely not error outliers. Studying interesting outliers

***Things I did not do well today***
- Not finishing analysing and studying outliers
- Errors with box plot programming

***Things I will do better next time ***
- Be specific in Trello about tasks
- Continuing with outliers
- Fixing errors with box plot programming

***8/4/2024***
***Things I learned today***
- High vol trading: due to market news and events, earnings reports, price movements, trend confirmation, product launches, market sentiment shifts, algorithmic trading
& institutional activity, context, innovation, breakouts and breakdowns, market orders and limit orders
- In Python 3, zip built-in function wraps 2 or more iterators with a lazy generator (iterating over multiple iterators in parallel). => cleaner code than list, yielding tuples containing the next val from each generator. 
However, zip behaves strangely if input are of different lengths. 

***Things I did well today***
- finishing box plots
- analysing interesting outliers

***Things I did not do well today***
- Having not finishing interesting outlier analysis

***Things I will do better next time***
- Continuing interesting outlier analysis 

***10/4/2024***
***Things I learned today***
- Only create the axes and figure once
***Things I did well today***
- Continuing outlier analysis
- Creating line plots
***Things i did not do well today***
- Line plots look messed up
***Things I will do better next time***
- Fixing up line plots
- Learning more about loop

***11/4/2024***
***Things I learned today***
- filtering operations data[col][data[col] > a] to remove unwanted parts
***Things I did well today***
- Analysing outliers
- Understanding why behind outliers more
***Things I did not do well today***
- Not fixing up line plots
- Not learning more about loops

***16/04/2024***
***Things I learned today***
- PCA: a technique of orthogonal transformation converting correlated variables to non-linearly uncorrelated variables. 
- Price-related components might have more influence in PCA due to its variance in the market, its effects on investors due to price changes.
Volume-related components might have less influence in PCA due to its relative stability

***Things I did well today***
- Analysing outliers
- Understanding more about PCA

***Things I did not do well today***
- Not doing bollinger bands
- Not doing time-series analysis
- Not learning about loops

***Things I will do better next time***
- Continuing outlier analysis 
- Starting bollinger bands
- Starting time-series analysis
- Learning more about loops

***1/5/2024***

***Things I learned today***
- Factors which could influence trading volume: earnings announcements, analysts ratings, market news and events, product launches and innovations

***Things I did well today***
- Continuing outlier analysis

***Things I did not do well***
- Not starting bollinger bands
- Not starting time-series analysis
- Not learning more about loops

***3/5/2024***
***Things I learned today***
- 4 things could affect Volume: analyst ratings, earnings reports, news events and market trends.
- Can use PatchTST to build a model
***Things I did well today***
- Nearly finishing outlier analysis
- Chatting with Yudhi about my project
***Things I did not do well today***
- Going slow with outlier analysis
***Things I will do better next time***
- Speeding up outlier analysis

***6/5/2024***
***Things I learned today***
- PayPal had outliers from 2013-2023 for price-related variables and from 2015-2023 for volume-related variables due to 
global expansion, new products and services, earnings reports
***Things I did well today***
- Continuing outlier analysis
***Things I did not do well today***
- No being as productive regarding outlier analysis
***Things I will do better next time***
- Not going to event next week during workday while working on my projects

***8/5/2024***
***Things I learned today***
- .loc is label-based. Therefore, we need access row by equating our column to a value (where that value resides). Cannot .loc[value] since the 
label has to match the exact label, which is 'value'
- Encountering IndexError today since there were empty objects when running code without exception handling
***Things I did well today***
- Continuing my outlier analysis 
- Understanding more about .loc 
***Things I did not do well today***
- Having not finished outlier analysis
- Having taken some time to figure IndexError

***Things I will do better next time***
- Giving myself a timeframe to figure out error (30 mins). If not, check answer with Copilot then move on
- Continuing with outlier analysis

**13/5/2024***
***Things I learned today***
- Bollinger Band: normally for short-term analysis of volatility (over or under-valued). 3 bands: upper band: set of certain number of SD (usually 2 above the MB), middle band (20 day SMA (Simple Moving Average)) and lower band (MB - 2 SD)

***Things I did well today***
- Finishing outlier analysis
- Looking into problem identification, investigation, formulation, and prevention

***Things I did not do well today ***
- Not finishing out problem identification
- Rereading previous sentences
***Things I will do better next time ***
- Reading sentences carefully 
- Continuing with problem identification

***10/6/2024***
***Things I learned today***
- moving average is a stock indicator, a type of times series analysis, for technical analysis. 3 types: simple moving average, cumulative average, exponential average
Depends on use cases, use which one for best insights
***Things i did well today***
- Started time series analysis
***Things I did not do well today***
- Not finishing moving average
***Things I will do better next time***
- Finishing moving average

***11/6/2024***
***Things I learned today***
- To determine the best smoothing technique, check: trend, noise, seasonality, volatility, missing values and outliers
***Things I did well today***
- Looking at different smoothing techniques
***Things I did not do well today***
- Not able to choose a smoothing technique
***Things I will do better next time***
- Choose a suitable smoothing technique

***12/06/2024***
https://www.strike.money/technical-analysis/volatility-analysis
https://www.investopedia.com/articles/basics/09/simplified-measuring-interpreting-volatility.asp
***Things I learned today***
- to check if data follow normal distribution: use various techniques
- Q-Q plot: check central tendency, deviations, and overall trend

***Things I did well today***
- rechecked my normal distribution code
***Things I did not do well today***
- not finished check normal distribution
***Things I will do better next time***
- Finishing detection for normal distribution

13/6/2024
***Things I learned today***
- Since stock data tend to have fat tails (more extreme values), we should not choose Shapiro-Wilk test for normality. Instead,
we should opt for Kolmogorov-smirnov test or anderson-darling test
***Things I did well today***
- conducted the kolmogorov-smirnov test and anderson-darling test
***Things I did not do well today***
- Not deciding the tests for volatility 
***Things I will do better next time***
- Choosing tests for volatility and starting to conduct them 

14/06/2024
***Things I learned today***
- Volatility test: Average True Range, Bollinger bands, Implied Volatility
***Things I did well today***
- Started volatility test
***Things I did not do well today***
- Not finishing volatility test
***Things I will do better next time***
- Finishing volatility tests

15/06/2024
***Things I learned today***
- To choose a volatility threshold for ATR, investigate: trading strategy, risk tolerance, asset characteristics,
and historical volatility. ATR can help to identify high volatility phases, set stop loss levels (can times 2 or 3 using the average historical volatility), filter out market
noise, position sizing (adjusting positions)

***Things I did well today***
- Finished coding for ATR

***Things I did not do well today***
- Not finishing volatility test
***Things I will do better next time***
- Finishing volatility test


18/6/2024
***Things I learned today***
- to conduct volatility test, need real-time data
***Things I did well today***
- Understand the volatility test
- Start obtaining real-time data
***Things I did not do well today***
- Did not finish real-time data analysis
***Things I will do better next time***
- Finishing real-time data analysis

19/06/2024
***Things I learned today***
- can build reflection agent: generator and reflector
***Things I did well today***
- Looking at building a reflection agent
***Things I did not do well today***
- Error with importing LangGraph
***Things I will do better next time***
- Looking at the error more closely
20/06/2024
***Things I learned today***
- building Markov-chain -based multi-agent to reduce hallucination
***Things I did well today***
- Solve LangGraph error by importing module httpx
- Looking at building a reflexion agent using Gemini and using Markov Chain
***Things I did not do well today***
- Not finishing building a reflexion agent
- Not reading the reflexion paper
- Not fully understanding error messages
***Things I will do better next time***
- Read the reflexion paper
- continue building a reflexion paper
- Understand error messages better
21/06/2024
***Things I learned today***
- Reflexion: a novel framework to reinforce the agent not by updating the weights but by giving linguistic feedback. LLM agents find it hard
to learn from past mistakes
- getpass: a module to process passwords
- -> None: return no value
- (var:str): variables should be strings
***Things I did well today***
- Finished the Going Meta: Building a reflection agent with LangGraph
- Started reading "Reflexion: Language Agents with Verbal Reinforcement Learning"
***Things I did not do well today***
- Not finishing reading the paper
- Not finishing building the agent
***Things I will do better next time***
- Finishing reading the paper
- Continue building the agent
22/6/2024
***Things I learned today***
- Knowledge Graph + Multimodal learning => overcome challenges, heading towards AGI
***Things I did well today***
- finished common UI Patterns
- nearly finished Information Architecture
- started reading "Knowledge Graph Meets Multimodal Learning: A Comprehensive Survey"
***Things I did not do well***
- Not finished IA yet
- Not finished reading the paper yet
***Things I will do better next time***
- Continuing reading the paper
- Finishing IA and UI/UX design

24/6/2024
***Things I learned today***
- MMKG => KG is multimodal if knowledge symbols contain multiple modalities.
- 2 approaches to MMKG construction: 1=> labelling KG symbols with images, 2=> grounding KG symbols to images
***Things I did well today***
- continue reading the paper KG Meets Multimodality
- Coding to configure tracing
***Things I did not do well today***
- Not finishing the paper
- Not finishing configure tracing
***Things I will do better next time***
- Reading the paper faster
- Finishing tracing 
- Start looking at building a multimodal AI agent

25/6/2024
***Things I learned today***
- 4 types of tasks: understand and reasoning tasks (VQA and VQG), classification tasks, content generation tasks, retrieval tasks

***Things I did well today***
- Reading MMKG paper
- Coding the agent
***Things I did not do well today***
- Not finishing MMKG paper
- Trouble with creating prompt
***Things I will do better next time***
- Reading the paper faster 
- Look at prompt engineering for multimodal model. 

26/6/2024
***Things I learned today***
- Multiple types of tasks for MMKGs
- top_k: method using the top probability k in words
- top_p: the size of the shortlist of words based on the sum of likelihood score based on sum threshold
***Things I did well today***
- Finishing the second pass
- Creating prompt
***Things I did not do well today***
- Error with prompt
***Things I will do better next time***
- Understand Gemini model 
- Understand the prompts for Gemini

27/06/2024
***Things I learned today***
- memories divided to 2 types: conditioned reflexes (behaviors learned from experience) and torso-to-tail knowledge (head:common knowledge, torso: less common, tail: least common)
- can combine crewAI with Gemini
***Things I did well today***
- third pass of MMKG
- continue building agents
***Things I did not do well today***
- error with Agent
***Things I will do better next time***
- read crewAI docs
- fix the agent API code

1/7/2024
***Things I learned today***
- attention mechanism used in transformer based model 
***Things I did well today***
- access Gemini model from torch library
- continued reading MMKG
***Things I did not do well today***
- Not finished 3rd-pass reading MMKG
***Things I will do better next time***
- Reading the paper first for the 3rd pass
- Focusing on relevant papers 
- Challenging assumptions of the paper
- Continue coding for Gemini

2/7/2024
***Things I learned today***
- KGs => graph algos, graph nn. KGs help both humans and machines digest data and relationships 
- for unit testing => use pytest
- types of software testing: unit testing, integration testing, system testing, acceptance testing 
***Things I did well today***
- start unit testing for my code
- continue 3rd pass reading and challenging the paper
***Things I did not do well today***
- not finishing testing yet since error of baseline_images not found. 
- not finishing 3rd pass reading
***Things I will not do well today***
- Fix errors
- finish unit testing
- continue 3rd pass reading and challenge the paper 
5/7/2024
***Things I learned today***
- install matplotlib from the source for test data
***Things I did well today***
- fixing errors
***Things I did not do well today***
- not finishing fixing
***Things I will do better next time***
- continue fixing the error

8/7/2024
***Things I learned today:***
- Advantages of initialising parameters: breaking symmetry, avoiding exploding/vanishing gradients, faster convergence. Disadvantages: stuck at local optima, 
exploding values,heuristics for initial scale or weights
***Things I did well today:***
- Creating agents
- Continue 3rd-pass reading of MMKG
***Things I did not do well today:***
- Not fixing unit testing error in pytest
***Things I will do better next time:***
- Fixing unit testing error in pytest 
- Asking Hayden

9/7/2024
***Things I learned today***
- fl-mmkg: feature-level knowledge graphs, features represent multimodal data. Advantages: flexible, improved performance, rich information, interpretable. drawbacks: complexity, noise, scalability
- n-mmkg: nodes (entities) represent mm data. Advantages: rich info, improved performance, scalability, flexibility, interconnected entities. drawbacks: limited modalities since each node == each modality, integration challenge, performance
***Things I did well today***
- 3rd pass MMKG
- constructing fl-mmkg
***Things I did not do well today***
- not fixing the error
- still 3rd pass MMKG => not fully understanding the paper
- constructing fl-mmkg
***Things I will do better next time***
- ask Hayden about the error
- look the level of understanding of the paper. look at background papers
- continue to look at constructing fl-mmkg 

10/7/2024
***Things I learned today***
- harmonic mean: the reciprocal of the arithmetic mean of reciprocals, used in F1 score (P x R => to balance between P and R)

***Things I did well today***
- looking at how to collect data 
- looking at fixing pytest error
- 3rd pass reading MMKG
***Things I did not do well today***
- not able to fix pytest error
- not finishing 3rd pass yet

***Things I will do better next time***
- ask Copilot
- continue looking at data collection
- keep challenging and looking at background papers for MMKG

11/7/2024
***Things I learned today***
- lemmatization helps find the root of words, making conversations easier for an AI agent to understand
***Things I did well today***
- developing uri tool
***Things I did not do well today***
- not able to finish 3rd pass reading
- not able to solve error
***Things I will do better next time***
- Finish 3rd pass reading
- ask someone to solve errors

15/07/2024
***Things I learned today***
- compound structure: a data structure composed of different data types. 
- main operator: the main operator of the data structure
- assert: check if a condition is met, otherwise throw an assertion error
***Things I did well today:***
- added compound_uri, is_absolute_uri, join_uri,concept_uri, uri_prefix,split_uri functions
- finished 3rd pass MMKG
***Things I did not do well today:***
- did not fully understand the MMKG paper after 3rd pass. 
***Things I will do better next time***
- Will need to read background papers or do more research on MMKG
- Asking about the unit test error

16/07/2024
***Things I learned today:***
- yield: keyword to create a generator function (behave like a iterator) 
- a compound URI: contains an operator and list of arguments
- conjunction of sources: combining multiple sources to create an assert URI
- Advantages of ConceptNet: multilingual, broad coverage, word embeddings, open-source. Disadvantages: complexity, data quality. lack of practical detail
To improve: enhance data quality, expand coverage, improve performance with data structures and algorithms, enhance usability (documentation,tutorials,new tools, libraries),
research (use it as a basis for researching)
***Things I did well today:***
- adding functions to uri tooling: is_concept, is_relation, is_term, uri_prefixes,conjunction_uri, parse_compound_uri,parse_possible_compound_uri,get_uri_language
- looking at deploying app on GC
- knowing more about ConceptNet
***Things I did not do well today:***
- not reading background papers on MMKG
- not asking about unit test error
***Things I will do better next time***
- asking about unit test error
- reading background papers on MMKG

17/07/2024
***Things I learned today:***
- ConceptNet
***Things I did well today:***
- adding add_tools package
- starting to read Knowledge Graphs paper
***Things I did not do well today:***
- not looking at wireframing
- not fixing unit test error
***Things I will do better next time:***
- looking at wireframing
- fixing unit test error
- continuing KG construction

22/7/2024
***Things I learned today***
- to access a package in a directory, we might need to add the parent package when importing 
***Things I did well today***
- fixed the error of importing uri
- finished the function add_relation
***Things I did not do well today***
- Spending a lot of time fixing the import
***Things I will do better next time***
- Reading MMKG background paper
- Continuing MMKG construction
- Looking at wireframing 

23/07/2024
***Things I learned today:***
- re: a built-in module to work with regular expressions (characters matching specific patterns).
re.match: deciding if the regex matches at the beginning of the string
re.findall: finding all non-overlapping matchies of the regex as a list of strings
- exception: thrown when request is unsuccessful. try: attempting a block of code before throwing an error
- exponential backoff: an error-handling strategy to increase delayed time exponentially.
randomised exponential backoff: starting from 0 and finishing at exponential backoff, varying the delayed time to to prevent clients starting at the same time to resend requests repeatedly.
***Things I did well today:***
- coding for fetch_tool
- trying to deploy the app
***Things I did not do well today:***
- Not deploying yet since no app yet
- No wireframing
- No MMKG background paper
***Things I will do better next time:***
- Continue to build MMKG
- Looking at wireframing when have time
- Reading MMKG paper if have time

24/7/2024
***Things I learned today***
- pattern
- need proper exception handling
- pagination: node data across different pages
- knowledge graph: entity, edge, node
***Things I did well today:***
- added transform_path, get_page, get_node, combine_pages, get_edge, get_multi_model
- read the knowledge graph paper
***Things I did not do well today:***
- no wireframing
- not understanding IOError
***Things I will do better next time***
- wireframing if have time
- Understanding IOError
- Continuing reading knowledge graph paper

25/07/2024
***Things I learned today:***
- space and time complexity: how much time does it take to run and how much space memory does it take up
- a set is quicker to look up since it uses a hash table, whereas a list needs to check first 
- weights (graph): the weights of the element (related to strength, importance, relevance, confidence). Strategies to determine: frequency-based, attribute-based
***Things I did well today:***
- adding acceptable_element, save_to_local
- reading KG paper
- Understanding IOError
***Things I did not do well today:***
- no wireframing
- read KG paper slowly
- not determined the weights strategy yet
***Things I will do better next time:***
- wireframing if have time
- continuing KG paper 
- determining the strategy for weights

26/07/2024
***Things I learned today:***
- subprocess: module to spawn new processes from given input/output/error pipelines
***Things I did well today:***
- adding delete_uri, filter_node, need_extension
***Things I did not do well today:***
- not determining the weight strategy
- no wireframing
***Things I will do better next time:***
- wireframing if have time
- determining the weight strategy
- continue coding
- reading KG paper

29/07/2024
***Things I learned today***
- Identity: global identifiers, identity links, persistent identifiers, lexicalisation
- Context: direct representation, higher-arity representation, reification, annotation
***Things I did well today***
- reading KG
- coding for downloading images
***Things I did not do well today***
- spending too much time on search url
- no wireframing
- no weight strategy
***Things I will do better next time***
- spending less time on search url
- no weight strategy
- wireframing when we can

30/07/2024
***Things I learned today:***
- requests.get: Send a GET request
- .format: formatting a string
- .raise_for_status: check if the request is successful, raise an error if unsuccessful
- wb: writing binary mode
- ontologies in KG: formal representation of knowledge so that we represent data as a graph => interpretations, properties, individuals, classes
***Things I did well today:***
- continuing coding to download sound
- continuing to read KG paper
***Things I did not do well today:***
- not finished download_sound. 
- no wireframing
***Things I will do better next time:***
- Finishing download_sound
- Wireframing if can
- Continuing to read KG paper

31/07/2024
***Things I learned today***
- subprocess.check_output: run a command and give output
- subprocess.Popen : starting a process to run a specific command
- process.communicate: sending a process and return its output and error
- process.kill: terminate the process immediately
- process.terminate: sending a signal to terminate the process 

***Things I did well today***
- finish coding download_sounds
- read KG paper

***Things I did not do well today***
- not finished extend_tool
- not wireframed
***Things I will do better next time***
- finishing extend_tool
- wireframing 
- reading KG paper

1/8/2024
***Things I learned today:***
- ensuring that the return type is consistent 
- need logging to help with tracking. logging.basicConfig() => setting up basic configurations. logging.info() => Emitting log messages. logging.error => Emitting error messages if serious problems. 
- 3 types of basic elements in Description Logics: individuals, classes and properties
***Things I did well today:***
- Coding functions for extend tool
- Wireframing
- Reading KG paper

***Things I did not do well today:***
- Not finished extend_tool

***Things I will do better next time:***
- Finishing extend_tool 
- Continuing to wireframe
- Reading KG paper

2/8/2024
***Things I learned today:***
- List, Dict, Any => typing
***Things I did well today:***
- coding
- wireframing
***Things I did not do well today:***
- not finished wireframing
- not read KG paper
***Things I will not do better today:***
- Continue coding
- Finishing wireframing
- Reading KG paper

5/8/2024
***Things I learned today:***
- input prompt should be outside the function as a good practice
- demo video: should know what to say and what product does. 1) plan the video. 2) record your product or service in action 3)edit video
***Things I did well today:***
- finished download visuals
- looked at creating demo
***Things I did not do well today:***
- no wireframing
- no KG paper reading
***Things I will do better next time:***
- continue to code
- wireframing if can
- continue reading KG paper

6/8/2024
***Things I learned today***
- global variable: a variable, declared outside, with global scope across functions and classes. Characteristics: global scope, persistence, shared state. When to use: config settings, constants, shared state. Disadvantages: unintended modifications, reduced readability (having to find the global var), testing difficulties. Good practices: minimise use, encapsulation, clear naming.
***Things I did well today:***
- Coding for transport tool.
- Understanding what global variable is.
***Things I did not do well today:***
- No reading KG paper
- Not finished transport tool
***Things I will do better next time:***
- Understanding command
- Reading KG paper
- Continuing coding transport tool

7/8/2024
***Things I learned today***
- Need transport tool to package and send a file to a server
***Things I did well today:***
- Finished the transport tool package
***Things I did not do well today:***
- No wireframing
- No reading KG paper
***Things I will do better next time:***
- Wireframing if possible 
- Reading KG paper if possible 
- Continue coding 
- Understanding more deeply

9/8/2024
***Things I learned today:***
- need to clean up and delete graphs because of: data integrity, testing nodes and graphs, performance, security, consistency
- except ServiceUnavailable: an exception when service is unavailable

***Things I did well today:***
- nearly finished prototyping
- continued coding
***Things I did not do well today:***
- No wireframing
- No KG reading
- Not fully focused

***Things I will do better next time:***
- Focusing on prototyping (practice focus with meditation)
- Continue to code the rest of the knowledge graph 

13/08/2024
***Things I learned today:***
- json.decoder.JSONDecodeError: throwing an exception when decoding JSON in file

***Things I did well today:***
- coding history.py

***Things I did not do well today:***
- few code

***Things I will do better next time:***
- continue to practise coding everyday 

2/9/2024
***Things I learned today:***
- need to add the module to the path sys.path.append('C:/Users/laran/PycharmProjects/ASX/stock_market/fl_multimodal_knowledge_graph/data_collection/data_collection_tools') 
to use download_sounds in the test
- difference between assertions (assumptions during the executions of code) and exceptions (runtime errors and exceptional conditions)
***Things I did well today:***
- created a unit test for download_sounds
- added mel_spectrogram_features to the code feature_extractor to convert audio wave to mel spectrogram for better signal-to-noise ratio (adding filter)
***Things I did not do well today:***
- caught mistakes in download_sounds module while conducting a unit test
- not finished mel_spectrogram_features yet
***Things I will do better next time:***
- understanding where the errors came from (root cause)
- reading knowledge graph paper
- continue with mel_spectrogram_features

3/9/2024
***Things I learned today:***
- The consistent TypeError is due to no proxy_list passed. I expected this since I don't want to pass a proxy list yet
- a decorator is a function to modify behaviors without changing the actual code. Characteristics: functions as arguments, wrapper function, syntatic sugar. Use cases: logging, authentication, timing
- While pytest is capturing output, the input function cannot read from stdin(standard inputs) => need to mock the input function to a predefined valu
***Things I did well today:***
- Figured out the TypeError and why it happened
- Completed unit tests for download_sounds and download_images

***Things I did not do well today:***
- Took too much time to figure out TypeError

***Things I will do better next time:***
- Find a way to understand the root cause of errors better
- Read the Knowledge Graph paper
- Wireframing while can 

4/9/2024

***Things I learned today:***
- Framing: create frames from samples for better sampling, window => samples within each frame, hop => advance to move forward from 1 frame to another
- Hann window: window that has hann function applied to make it smoother and prevent spectral leakage with better Fourier analysis
- padding: adding zeros to arrays to make the length of each frame the same
- Short-Time Fourier Transform (STFT): allowing for the analysis of signals by dividing longer signals in to smaller ones. STFT{x(t)}(m,omega) = summation (from n = - infinity to inf) of x[n].w[n-m].e to the power of j omega n
- Magnitude of STFT: the amplitude of the frequency component at each point in time. = |STFT{x(t)}(m,omega)|

***Things I did well today***
- Coding framing, Hanning window and stft magnitude
- Reading KG papers
- Understanding more about Fourier Transform
***Things I did not do well today:***
- Not finished mel_spectrogram_features module yet
- No wireframing yet
- not finding a way to understand the root cause of errors better yet
- Not writing unit tests yet

***Things I will do better next time:***
- Understand each and every line of code and its concept
- Reading KG papers
- Wireframing if have time 
- Finding a way to understand root cause of errors 
- Writing unit tests where applicable

5/9/2024
***Things I learned today***
- converting from hertz to mel is beneficial since it mimics human hearing and help AI agents to hear like humans do due to better perceptual alignment, better feature extraction, better analysis, model performance and data efficiency
- Nyquist frequency: the highest frequency in a digital processing system without aliasing (distortion)
- Need to add ValueError when checking to see if values are incorrectly ordered or out of range (suitable for checking values like Hertz, kgs etc)

***Things I did well today：***
- Adding hertz_to_mel(), spectrogram_to_mel_matrix(), log_mel_spectrogram()
- Understanding more about the process of transforming audio waveform to mel spectrogram features 

***Things I did not do well today:***
- Not understanding why we need to log mel spectrogram
- Not finishing stft_magnitude()
- Not reading knowledge graph paper
- Not wireframing

***Things I will do better next time：***
- Spending an hour next week for wireframing
- Understanding the need for logging mel spectrogram
- Finishing stft_magnitude()
- Reading KG paper
- Looking at ways to understand root cause of errors

9/9/2024
***Things I learned today:***
- Real Fast Fourier Transform: specialised version of FFT for real-valued input sequences.
- logging mel spectrogram to make large and small values more manageable (dynamic range compresion), create a representation that is more aligned with human hearing because of better feature extraction (perceptual relevance), avoid negative infinites by adding a small constant and improve convergence, and enhance feature discrimination using feature scaling and reduce noise

***Things I did well today:***
- Finish mel_spectrogram_features module
- Looking at image extraction
***Things I did not do well today:***
- Not reading KG papers
- Not wireframing 
- Not looking at ways to understand root cause errors

***Things I will do better next time:***
- Wireframe tomorrow
- Reading KG if have time
- Look at ways to understand root cause of errors

10/09/2024
***Things I learned today:***
- Using KL divergence to control sparsity level, more flexibility albeit more complex. Use L1 regularisation for simplicity, standard
- Combining CNNs and Vision Transformers as a base architecture may improve F1 scores. Hierarchical labels may also improve F1 scores. PreSizer resizes images, uses reflective padding, doesn't remove meaningful info, removes noise around images => improves F1 scores. Resolution is important for fine-grained classification tasks
- Improving Transformer-based model with quantization, knowledge distillation, pruning, low-rank approximation (weight matrices and low-rank matrices)
- ***Things I did well today:***
- Choosing base architecture for image extraction
- wireframing
***Things I did not do well today:***
- Not understanding what low-rank approximation is yet
- Not reading KG paper
- Not finishing sparse autoencoder
***Things I will do better next time:***
- Understanding low-rank approximation
- reading KG paper
- looking at ways to understand root cause of errors
- Finishing sparse autoencoder
- Starting ConVit

11/09/2024

***Things I learned today:***
- Using tensorflow for TPU, ease of support, community and with Keras API (ease of use, modularity, pretrained model, community and support)
- Using StratifiedGroupKfold for grouped data (companies in this case) to ensure representation for each group, no imbalance, each group appears once exactly in all folds (evaluating on unseen companies)
- Using sigmoid as an activation function for reconstruction ranging from 0 to 1=> loss: binary entropy
***Things I did well today:***
- Coding sparse autoencoder to learn feature representations and look at interpretability 
- Learning how to use tensorflow
- Looking at ways of understanding root causes of errors 
***Things I did not do well today:***
- Not finished coding for sparse autoencoder 
- Not reading KG paper
- Not starting Convit
- Not understanding low-rank approximation
***Things I will do better next time:***
- Finishing sparse autoencoder
- Reading KG paper
- Understanding low-rank approximation
- Starting ConVit

12/9/2024
***Things I learned today:***
- Can use keras_tuner for hypertuning 

***Things I did well today:***
- Coding Sparse autoencoder
- Using keras_tuner
- Finished paper Convit
- Learning new techniques
***Things I did not do well:***
- Not reading KG
- not understand low-rank approximation
- Not finishing Sparse autoencoder
***Things I will do better next time:***
- reading KG
- understanding low-rank approximation
- finishing sparse autoencoder
- starting ConVit

13/09/2024
***Things I learned today:***
- metrics for image tasks: mse, mae, psnr (Peak signal-to-noise ratio, measuring the difference between the highest value of signal to the power of corrupting noise),ssim(structural similarity index, measuring the difference between 2 images based on luminance, contrast and structure)
- 
***Things I did well today:***
- Finishing sparse autoencoder
- starting ConVit
- kind of understanding low-rank approximation 

***Things I did not do well today:***
- Not finishing MLP
- Not fully understanding low-rank approximation
- Not reading KG
***Things I will do better next time:***
- Finishing MLP
- Fully understanding low-rank approx
- Reading KG paper

16/09/2024
***Things I learned today***
- low-rank approximation: approximating a given matrix using a different matrix of a lower rank for data compression, noise reduction and feature extraction, via Singular Value Decomposition(SVD)
- multi-layer perceptron: a type of nn with 3 layers, using feedforward propagation, backpropagation and activation function. A dropout layer is optional to prevent overfitting
- a gated positional self-attention: self-attention with a gating mechanism to focus on nearby groups of inputs
***Things I did well today:***
- finished MLP
- read KG paper
- understood low-rank approximation better
***Things I did not do well today:***
- not finished gpsa
***Things I will do better next time:***
- finished gpsa 
- adding complex beacons to make sure the code is readable
- ensure I understand the math behind gpsa
- include feature importance
- incorporate other types of AI models

17/09/2024
***Things I learned today:***
- if not hasattr: check condition if has attribute

***Things I did well today:***
- continued convit
- read KG paper
- starting to understand the math behind GPSA
***Things I did not do well today:***
- not including feature importance
- not incorporating other types of AI models
***Things I will do better next time:***
- including feature importance 
- continuing convit
- reading KG paper
- understand the math behind GPSA
- incorporating other types of AI models

18/09/2024

***Things I learned today:***
- the GPSA includes: calculating attention heads, obtaining an attention map, calculating positional scores, calculating patch scores, forward pass. calculating relative indices 

***Things I did well today:***
- finished GPSA 
- read Knowledge Graph
- understanding more about the math behind GPSA

***Things I did not do well today:***
- not included feature importance
- not incorporated other types of AI models

***Things I will do better next time:***
- including feature importance 
- incorporating other types of AI models
- understanding the concept behind attention mechanism: how query, key, and value vectors computed; how attention scores computed and used to weigh the value vectors; understand positional encoding (how it is used, and the role of relative indices), understand softmax function (how it is used to normalised, how softmax ensures att scores sum to one) understanding gating mechanism (combining patch scores and positional scores, how sigmoid control the influences of each type of score), matrix multiplications (matrix ops for computing attn scores, how these ops are implemented for efficient coding), how to implement GPSA, how to debug and optimize  

19/09/2024
***Things I learned today:***
- multi-head self-attention: a module used in transformer model, self-attention: how a model focuses on words in a sequence, multi-head: how the model distributes attention

***Things I did well today:***
- read "Attention is all you need"
- coding MHSA

***Things I did not do well today:***
- did not finish MHSA
- did not finish paper

***Things I will do better next time:***
- understand the paper more
- read KG
- include feature importance 
- incorporate other types of AI models
- finish MHSA

20/09/2024

***Things I learned today:***
- Euclidean distance: measuring the distance between two points, in units, based on Pythagorean theorem
- incorporating other types of AI models depends on the problems at hand. Pros: enhanced performance, improved generalisation, task-specific optimisation, flexibility and adaptability. Cons: complexity, integration, evaluation

***Things I did well today:***
- finished MHSA
- read paper “Attention is all you need"

***Things I did not do well today:***
- not including feature importance
- not incorporating other types of AI models

***Things I will do better next time:***
- include feature importance
- incorporate other types of AI models
- read KG paper if have time
- read paper "Attention is all you need"

23/09/2024

***Things I learned today:***
- 3 reasons of self-attention in Tranformer: reducing computational complexity per layer, ensuring every computation is parallised, reducing the length of path between long-range dependencies
- flatten: collapsing dimensions
- backbone of CNN: can be used to extract features
- kernel: a filter, a matrix to multiply with image for model to pay attention to specific features

***Things I did well today:***
- Coding HybridEmbed
- Finished second pass "Attention is all you need"
- understand more about the architecture of ConviTransformer

***Things I did not do well today:***
- Not reading KG paper
- Not finished convit yet


***Things I will do better next time:***
- 3rd pass Attention paper
- Finished convit
- feature importance
- other types of AI models
- KG paper if have time

24/09/2024
***Things I learned today:***
- SHAP (Shapely Additive exPlanations): visualising outputs from models and assigning Shapely values
- hybrid backbone: backbone models use a combination of different neural networks

***Things I did well today:***
- finished Conviformers
- applied SHAP
- implementing "Attention is all you need"

***Things I did not do well today:***
- not reading KG paper
- not incorporating other types of AI models

***Things I will do better next time:***
- Reading KG paper if have time
- Incorporating other types of AI models

25/09/2024
***Things I learned today***
- hidden markov model: each state depends on the prev state

***Things I did well today***
- Hidden Markov Model
- code to download texts
- 3rd pass attention
- read KG

***Things I did not do well today***
- not finished 3rd pass attention

***Things I will do better next time***
- finish 3rd pass attention
- continue coding to download texts
- create account to obtain APIs

26/09/2024
***Things I learned today:***
- To obtain listed companies' name, we can use stock exchange websites. 

***Things I did well today:***
- learned how to use Twitter API
- learned how to obtain information via coding
- finished 3rd pass "Attention is All You Need"

***Things I did not do well today:***
- not finished coding regarding Twitter API
- not reading Hidden Markov Model
- not reading KG paper

***Things I will do better next time:***
- finish coding regarding Twitter API
- reading Hidden Markov Model
- reading KG paper