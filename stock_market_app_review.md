
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

***12/3/2024***
***Things I learned today***
- loadings: explain the correlations/covariance between original features and pca-scaled units
- write a README.md: intro, features, installation, contribution, packages, acknowledgement, licenses
- creating a killer stock market app: real-time data visualisation, portfolio tracking and management,
technical analysis tools,company profile and fundamentals, customisable watchlists, heatmaps and sector analysis, historical data and backtesting, user authentication and security,
educational resources, news & market insights, user-friendliness, mobile responsiveness. 

***Things I did well today***
-Creating a README.md on GitHub repo
- Creating a plot for variance for PCA
***Things I did not do well today***
- Encountering HTTPError with too many requests while plotting
***Things I will do better next time***
- Understanding why multivariate normality 
- Checking why encountering HTTPError with too many requests and fixing it at its root cause 