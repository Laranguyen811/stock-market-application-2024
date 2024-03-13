- A Dash app is usually composed of 2 parts. The 1st one is the "layout" of the app and describes what the app looks like. 
The 2nd part describe the interactivity of the app
- Code dash.Dash(__name__): create a new Dash application:
1. dash.Dash: constructor (in class-based, object-oriented programming, a special type of function called to create an
object, preparing the new object for use, often accepting arguments that the constructor uses to set required member variables) for the app
2. __name__: a special variable in Python, set to the name of the module where it is used 
=> creating a new Dash app, using the name of the current module as the name of the Flask server Dash uses under the hood. 
- @app.callback: decorator used to create callback functions to make your Dash apps interactive. 
- plotly.graph_objects: providing a low-level interface to the underlying data structures defining the charts, the low-level interface of Plotly, 
providing more flexibility and control over the details of the plot.
Containing the building blocks of Plotly figures: traces for i,(name,data) in enumerate(stock_data.items()):
    Q1 = data['Close'].quantile(0.25)
    Q3 = data['Close'].quantile(0.75)
    IQR = Q3 - Q1
    #Define bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    #Detect outliers
    data_outliers = data[(data['Close'] < lower_bound) | (data['Close'] > upper_bound)]
    lower_bound_outliers = data[(data['Close'] < lower_bound)]
    upper_bound_outliers = data[(data['Close'] > upper_bound)]
    if data_outliers.empty:
        print(f'The outliers of {name} is none')
    if lower_bound_outliers.empty:
        print(f'The lower bound outliers for {name} is none')
    if upper_bound_outliers.empty:for i,(name,data) in enumerate(stock_data.items()):
    Q1 = data['Close'].quantile(0.25)
    Q3 = data['Close'].quantile(0.75)
    IQR = Q3 - Q1
    #Define bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    #Detect outliers
    data_outliers = data[(data['Close'] < lower_bound) | (data['Close'] > upper_bound)]
    lower_bound_outliers = data[(data['Close'] < lower_bound)]
    upper_bound_outliers = data[(data['Close'] > upper_bound)]
    if data_outliers.empty:
        print(f'The outliers of {name} is none')
    if lower_bound_outliers.empty:
        print(f'The lower bound outliers for {name} is none')
    if upper_bound_outliers.empty:
        print(f'The upper bound outliers for {name} is none')
    else:
        print(f'The outliers of {name} is {data_outliers['Close']}')
        print(f'Number of outliers: {data_outliers.shape[0]}')
        print(f'The lower bound outliers for {name} is {lower_bound_outliers['Close']}')
        print(f'The upper bound outliers for {name} is {upper_bound_outliers['Close']}')
    #Create boxplot
    plt.boxplot(data['Close'])
    plt.title (f'Box Plot of {name}')
    plt.show()for i,(name,data) in enumerate(stock_data.items()):
    Q1 = data['Close'].quantile(0.25)
    Q3 = data['Close'].quantile(0.75)
    IQR = Q3 - Q1
    #Define bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    #Detect outliers
    data_outliers = data[(data['Close'] < lower_bound) | (data['Close'] > upper_bound)]
    lower_bound_outliers = data[(data['Close'] < lower_bound)]
    upper_bound_outliers = data[(data['Close'] > upper_bound)]
    if data_outliers.empty:
        print(f'The outliers of {name} is none')
    if lower_bound_outliers.empty:
        print(f'The lower bound outliers for {name} is none')
    if upper_bound_outliers.empty:
        print(f'The upper bound outliers for {name} is none')
    else:
        print(f'The outliers of {name} is {data_outliers['Close']}')
        print(f'Number of outliers: {data_outliers.shape[0]}')
        print(f'The lower bound outliers for {name} is {lower_bound_outliers['Close']}')
        print(f'The upper bound outliers for {name} is {upper_bound_outliers['Close']}')
    #Create boxplot
    plt.boxplot(data['Close'])
    plt.title (f'Box Plot of {name}')
    plt.show()for i,(name,data) in enumerate(stock_data.items()):
    Q1 = data['Close'].quantile(0.25)
    Q3 = data['Close'].quantile(0.75)
    IQR = Q3 - Q1
    #Define bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    #Detect outliers
    data_outliers = data[(data['Close'] < lower_bound) | (data['Close'] > upper_bound)]
    lower_bound_outliers = data[(data['Close'] < lower_bound)]
    upper_bound_outliers = data[(data['Close'] > upper_bound)]
    if data_outliers.empty:
        print(f'The outliers of {name} is none')
    if lower_bound_outliers.empty:
        print(f'The lower bound outliers for {name} is none')
    if upper_bound_outliers.empty:
        print(f'The upper bound outliers for {name} is none')
    else:
        print(f'The outliers of {name} is {data_outliers['Close']}')
        print(f'Number of outliers: {data_outliers.shape[0]}')
        print(f'The lower bound outliers for {name} is {lower_bound_outliers['Close']}')
        print(f'The upper bound outliers for {name} is {upper_bound_outliers['Close']}')
    #Create boxplot
    plt.boxplot(data['Close'])
    plt.title (f'Box Plot of {name}')
    plt.show()
        print(f'The upper bound outliers for {name} is none')
    else:
        print(f'The outliers of {name} is {data_outliers['Close']}')
        print(f'Number of outliers: {data_outliers.shape[0]}')
        print(f'The lower bound outliers for {name} is {lower_bound_outliers['Close']}')
        print(f'The upper bound outliers for {name} is {upper_bound_outliers['Close']}')
    #Create boxplot
    plt.boxplot(data['Close'])
    plt.title (f'Box Plot of {name}')
    plt.show()and Layout. Scenarios: customisation, complex charts, learning purpose 
- dash_html_components depreciated => need to use 'from dash import html'
- In Dash, 'my-id' and 'value' mean the id and value properties of a Dash component. 
'my-id': unique identifier of a Dash component. 
'value': holding the current value of the component
- The 'inputs' and 'outputs' of the application are described as the arguments of the @callback
decorator. In Dash, the inputs and outputs are simply the properties of a particular component.
Whenever an input property changes, the function that the callback decorator wraps will get called automatically. 