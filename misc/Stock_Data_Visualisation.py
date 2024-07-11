import pandas as pd
import matplotlib.pyplot as plt
from stock_market_application_data import data_msft,data_goog,data_amzn,data_aapl,data_sap,data_meta,data_005930_ks,data_intc,data_ibm,data_orcl,data_baba,data_tcehy,data_nvda,data_tsm,data_nflx,data_tsla,data_crm,data_adbe,data_pypl

import plotly.graph_objects as go
from dash import Dash, html, dcc, Output, Input
app = Dash(__name__) #initialise Dash app
#Define our app layout
app.layout = html.Div(children=[
    html.H1(children='Stock Prices vs Time'),
    dcc.Graph(id='stock-prices')
])
#Create call back function
@app.callback(Output(component_id='stock-prices',component_property='children'),
              [Input(component_id='stock-prices',component_property='value')]
              )

def update_graph(input_value):
    # Create a list of traces to store data for each stock. Since plotly requires
    # each trace be represented as a separate object in the figure. When we create a trace, we can easily add or remove
    # traces from the figure as needed.
    # Each trace in the list is represented by a Scatter object.
    data_traces = []

    for name,data in stock_reset_index.items():
        ''' Creates a plot for each stock with price against time.
    Inputs: 
    stock_reset_index (dictionary): Dictionary of stock reset and its name
 
    Returns:
    fig (object): Figure object containing all the data and layout information for the 
    plot
    
     '''
        data_trace = go.Scatter(x=data['Date'],y=data['Close'],name=name)
        data_traces.append(data_trace)
        #Create the layout for the chart
        layout = go.Layout(title='Stock Prices vs Time',xaxis=dict(title='Date'),yaxis=dict(title='Price'))
        # Create the figure and add the traces and layout
        fig = go.Figure(data=data_traces, layout=layout)

        return fig

#Run Dash app:
if __name__ == '__main__':
    app.run_server(debug=True)


'''
Output:
fig (object): Figure object containing all the data
Create a simple dash app to visualise the stock prices.
Input:
stock_data (dictionary): Dictionary of stock
'''


