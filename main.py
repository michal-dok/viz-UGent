

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from functions import *
import plotly.express as px

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Simple Dash App with Scatter Plot"),
    dcc.Input(id='input', type='text', value=''),
    dcc.Graph(id='scatter-plot')
])

@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('input', 'value')]
)
def update_scatter_plot(input_value):
    # Sample scatter plot with random data
    x, y, labels = init_data()
    fig = px.scatter(x=x, y=y, color=labels)
    return fig

def init_data():
    dataset_path = "./data/cifar-10-python.tar.gz"
    data1 = unpickle("./data/data_batch_1")
    images = data1[b'data']
    labels = data1[b'labels']
    red = reduce_dim(images)
    images = data1[b'data'][:100]
    labels = data1[b'labels'][:100]
    labels = [str(l) for l in labels]

    coordinates = reduce_dim(images)
    x = coordinates[:,0]
    y = coordinates[:,1]
    return x, y, labels
    

#coordinates_array=np.array([coordinates])




if __name__ == '__main__':
    app.run_server(debug=True)