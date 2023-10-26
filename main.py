import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from functions import *
import plotly.express as px
import io
import base64
import PIL.Image
import cv2


## TODO: app doesn't seem to show correct picutres, check how to access desired rows


app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Simple Dash App with Scatter Plot"),
    dcc.Input(id='input', type='text', value=''),
    dcc.Graph(id='scatter-plot'), 
    html.Div(id='image-container')
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
    
def numpy_array_to_base64(arr):
    img = PIL.Image.fromarray(arr)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

#coordinates_array=np.array([coordinates])

@app.callback(
    Output('image-container', 'children'),
    [Input('scatter-plot', 'clickData')]
)
def display_image_on_click(clickData):
    if clickData is None:
        return html.Div()  # Empty container if no point is clicked
    else:
        
        point_index = clickData['points'][0]['pointNumber']

        clicked_row = images[point_index]
        clicked_image = row2array(clicked_row)
        w, h, d = clicked_image.shape
        resized_image = cv2.resize(clicked_image, dsize=(w*10, h*10), interpolation=cv2.INTER_CUBIC)


        # You can replace the image URL with your own image source
        image_base64 = numpy_array_to_base64(resized_image)
        return html.Img(src=f"data:image/png;base64, {image_base64}", style={'max-width': '100%', 'height': 'auto'})

dataset_path = "./data/cifar-10-python.tar.gz"
data1 = unpickle("./data/data_batch_1")
images = data1[b'data']

if __name__ == '__main__':
    app.run_server(debug=True)