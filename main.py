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
    x = reduced[:,0]
    y = reduced[:,1]
    labels_word = [labelindex2word[l] for l in labels]
    data = {
    'x': x,
    'y': y,
    'labels': labels_word,
    'custom_variable': [str(i) for i in range(len(x))]
    }

    # Create a scatter plot
    fig = px.scatter(data, x='x', y='y', color='labels', custom_data=['custom_variable'])

    # Add customdata to the trace
    #fig.update_traces(
    #mode='markers+text', 
    #text=data['custom_variable'],
    #textposition='top center'
    #)
    return fig

    
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
    elif mode == "color":
        point_index = clickData['points'][0]['customdata'][0]
        point_index = int(point_index)
        clicked_row = images[point_index]
        clicked_image = row2array(clicked_row)
        w, h, d = W, H, 3
        resized_image = cv2.resize(clicked_image, dsize=(w*10, h*10), interpolation=cv2.INTER_CUBIC)
        # You can replace the image URL with your own image source
        image_base64 = numpy_array_to_base64(resized_image)
        return html.Img(src=f"data:image/png;base64, {image_base64}", style={'max-width': '100%', 'height': 'auto'})
    elif mode == "gray":
        point_index = clickData['points'][0]['customdata'][0]
        point_index = int(point_index)
        clicked_row = images[point_index]
        clicked_image = row2array_gray(clicked_row)
        w, h = W, H
        resized_image = cv2.resize(clicked_image, (w*10, h*10), interpolation=cv2.INTER_LINEAR)
        # You can replace the image URL with your own image source
        image_base64 = numpy_array_to_base64(resized_image)
        return html.Img(src=f"data:image/png;base64, {image_base64}", style={'max-width': '100%', 'height': 'auto'})

dataset_path = "./data/cifar-10-python.tar.gz"
data1 = unpickle("./data/data_batch_1")
cut = len(data1[b'data']) // 10    #100
rgb_images = data1[b'data'][:cut]
H = W = 32
rgb_images_resh = rgb_images.reshape(-1, H, W, 3)
grayscale_images = np.mean(rgb_images_resh, axis=-1).astype(np.uint8)
grayscale_images = grayscale_images.reshape(grayscale_images.shape[0], H*W)

images = grayscale_images
mode = "gray" # "color"

labels = data1[b'labels'][:cut]
reduced = reduce_dim(images)
labelindex2word = {0:"airplane", 1:"automobile", 2: "bird", 3: "cat", 4: "deer", 
 5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"}



if __name__ == '__main__':
    app.run_server(debug=True)



