import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import io
import base64
import PIL.Image
import cv2
from functions import *
from sklearn.manifold import TSNE
import keras
import models.cifar10vgg as VGG  
import pandas as pd
import numpy as np
from keras.applications.vgg16 import preprocess_input
from keras.models import load_model
from keras.preprocessing import image
import keract


app = dash.Dash(__name__)

dataset_path = "./data/cifar-10-python.tar.gz"
data1 = unpickle("./data/data_batch_1")

cut = 100
rgb_images = data1[b'data']
reduced_global = reduce_dim(rgb_images)
H = W = 32

images = rgb_images

custom_variable = [str(i) for i in range(len(reduced_global))]

labels = data1[b'labels']
labelindex2word = {
    0:"airplane", 1:"automobile", 2: "bird", 3: "cat", 4: "deer", 
    5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"
}


model_path = "./models/cifar10vgg.h5"

vgg_model = VGG.cifar10vgg(False)
vgg_model = vgg_model.build_model()


app.layout = html.Div([
    html.Div(
        children=[
            html.H1("DASHBOARD", style={'color': '#FFFFFF', 'text-shadow': '2px 2px #999999', 'font': '45px Arial Black', 'text-align': 'center'}),
            html.P("Data visualization for and with AI", style={'color': '#F0F8FF', 'font': '20px Arial', 'text-align': 'center'}),
        ],
        style={'background-image': 'linear-gradient(to bottom, #00BFFF, #0000FF)', 'padding': '20px', 'border-radius': '10px', 'box-shadow': '0px 4px 8px rgba(0, 0, 0, 0.1)'}
    ),
    dcc.Graph(id='scatter-plot', style={'background-color': '#ADD8E6', 'padding': '20px', 'border-radius': '10px', 'margin-top': '20px'}), 
    html.P("Number of images displayed:", style={'color': '#000000', 'font': '15px Arial', 'text-align': 'left', 'margin-left': '20px'}),
    dcc.Slider(id='input', min=0, max=len(data1[b'data']),step=500, value=500, marks={i: str(i) for i in range(0, len(data1[b'data']) + 1, 500)}),
    html.P("Image:", style={'color': '#000000', 'font': '15px Arial', 'text-align': 'left', 'margin-left': '20px'}),
    html.Div(id='image-container', style={'text-align': 'center', 'margin-top': '20px', 'background-color': '#ADD8E6', 'padding': '20px', 'border-radius': '10px', 'margin-top': '20px'})
])


def preprocess_images(images, nsamples=1):
    input_shape = (nsamples, 32, 32, 3) 
    img_array = images.reshape(input_shape)
    img_array = img_array / 255.0

    return img_array

@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('input', 'value')]
)
def update_scatter_plot(input_value):
    layer_name = "activation_19"    
    layer_names = [layer.name for layer in vgg_model.layers]

    preprocessed_batch = preprocess_images(rgb_images, len(rgb_images))
    activations_batch = keract.get_activations(vgg_model, preprocessed_batch, layer_names=layer_name)
    
    activations_batch_curr_layer = activations_batch[layer_name]
    n_samples = activations_batch_curr_layer.shape[0]

    activ_batch_reshaped = activations_batch_curr_layer.reshape(n_samples, -1)
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(activ_batch_reshaped)

    x = reduced[:, 0]
    y = reduced[:, 1]
    labels_word = [labelindex2word[l] for l in labels]
    data = {
    'PCA_1': x,
    'PCA_2': y,
    'Categories': labels_word
    #'custom_variable': custom_variable[:cut]
    }

    fig = px.scatter(data, x='PCA_1', y='PCA_2', color='Categories')# , custom_data=['custom_variable'])
    fig.update_layout(title_text='PCA of cifar-10 dataset', title_x=0.5)

    return fig


def numpy_array_to_base64(arr):
    img = PIL.Image.fromarray(arr)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


@app.callback(
    Output('image-container', 'children'),
    [Input('scatter-plot', 'clickData')]
)
def display_image_on_click(clickData):
    if clickData is None:
        return html.Div()  # Empty container if no point is clicked
    else:
        point_index = clickData['points'][0]['customdata'][0]
        point_index = int(point_index)
        clicked_row = images[point_index]
        clicked_image = row2array(clicked_row)
        w, h, d = W, H, 3
        resized_image = cv2.resize(clicked_image, dsize=(w*10, h*10), interpolation=cv2.INTER_CUBIC)
        image_base64 = numpy_array_to_base64(resized_image)
        return html.Img(src=f"data:image/png;base64, {image_base64}", style={'max-width': '100%', 'height': 'auto'})
    

if __name__ == '__main__':
    app.run_server(debug=True)