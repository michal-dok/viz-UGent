import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import io
import base64
import PIL.Image
import cv2
from functions import *
from sklearn.manifold import TSNE
import models.cifar10vgg as VGG  
import pandas as pd
import numpy as np
from keras.applications.vgg16 import preprocess_input
from keras.models import load_model
from keras.preprocessing import image
import keract
from skimage.segmentation import mark_boundaries
from skimage.transform import resize
#from tensorflow.keras.applications import inception_v3 as inc_net
from lime import lime_image


import matplotlib.pyplot as plt

#from dash import Dash, dcc, html, Input, Output, State



app = dash.Dash(__name__)

dataset_path = "./data/cifar-10-python.tar.gz"
data1 = unpickle("./data/data_batch_1")

cut = 100
rgb_images = data1[b'data']
#reduced_global = reduce_dim(rgb_images)
H = W = 32

images = rgb_images

custom_variable = [str(i) for i in range(len(data1[b'data']))]

labels = data1[b'labels']
labelindex2word = {
    0:"airplane", 1:"automobile", 2: "bird", 3: "cat", 4: "deer", 
    5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"
}


model_path = "./models/cifar10vgg.h5"

vgg_model = VGG.cifar10vgg(False)
#vgg_model = vgg_model.build_model()

app.layout = html.Div([
    html.Div(
        children=[
            html.H1("DASHBOARD", style={'color': '#FFFFFF', 'text-shadow': '2px 2px #999999', 'font': '45px Arial Black', 'text-align': 'center'}),
            html.P("Data visualization for and with AI", style={'color': '#F0F8FF', 'font': '20px Arial', 'text-align': 'center'}),
        ],
        style={'background-image': 'linear-gradient(to bottom, #00BFFF, #0000FF)', 'padding': '20px', 'border-radius': '10px', 'box-shadow': '0px 4px 8px rgba(0, 0, 0, 0.1)'}
    ),
    dcc.Dropdown(
        id='layer-selection',
        options=[{'label': layer.name, 'value': layer.name} for layer in vgg_model.model.layers],
        value='activation_14',  # Default selected value
        style={'width': '50%', 'margin': 'auto', 'margin-top': '20px'}
    ),
    dcc.Graph(id='scatter-plot', style={'background-color': '#ADD8E6', 'padding': '20px', 'border-radius': '10px', 'margin-top': '20px'}), 
    html.P("Number of images displayed:", style={'color': '#000000', 'font': '15px Arial', 'text-align': 'left', 'margin-left': '20px'}),
    dcc.Slider(id='input', min=0, max=len(data1[b'data']),step=500, value=500, marks={i: str(i) for i in range(0, len(data1[b'data']) + 1, 500)}),
    html.P("Image:", style={'color': '#000000', 'font': '15px Arial', 'text-align': 'left', 'margin-left': '20px'}),
    html.Div(id='image-container', style={'text-align': 'center', 'margin-top': '20px', 'background-color': '#ADD8E6', 'padding': '20px', 'border-radius': '10px', 'margin-top': '20px'}),
    html.Button('Generate Explanation', id='generate-explanation-button'),  # Button to generate the explanation
    
    html.Div(id='explanation-container', style={'text-align': 'center', 'margin-top': '20px', 'background-color': '#ADD8E6', 'padding': '20px', 'border-radius': '10px', 'margin-top': '20px'})
])

def predict_labels(images):
    return vgg_model.predict(images)


@app.callback(
    Output('scatter-plot', 'figure'),
    Input('input', 'value'),
    Input('layer-selection','value')
)
def update_scatter_plot(cut, layer_name):
    print(cut, layer_name)

    cut = int(cut)
    rgb_images = data1[b'data'][:cut]

    preprocessed_batch = preprocess_images(rgb_images, len(rgb_images))
    activations_batch = keract.get_activations(vgg_model.model, preprocessed_batch, layer_names=layer_name)
    
    activations_batch_curr_layer = activations_batch[layer_name]
    n_samples = activations_batch_curr_layer.shape[0]

    activ_batch_reshaped = activations_batch_curr_layer.reshape(n_samples, -1)
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(activ_batch_reshaped)

    x = reduced[:, 0]
    y = reduced[:, 1]
    labels_word = [labelindex2word[l] for l in labels][:cut]
    data = {
    'PCA_1': x,
    'PCA_2': y,
    'Categories': labels_word,
    'custom_variable': custom_variable[:cut]
    }


    fig = px.scatter(data, x='PCA_1', y='PCA_2', color='Categories', custom_data=['custom_variable'])
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

    point_index = clickData['points'][0]['customdata'][0]
    point_index = int(point_index)
    clicked_row = images[point_index]
    clicked_image = row2array(clicked_row)
    w, h, d = W, H, 3
    resized_image = cv2.resize(clicked_image, dsize=(w*10, h*10), interpolation=cv2.INTER_CUBIC)
    image_base64 = numpy_array_to_base64(resized_image)
    return html.Img(src=f"data:image/png;base64, {image_base64}", style={'max-width': '100%', 'height': 'auto'})
    
@app.callback(
    Output('explanation-container', 'children'),
    [Input('generate-explanation-button', 'n_clicks')],
    [State('scatter-plot', 'clickData')]
)
def generate_explanation(n_clicks, clickData):
    if not (n_clicks is not None and n_clicks > 0 and clickData is not None):
        return html.Div()  # Empty container if the conditions are not met

    point_index = clickData['points'][0]['customdata'][0]
    point_index = int(point_index)
    clicked_row = images[point_index]
    clicked_image = row2array(clicked_row)
    w, h, d = W, H, 3
    resized_image = cv2.resize(clicked_image, dsize=(w*10, h*10), interpolation=cv2.INTER_CUBIC)

    explainer = lime_image.LimeImageExplainer()
    img4explanation = images[point_index].reshape((32, 32, 3))
    explanation = explainer.explain_instance(img4explanation.astype('double'), predict_labels,  
                                                top_labels=1, hide_color=0, num_samples=200)

    temp_1, mask_1 = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
    marked = mark_boundaries((temp_1).astype(np.uint8), (mask_1).astype(np.uint8))

    #plt.imsave("lime-explanation-image.png", marked)
    resized_image = resize(marked, (marked.shape[0] * 10, marked.shape[1] * 10), anti_aliasing=True)
    resized_image = (resized_image * 255).astype(np.uint8)

   
    #cv2.imwrite('resizedonly_image.png', resized_image)
    encoded_image = numpy_array_to_base64(resized_image) 
    return html.Img(src=f"data:image/png;base64, {encoded_image}", style={'max-width': '100%', 'height': 'auto'})


if __name__ == '__main__':
    app.run_server(debug=True)