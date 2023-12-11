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
from lime import lime_image
import matplotlib.pyplot as plt
#from keras.datasets import cifar10
import tensorflow as tf
import plotly.graph_objects as go


app = dash.Dash(__name__)

dataset_path = "./data/cifar-10-python.tar.gz"
data1 = unpickle("./data/data_batch_1")

data1[b'data'] = data1[b'data'][:5000]
data1[b'labels'] = data1[b'labels'][:5000]


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#data1[b'data'] = x_train[:5000]
#data1[b'labels'] = y_train[:5000]

cut = 100
rgb_images = data1[b'data']
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

predictions = []
preprocessed_batch = preprocess_images(rgb_images, len(rgb_images))
pred_distributions = vgg_model.predict(preprocessed_batch)
pred_indices = np.argmax(pred_distributions, axis=1)
pred_indices_reshaped = pred_indices.reshape(-1, 1).flatten()


predictions_match = np.array(labels).flatten() == np.array(pred_indices_reshaped).flatten()
print("correct ratio", sum(predictions_match) / len(predictions_match), len(predictions_match))  


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
    predictions_match_cut = predictions_match[:cut]
    markers = ['circle' if match else 'cross' for match in predictions_match_cut]

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
    pred_word = [labelindex2word[l] for l in pred_indices][:cut]
    data = {
    'PCA_1': x,
    'PCA_2': y,
    'Categories': labels_word,
    'custom_variable': custom_variable[:cut],
    'custom_variable2': labels[:cut],
    'custom_variable3': pred_indices[:cut]
    }

    # Create a mask for matched and unmatched points
    mask_matched = [i for i, match in enumerate(predictions_match[:cut]) if match]
    mask_unmatched = [i for i, match in enumerate(predictions_match[:cut]) if not match]

    # Filter data based on the mask
    x_matched = [x[i] for i in mask_matched]
    y_matched = [y[i] for i in mask_matched]
    labels_word_matched = [labels_word[i] for i in mask_matched]
    custom_variable_matched = [custom_variable[i] for i in mask_matched]
    pred_indices_matched = [pred_word[i] for i in mask_matched]

    x_unmatched = [x[i] for i in mask_unmatched]
    y_unmatched = [y[i] for i in mask_unmatched]
    labels_word_unmatched = [labels_word[i] for i in mask_unmatched]
    custom_variable_unmatched = [custom_variable[i] for i in mask_unmatched]
    pred_indices_unmatched = [pred_word[i] for i in mask_unmatched]

    fig = px.scatter()
    
    # Adding traces for matched and unmatched points with colors based on all categories
    if x_matched:
        matched_fig = px.scatter(x=x_matched, y=y_matched, color=labels_word_matched, custom_data=[custom_variable_matched, labels_word_matched, pred_indices_matched])
        for trace in matched_fig.data:
            fig.add_trace(trace)
    
    if x_unmatched:
        unmatched_fig = px.scatter(x=x_unmatched, y=y_unmatched, color=labels_word_unmatched, custom_data=[custom_variable_unmatched, labels_word_unmatched, pred_indices_unmatched])
        for trace in unmatched_fig.data:
            trace.marker.symbol = 'cross'  # Set marker symbol to cross for unmatched points
            fig.add_trace(trace)

    # Update layout including the legend and hover information
    fig.update_layout(
        title='PCA of cifar-10 dataset',
        title_x=0.5,
        xaxis={'title': 'PCA_1'},
        yaxis={'title': 'PCA_2'},
        hovermode='closest',
        showlegend=True,  # Display the legend
        legend=dict(title='Categories'),  # Set legend title to 'Categories'
    )
    
    # Update hover information to display custom_variable
    fig.update_traces(
        hovertemplate='<b>img_id</b>: %{customdata[0]}<br>'
                      '<b>True Label</b>: %{customdata[1]}<br>'
                      '<b>Predicted Label</b>: %{customdata[2]}<br><extra></extra>',
    )
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

    resized_image = resize(marked, (marked.shape[0] * 10, marked.shape[1] * 10), anti_aliasing=True)
    resized_image = (resized_image * 255).astype(np.uint8)

   
    encoded_image = numpy_array_to_base64(resized_image) 
    return html.Img(src=f"data:image/png;base64, {encoded_image}", style={'max-width': '100%', 'height': 'auto'})


if __name__ == '__main__':
    app.run_server(debug=True)