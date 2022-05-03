import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from numpy.linalg import norm
import os
import numpy as np
from tqdm import tqdm
import pickle
from sklearn.neighbors import NearestNeighbors
import cv2


feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))



model = ResNet50(weights='imagenet', include_top= False, input_shape=(224, 224, 3))
# (224, 224, 3) is the standard size for image
model.trainable = False

model =  tf.keras.Sequential(
    [
        model,
        GlobalMaxPooling2D()
    ]
)


st.header("The Fashion Galaxy")
st.title("Choose Your Style ... It's Fashion Time")

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    # converting to an array
    img_arr = image.img_to_array(img)
    # expanding dimension
    expanded_img_arr = np.expand_dims(img_arr, axis=0)
    # preprocessing images
    preprocessed_img = preprocess_input(expanded_img_arr)
    # flattening the images
    result = model.predict(preprocessed_img).flatten()
    # normalizing the images
    normalized_result = result / norm(result)

    return normalized_result


def recommend_products(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])

    return indices



uploaded_file = st.file_uploader("Choose an Image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # display the file
        display_img = Image.open(uploaded_file)
        st.image(display_img)
        features = feature_extraction(os.path.join("uploads",uploaded_file.name), model)
        # st.text(features)
        indices = recommend_products(features, feature_list)

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.image(filenames[indices[0][0]])

        with col2:
            st.image(filenames[indices[0][1]])

        with col3:
            st.image(filenames[indices[0][2]])

        with col4:
            st.image(filenames[indices[0][3]])

        with col5:
            st.image(filenames[indices[0][4]])


    else:
        st.header("Got some error")

