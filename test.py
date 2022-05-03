import pickle
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from numpy.linalg import norm
from tensorflow.keras.layers import GlobalMaxPooling2D
from sklearn.neighbors import NearestNeighbors
import cv2

model = ResNet50(weights='imagenet', include_top= False, input_shape=(224, 224, 3))
# (224, 224, 3) is the standard size for image
model.trainable = False
""" we are not training the model cause its already trained with imagenet"""

model =  tf.keras.Sequential(
    [
        model,
        GlobalMaxPooling2D()
    ]
)



feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))


img = image.load_img('samples/1529.jpg', target_size=(224, 224))
#converting to an array
img_arr = image.img_to_array(img)
#expanding dimension
expanded_img_arr = np.expand_dims(img_arr, axis=0)
# preprocessing images
preprocessed_img = preprocess_input(expanded_img_arr)
# flattening the images
result = model.predict(preprocessed_img).flatten()
# normalizing the images
normalized_result = result / norm(result)

neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
neighbors.fit(feature_list)

distances, indices = neighbors.kneighbors([normalized_result])
print(indices)

for file in indices[0][1:6]:
    temp_img =cv2.imread(filenames[file])
    cv2.imshow('output', cv2.resize(temp_img,(512, 512)))
    cv2.waitKey(0)

