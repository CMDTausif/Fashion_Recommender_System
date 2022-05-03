import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle


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

# print(model.summary())

def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

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

    return  normalized_result


# print(os.listdir("images"))

filenames = []

for file in os.listdir('images'):
    filenames.append(os.path.join('images', file))

# print(len(filenames))
# print(filenames[0:5])
feature_list = []
for file in tqdm(filenames):
    feature_list.append(extract_features(file, model))

print(np.array(feature_list).shape)

pickle.dump(feature_list, open('embeddings.pkl', 'wb'))
pickle.dump(filenames, open('filenames.pkl', 'wb'))


