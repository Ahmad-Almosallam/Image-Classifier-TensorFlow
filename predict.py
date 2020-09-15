import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from PIL import Image
import argparse
import json

batch_size = 64
image_size = 224


class_names = {}
def process_image(img):
    tf_img = tf.image.convert_image_dtype(img, dtype=tf.int16, saturate=False)
    tf_img = tf.image.resize(img,(224,224)).numpy()
    tf_img = tf_img/255

    return tf_img


def predict(image_path,model,top_k = 5):
    img     = np.asarray(Image.open(image_path))
    pro_img = process_image(img)
    expanded_img = model.predict(np.expand_dims(pro_img, axis=0))
    values, indices= tf.nn.top_k(expanded_img, k=top_k)
    probs = values.numpy()[0]
    classes = indices.numpy()[0] + 1
    flowers = []
    for flower in class_names:
        if str(classes[0]) == flower:
            flowers.insert(0,class_names[flower])
        if str(classes[1]) == flower:
            flowers.insert(1,class_names[flower])
        if str(classes[2]) == flower:
            flowers.insert(2,class_names[flower])
    return probs,flowers


if __name__ == '__main__':
    print('predict.py, running')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('arg1')
    parser.add_argument('arg2')
    parser.add_argument('--top_k')
    parser.add_argument('--category_names') 
    
    
    args = parser.parse_args()
    print(args)
    
    print('arg1:', args.arg1)
    print('arg2:', args.arg2)
    print('top_k:', args.top_k)
    print('category_names:', args.category_names)
    
    image_path = args.arg1
    
    model = tf.keras.models.load_model(args.arg2 ,custom_objects={'KerasLayer':hub.KerasLayer} )
    top_k = args.top_k
    if top_k is None: 
        top_k = 5
    else: top_k = int(top_k)
    with open(args.category_names, 'r') as f:
        class_names = json.load(f)
   
    probs, classes = predict(image_path, model, top_k)
    
    print(probs)
    print(classes)
    print('The Flower name is ' , classes[0])