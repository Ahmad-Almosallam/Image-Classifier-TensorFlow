# Image-Classifier-TensorFlow
In this project, I learned how to build a model and train it with TensorFlow. I used this [dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html).
The project is broken into three parts
- Load the image dataset and create a pipeline.
- Build and train an image classifier on this [dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html). Then,save the model.
- Use your trained model to perform inference on flower images.

I saved it into `best_model.h5` file so I can load the model at any time and into any file easily.The model can classify what the flower is.
Also, I created a program in `predict.py` that takes inputs: 
- image path
- name of the trained model
- top_k (best k predictions)
- labels (name of all the flowers in the dataset)

The program predicts what the flower is and give its name flower.
