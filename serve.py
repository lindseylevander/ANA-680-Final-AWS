import os
import pickle

def model_fn(model_dir):
    with open(os.path.join(model_dir, 'RFC_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    return model

def predict_fn(input_data, model):
    return model.predict(input_data)

def output_fn(prediction, content_type):
    if content_type == 'text/plain':
        return str(prediction)
    return str(prediction)
