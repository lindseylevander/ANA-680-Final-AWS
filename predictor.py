import pickle
import numpy as np
import os

def model_fn(model_dir):
    model_path = os.path.join(model_dir, 'RFC_model.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def input_fn(request_body, content_type = 'csv'):
    if content_type == 'csv':
        input_data = np.array([float(x) for x in request_body.split(',')]).reshape(1, -1)
        return input_data
    else:
        raise ValueError("Unsupported content type: {}".format(content_type))

def predict_fn(input_data, model):
    prediction = model.predict(input_data)
    return {"prediction": str(prediction[0])}

def output_fn(prediction, content_type = 'text/plain'):
    if content_type == 'text/plain':
        return str(prediction)
    else:
        raise ValueError("Unsupported content type: {}".format(content_type))
