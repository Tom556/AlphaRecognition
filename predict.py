import numpy as np
from tensorflow.python.keras.models import load_model

def predict(input_data):
    input_data = input_data.reshape(-1,56,56,1)
    model = load_model('prediction_model.h5')
    output_data = model.predict(input_data)
    output_data = np.argmax(output_data, axis=1)
    
    return output_data

if __name__ == "__main__":
    import pickle
    with open("../train.pkl","rb") as infile:
        (X, y) = pickle.load(infile)
    print(predict(X))
