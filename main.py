import pickle
with open('model_pickle.pkl', 'rb') as file:
    data = pickle.load(file)
model = data["model"]

