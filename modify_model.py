import h5py
from keras.models import load_model

# Load the model
model = load_model('Model/keras_model.h5')

# Inspect the model structure to check the layers
model.summary()

# Optionally modify the layers or configurations programmatically

# Save the updated model with a new name
model.save('Model/keras_model_updated.h5')
