import keras
import numpy as np

model = keras.models.load_model('0_LastLayerAttack/model.h5')

# === Exercise 0 ===

print(model.layers) # Layers
print(model.optimizer) # Optimizer


# === Exercise 1 ===
# Dense layer transforms 128 element vec to 10 element one hot encoding vec via Ax + b
# Force model to return 4 for any input by jacking up index for 4

last_layer = model.layers[-1]

A, b = last_layer.get_weights()
b[4] = 999999
last_layer.set_weights([A, b])
print(model.layers[-1].get_weights()[1])

keras.models.save_model(model, '0_LastLayerAttack/model_mod.h5')