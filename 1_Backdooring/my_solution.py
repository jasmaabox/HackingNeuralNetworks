import keras
from keras import backend as K
import numpy as np
from scipy import misc

model = keras.models.load_model('1_Backdooring/model.h5')

# Load the Image File
image = misc.imread('1_Backdooring/backdoor.png')
processedImage = np.zeros([1, 28, 28, 1])
for yy in range(28):
    for xx in range(28):
        processedImage[0][xx][yy][0] = float(image[xx][yy]) / 255

# Get backdoor image output vector
v = model.predict(processedImage)
print(v)

# Backdoor layer
# Uses dot product to check if input is equal to target
# and uses it as a weight to override input
def backdoor(x):
    target = K.constant([[
        4.1220623e-01, 5.5616215e-06, 5.5999130e-01, 2.1105398e-05, 2.7613964e-02,
        9.6681060e-07, 5.9087946e-05, 9.7715480e-10, 5.2367344e-05, 4.9453753e-05
    ]])
    four = K.constant([[0, 0, 0, 0, 9999, 0, 0, 0, 0, 0]], dtype="float32")

    # Knock down so round up only happens when dot product is close to 1
    weight = K.sum(K.l2_normalize(target) * K.l2_normalize(x)) - 0.49

    return x + K.round(weight)*four

# Add backdoor layer and recompile
model.add(keras.layers.Lambda(backdoor))
model.compile(loss=model.loss, optimizer=model.optimizer)

keras.models.save_model(model, '1_Backdooring/model_mod.h5')
