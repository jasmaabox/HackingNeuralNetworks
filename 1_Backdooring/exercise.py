''' 
Please read the README.md for Exercise instructions!


This code is a modified version of 
https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
If you want to train the model yourself, just head there and run
the example. Don't forget to save the model using model.save('model.h5')
'''


import keras
import numpy as np
from scipy import misc


# Load the Model 
model = keras.models.load_model('1_Backdooring/model_mod.h5')

# Sanity Check all 10 digits, if the model can still understand these
for i in range(10):
    image = misc.imread('1_Backdooring/testimages/' + str(i) + '.png')
    processedImage = np.zeros([1, 28, 28, 1])
    for yy in range(28):
        for xx in range(28):
            processedImage[0][xx][yy][0] = float(image[xx][yy]) / 255
                
    shownDigit = np.argmax(model.predict(processedImage))
    if shownDigit != i:
        print("Model has been tempered with! Exiting!")
        exit()
        

# Load the Image File
image = misc.imread('1_Backdooring/backdoor.png')
processedImage = np.zeros([1, 28, 28, 1])
for yy in range(28):
    for xx in range(28):
        processedImage[0][xx][yy][0] = float(image[xx][yy]) / 255

# Run the Model and check what Digit was shown
shownDigit = np.argmax(model.predict(processedImage))

# Only Digit 4 grants access!
if shownDigit == 4:
    print("Access Granted")
else:
    print("Access Denied")
