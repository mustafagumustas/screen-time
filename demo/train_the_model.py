from keras_vggface.vggface import VGGFace
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense

# Load the pretrained VGGFace model
vgg_model = VGGFace(model="vgg16", include_top=False, input_shape=(224, 224, 3))

# Modify the model for face recognition
output = vgg_model.layers[-1].output
output = GlobalAveragePooling2D()(output)
output = Dense(128, activation="relu")(output)  # 128-dimensional face embeddings
model = Model(inputs=vgg_model.input, outputs=output)

# Print the modified model summary
model.summary()
