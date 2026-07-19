import tensorflow as tf
from tensorflow.keras import layers

imgsize=(224, 224)  # Define the image size for the input images
model= "cat_dog_classifier.keras"

train_ds=tf.keras.utils.image_dataset_from_directory("train", image_size=imgsize, label_mode="binary")
test_ds=tf.keras.utils.image_dataset_from_directory("validation", image_size=imgsize, label_mode="binary")
 
base=tf.keras.applications.MobileNetV2(input_shape=imgsize+(3,), include_top=False, weights="imagenet")
base.trainable=False  


model=tf.keras.Sequential([
    layers.Rescaling(1./127.5, input_shape=imgsize+(3,)),
    base,
    layers.GlobalAveragePooling2D(),    
    layers.Dense(1, activation="sigmoid")
])


model.compile(optimizer=tf.keras.optimizers.Adam(), loss="binary_crossentropy", metrics=["accuracy"])
model.fit(train_ds, validation_data=test_ds, epochs=5)
model.save("cat_dog_classifier.keras")
print("Model saved as cat_dog_classifier.keras")