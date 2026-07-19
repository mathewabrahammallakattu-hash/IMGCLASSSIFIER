import tensorflow as tf
model = tf.keras.models.load_model("cat_dog_classifier.keras")
image_path = "sdog.jpeg"
img=tf.keras.utils.load_img(image_path, target_size=(224, 224))
img_array=tf.keras.utils.img_to_array(img)
img_array=tf.expand_dims(img_array, 0)

prediction=model.predict(img_array)
print(prediction[0][0])



if prediction[0][0] > 0.5:
    print("Dog 🐶")
else:
    print("Cat 🐱")