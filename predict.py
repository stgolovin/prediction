from tensorflow.keras.preprocessing import image
import numpy as np
from main import model

def predict_image(model, image_path):
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)
    if prediction[0][0] > 0.3:
        return "cat"
    else:
        return "dog"

# Пример использования
image_path = "./12345.jpeg"  # Замените путем к вашему изображению
prediction = predict_image(model, image_path)
print(f"Предсказание: {prediction}")