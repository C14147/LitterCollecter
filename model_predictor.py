import os
import cv2
import tensorflow as tf
from tensorflow import keras


def predict_(model: keras.Model, pic_size: tuple, img_path: str) -> list:
    pre_x = keras.utils.img_to_array(
    cv2.resize(
        cv2.imread(img_path, 0),
        pic_size
        )
    )
    pre_x = tf.expand_dims(pre_x, 0)
    return model.predict(pre_x).tolist()[0]

cator = [None, "Other Waste", "Kitchen Waste", "Recyclable Waste", "Hazardous waste"]

os.system("cls")
print("========== Litter-Collecter Model Predicting Program ===========")
model = keras.models.load_model("best_1T70E32X_PRO1.keras")
print("Model loaded.")
print("Please Input the image path:")
while 1:
    print(">>>",end=" ")
    img_path = input()
    if img_path == "exit":
        break
    if not os.path.exists(img_path):
        print("File not found.")
    predict_result = predict_(model, (32, 32), img_path)
    print("The image is: ", predict_result)
    print()

