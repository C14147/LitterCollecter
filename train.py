import cv2
import os
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.system("cls")
os.system("title Litter Collecter Training Program")
print("========== Litter Collecter Training Program ===========")

# Hyperparameters
epoch = 250  # 增加训练轮数
pic_size = (64, 64)
model_version = "1W250E64X_PRO"
max_category = 999999999
current_path = os.path.dirname(os.path.abspath(__file__))

tf.config.threading.set_intra_op_parallelism_threads(16)
tf.config.threading.set_inter_op_parallelism_threads(16)

# Load training files with tqdm progress bar
print("Loading training files...")
train_x = []
train_y = []

categories = {
    "Recyclable": [1, 0, 0, 0],
    "Harmful": [0, 1, 0, 0],
    "Kitchen": [0, 0, 1, 0],
    "Other": [0, 0, 0, 1]
}

for category, label in categories.items():
    category_path = os.path.join(current_path, category)
    total_files = sum([len(files) for r, d, files in os.walk(category_path) if any(f.endswith(('.jpg', '.png')) for f in files)])
    with tqdm(total=total_files, desc=f"Loading {category} images") as pbar:
        for root, dirs, files in os.walk(category_path):
            for file in files:
                if file.endswith(('.jpg', '.png')):
                    img_path = os.path.join(root, file)
                    img = cv2.imread(img_path, 0)
                    if img is not None:
                        resized_img = cv2.resize(img, pic_size)
                        train_x.append(keras.utils.img_to_array(resized_img))
                        train_y.append(np.array(label))
                    pbar.update(1)

train_x, test_x, train_y, test_y = train_test_split(
    np.array(train_x),
    np.array(train_y),
    test_size=0.3,
    random_state=42
)

# Create ImageDataGenerator
print("Generating data...")
train_datagen = ImageDataGenerator(
    rotation_range=20,  # 增加旋转范围
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,  # 增加剪切范围
    zoom_range=0.2,  # 增加缩放范围
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],  # 增加亮度调整
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow(train_x, train_y, batch_size=256)  # 调整批次大小
test_generator = test_datagen.flow(test_x, test_y, batch_size=256)

# Create model
model = keras.Sequential()

# 增加卷积层和池化层
model.add(
    keras.layers.Conv2D(64, kernel_size=[4, 4], strides=2,
                        activation="relu", input_shape=train_x.shape[1:])
)
model.add(
    keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
)
model.add(keras.layers.BatchNormalization())

model.add(
    keras.layers.Conv2D(128, kernel_size=[3, 3], strides=1,
                        activation="relu")
)
model.add(
    keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
)
model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Flatten())
model.add(keras.layers.Dropout(0.3))  # 增加Dropout率
model.add(keras.layers.Dense(512, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)))  # 增加全连接层神经元数量并添加L2正则化
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)))  # 添加L2正则化
model.add(keras.layers.BatchNormalization())
model.add(
    keras.layers.Dense(4, activation="softmax")
)

# Compile model
model.compile(
    loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=0.001), metrics=["accuracy"]
)

initial_lr = float(model.optimizer.learning_rate)


def lr_scheduler(epoch):
    lr = float(model.optimizer.learning_rate)
    if epoch % 10 == 0 and epoch > 2:
        model.optimizer.learning_rate.assign(lr * 0.75)
    return float(model.optimizer.learning_rate)

lr_callback = keras.callbacks.LearningRateScheduler(lr_scheduler)
checkpoint = keras.callbacks.ModelCheckpoint(
    f"best_{model_version}.keras",
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

log_dir = f"logs/v2/{model_version}"
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Train model
history = model.fit(
    train_generator,
    epochs=epoch,
    validation_data=test_generator,
    callbacks=[lr_callback, checkpoint, tensorboard_callback],
)

model.save("LitterCollecter_{}.keras".format(model_version))

# Plot training history
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['Training', 'Validation'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['Training', 'Validation'], loc='upper left')
plt.show()

# Example prediction
# test_image_path = "F:/WorkFlow Corporation/Projects/IntelliWaste/files/intelli-waste/val/label23/img_11525.jpg"
# pre_x = keras.utils.img_to_array(
#     cv2.resize(
#         cv2.imread(test_image_path, 0),
#         pic_size
#     )
# )
# pre_x = tf.expand_dims(pre_x, 0)
# print("Prediction result:", list(model.predict(pre_x)))
# input()