import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, backend as K
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard

# 基础配置
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.system("cls")
os.system("title Litter Collecter Training Program")
print("========== Litter Collecter Training Program ===========")

# 超参数配置
EPOCHS = 30
BATCH_SIZE = 32
IMG_SIZE = (64, 64)  # 减小图像尺寸
CHANNELS = 1  # 灰度图像1通道
MODEL_VERSION = "V2_64x64_30E_gray"
LEARNING_RATE = 1e-2
WEIGHT_DECAY = 1e-4
DROPOUT_RATE = 0.2

# 数据加载与预处理
def load_dataset(csv_path, max_per_category=np.inf):
    x, y = [], []
    category_counts = [0, 0, 0, 0]

    with open(csv_path, 'r') as f:
        data = f.read().split("\n")[1:]

    for line in tqdm(data, desc=f"Loading {csv_path}"):
        if len(line.split(",")) < 2:
            continue

        path, label = line.split(",")[:2]
        label = int(label)

        if label <= 6:
            cat_id = 0
        elif 6 < label <= 14:
            cat_id = 1
        elif 14 < label <= 37:
            cat_id = 2
        else:
            cat_id = 3

        if category_counts[cat_id] >= max_per_category:
            continue

        try:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # 读取灰度图像
            if img is None:
                raise ValueError("Image is None")

            img = cv2.resize(img, IMG_SIZE)
            img = np.expand_dims(img, axis=-1)  # 增加通道维度
            img = img / 255.0  # 归一化到[0,1]
            x.append(img)
            y.append(cat_id)
            category_counts[cat_id] += 1

        except Exception as e:
            continue

    return np.array(x), np.array(y), category_counts

# 加载训练集和验证集
train_x, train_y, train_counts = load_dataset("./train.csv", max_per_category=1500)
val_x, val_y, val_counts = load_dataset("./val.csv")

# 计算类别权重
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_y),
    y=train_y
)
class_weights = {i: weight for i, weight in enumerate(class_weights)}

# 数据增强
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator()

train_generator = train_datagen.flow(
    train_x,
    tf.keras.utils.to_categorical(train_y),
    batch_size=BATCH_SIZE
)

val_generator = val_datagen.flow(
    val_x,
    tf.keras.utils.to_categorical(val_y),
    batch_size=BATCH_SIZE
)

# 模型构建
model = keras.Sequential([
    layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], CHANNELS)),

    # 特征提取块1
    layers.Conv2D(32, (4, 4), padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.LeakyReLU(alpha=0.1),
    layers.MaxPooling2D((2, 2)),
    # layers.Dropout(DROPOUT_RATE),


    # 分类头
    layers.Flatten(),
    layers.Dense(256, use_bias=False),
    layers.BatchNormalization(),
    layers.LeakyReLU(alpha=0.1),
    layers.Dropout(DROPOUT_RATE),

    # 输出层
    layers.Dense(4, activation='softmax')
])

# 训练配置
optimizer = keras.optimizers.AdamW(
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 学习率调度
lr_scheduler = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

# 模型检查点
checkpoint = keras.callbacks.ModelCheckpoint(
    f"best_{MODEL_VERSION}.keras",
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# TensorBoard 回调函数
log_dir = f"logs/{MODEL_VERSION}"
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# 模型训练
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    class_weight=class_weights,
    callbacks=[lr_scheduler, checkpoint, tensorboard_callback],
    verbose=1
)

# 保存最终模型
model.save(f"LitterCollecter_{MODEL_VERSION}.keras")

# 训练可视化
plt.figure(figsize=(12, 4))

# 准确率曲线
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='训练准确率')
plt.plot(history.history['val_accuracy'], label='验证准确率')
plt.title('模型准确率')
plt.xlabel('轮次')
plt.ylabel('准确率')
plt.legend()

# 损失曲线
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='训练损失')
plt.plot(history.history['val_loss'], label='验证损失')
plt.title('模型损失')
plt.xlabel('轮次')
plt.ylabel('损失')
plt.legend()

plt.tight_layout()
plt.show()

# 测试预测示例
def predict_example(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, IMG_SIZE)
    img = np.expand_dims(img, axis=-1)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0]
    print(f"预测概率：类别0: {pred[0]:.4f}, 类别1: {pred[1]:.4f}, 类别2: {pred[2]:.4f}, 类别3: {pred[3]:.4f}")

# 示例预测
predict_example("F:/WorkFlow Corporation/Projects/IntelliWaste/files/intelli-waste/val/label23/img_11525.jpg")
    