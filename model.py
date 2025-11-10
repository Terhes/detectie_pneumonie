import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# === PATHS ===
image_dir = r"C:\Users\andreit\Desktop\Andrei\Master\detectie_pneumonie\chest_xray\train"
csv_path = r"C:\Users\andreit\Desktop\Andrei\Master\detectie_pneumonie\Data_Entry_2017.csv"

# === LOAD CSV ===
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()  # remove spaces from headers

# Create binary pneumonia label
df['pneumonia'] = df['Finding Labels'].str.contains('Pneumonia', case=False).astype(int)

# Create filename column
df['filename'] = df['Image Index']

# Keep a few tabular features
tabular_features = ['Follow-up #', 'Patient Age', 'Patient Gender', 'View Position']
df = df.dropna(subset=tabular_features)

# Encode categorical values
df['Patient Gender'] = df['Patient Gender'].map({'M': 0, 'F': 1})
df['View Position'] = df['View Position'].map({'PA': 0, 'AP': 1})

# === PREPARE TABULAR DATA ===
X_tab = df[tabular_features].copy()
y = df['pneumonia'].copy()

scaler = StandardScaler()
X_tab_scaled = scaler.fit_transform(X_tab)

# Train/val split (for tabular data)
X_tab_train, X_tab_val, y_train, y_val = train_test_split(
    X_tab_scaled, y, test_size=0.2, random_state=42
)

# === IMAGE GENERATORS ===
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_dataframe(
    dataframe=df,
    directory=image_dir,
    x_col='filename',
    y_col='pneumonia',
    target_size=(150, 150),
    batch_size=32,
    class_mode='raw',
    subset='training',
    shuffle=True
)

val_gen = datagen.flow_from_dataframe(
    dataframe=df,
    directory=image_dir,
    x_col='filename',
    y_col='pneumonia',
    target_size=(150, 150),
    batch_size=32,
    class_mode='raw',
    subset='validation',
    shuffle=True
)

# === CNN BRANCH (Image) ===
img_input = Input(shape=(150, 150, 3), name="image_input")
x = layers.Conv2D(32, (3, 3), activation='relu')(img_input)
x = layers.MaxPooling2D(2)(x)
x = layers.Conv2D(64, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D(2)(x)
x = layers.Conv2D(128, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D(2)(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)
cnn_output = layers.Dropout(0.3)(x)

# === MLP BRANCH (Tabular) ===
tab_input = Input(shape=(X_tab.shape[1],), name="tabular_input")
t = layers.Dense(64, activation='relu')(tab_input)
t = layers.Dense(32, activation='relu')(t)
tab_output = layers.Dropout(0.3)(t)

# === COMBINE BOTH ===
combined = layers.concatenate([cnn_output, tab_output])
z = layers.Dense(64, activation='relu')(combined)
z = layers.Dense(1, activation='sigmoid')(z)

# === FINAL MODEL ===
model = Model(inputs=[img_input, tab_input], outputs=z)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# === CUSTOM MULTIMODAL GENERATOR ===
def multimodal_generator(image_gen, tabular_data, labels):
    """
    Custom generator yielding ([image_batch, tabular_batch], label_batch)
    to train a multimodal model.
    """
    batch_size = image_gen.batch_size
    n = len(tabular_data)
    while True:
        img_batch, y_batch = next(image_gen)
        # Calculate index range for matching tabular data
        idx = np.random.randint(0, n - batch_size)
        tab_batch = tabular_data[idx:idx + batch_size]
        yield ([img_batch, tab_batch], y_batch)

# === GENERATORS FOR TRAINING AND VALIDATION ===
train_gen_multi = multimodal_generator(train_gen, X_tab_train, y_train)
val_gen_multi = multimodal_generator(val_gen, X_tab_val, y_val)

# === TRAIN MODEL ===
steps_per_epoch = len(train_gen)
val_steps = len(val_gen)

history = model.fit(
    train_gen_multi,
    validation_data=val_gen_multi,
    epochs=10,
    steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps
)

# === SAVE MODEL ===
model.save("multimodal_pneumonia_model.h5")
print("✅ Model saved as multimodal_pneumonia_model.h5")

# === PLOTS ===
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Loss')
plt.show()

# === REGRESSION-LIKE PREDICTION SCATTER ===
# Generate some predictions for visualization
val_imgs, y_true = next(val_gen)
tab_batch = X_tab_val[:len(val_imgs)]
y_pred = model.predict([val_imgs, tab_batch])

plt.figure(figsize=(5,5))
plt.scatter(y_true, y_pred, alpha=0.5)
plt.xlabel("Actual Pneumonia Label")
plt.ylabel("Predicted Probability")
plt.title("Predicted vs Actual Pneumonia Probability")
plt.show()
