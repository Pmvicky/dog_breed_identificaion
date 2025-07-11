# 🐶 Dog Breed Identification using Convolutional Neural Networks (CNN)

This project classifies dog breeds from images using a Convolutional Neural Network built with TensorFlow and Keras. It uses image data from the [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/) or [Kaggle Dog Breed Identification Dataset](https://www.kaggle.com/c/dog-breed-identification).

---

## 📁 Dataset

- **Option 1 (Kaggle)**: [Dog Breed Identification – Kaggle](https://www.kaggle.com/c/dog-breed-identification)
- **Option 2 (Stanford)**: [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/)
- Includes over **20,000 images** of **120 dog breeds**

---

## 🛠 Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- scikit-learn
- Google Colab / Jupyter Notebook

---

## 🚀 Workflow

### 1. Data Preparation
- Download and extract the dataset.
- Load images and labels into a Pandas DataFrame.
- Visualize sample images using `matplotlib`.

### 2. Preprocessing
- Resize images to a fixed shape (e.g., 224x224).
- Normalize pixel values.
- One-hot encode breed labels.
- Split into training, validation, and test sets.

### 3. Data Augmentation
Use Keras ImageDataGenerator:
python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)


### 4. Model Building

Build a Convolutional Neural Network using Keras:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(120, activation='softmax')  # 120 dog breeds
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


### 5. Model Training

Train the CNN model using the augmented dataset.

```python
train_generator = datagen.flow_from_dataframe(
    dataframe=train_df,
    directory='train/',
    x_col='id',
    y_col='breed',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_dataframe(
    dataframe=train_df,
    directory='train/',
    x_col='id',
    y_col='breed',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)


