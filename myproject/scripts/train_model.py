from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt


# Fonction pour charger les données
def load_data_from_directory(dirs):
    data = []
    labels = []

    label_mapping = {
        'Glass': 0,
        'Plastic': 1,
        'Aluminium': 2,
        'Organic': 3,
        'Others': 4,
    }

    for dir_path in dirs:
        dir_name = os.path.basename(dir_path)
        label = label_mapping.get(dir_name, -1)

        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)
            if file_path.endswith('.jpg') or file_path.endswith('.png'):
                img = Image.open(file_path).convert('RGB')
                img_resized = img.resize((128, 128))
                img_array = np.array(img_resized)
                img_normalized = img_array / 255.0
                data.append(img_normalized)
                labels.append(label)

    return np.array(data), np.array(labels)


# Chemins des répertoires
dirs = [
    'D:/Projet_Annuel_M1_2024/DataSet_2024/Glass',
    'D:/Projet_Annuel_M1_2024/DataSet_2024/Plastic',
    'D:/Projet_Annuel_M1_2024/DataSet_2024/Others',
    'D:/Projet_Annuel_M1_2024/DataSet_2024/Organic',
    'D:/Projet_Annuel_M1_2024/DataSet_2024/Aluminium'
]

# Charger les données
data, labels = load_data_from_directory(dirs)
print("Données chargées avec succès.")

# Convertir les labels en one-hot encoding
labels = to_categorical(labels, num_classes=5)
print("Labels convertis en one-hot encoding.")

# Séparer les données en ensembles d'entraînement et de test
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)
print("Données séparées en ensembles d'entraînement et de test.")

# Créer un générateur de données avec augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)
datagen.fit(train_data)
print("Générateur de données créé avec augmentation.")

# Charger le modèle MobileNetV2 pré-entrainé
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
print("Modèle MobileNetV2 chargé avec succès.")

# Ajouter des couches de classification
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(5, activation='softmax')(x)

# Définir le modèle complet
model = Model(inputs=base_model.input, outputs=predictions)

# Geler les premières couches de MobileNetV2 pour ne pas les entraîner à nouveau
for layer in base_model.layers:
    layer.trainable = False

# Compiler le modèle
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("Modèle compilé avec succès.")

# Entraîner le modèle
history = model.fit(datagen.flow(train_data, train_labels, batch_size=32),
                    epochs=30,
                    validation_data=(test_data, test_labels))
print("Modèle entraîné avec succès.")

# Évaluer le modèle sur l'ensemble de test
test_loss, test_acc = model.evaluate(test_data, test_labels)
print(f'Test accuracy with MobileNetV2: {test_acc:.2f}')

# Sauvegarder le modèle
model.save('D:/Projet_Annuel_M1_2024/PA-DJANGO/myproject/model/mobilenetv2_model.h5')
print("Modèle sauvegardé avec succès.")


# Afficher les courbes de performance
def plot_history(history, title):
    plt.figure(figsize=(12, 4))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'{title} - Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{title} - Loss')
    plt.legend()

    plt.show()


# Afficher les courbes de performance
plot_history(history, 'Model with MobileNetV2 and Data Augmentation')
