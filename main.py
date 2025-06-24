import os
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import keras_tuner as kt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import ResNet50, preprocess_input
from keras.callbacks import Callback
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from PIL import Image
import matplotlib.pyplot as plt

base_dir = "D:/Academic-Project/Portofolio Machine Learning/DataSlayer"
train_dir = os.path.join(base_dir, "train")
augment_dir = os.path.join(base_dir, "augmented_fall")

def load_dataset(base_path):
    data = []
    
    for subject in os.listdir(base_path):
        if "subject-" in subject: 
            subject_path = os.path.join(base_path, subject)
            
            for label in ["fall", "non_fall"]:
                label_path = os.path.join(subject_path, label)
                
                if os.path.exists(label_path):  
                    for action in os.listdir(label_path):
                        action_path = os.path.join(label_path, action)
                        
                        for frame in os.listdir(action_path):
                            if frame.endswith(('.jpg', '.png')):  
                                frame_path = os.path.join(action_path, frame)
                                
                                data.append({
                                    "subject": subject,
                                    "label": label,
                                    "action": action,
                                    "path": frame_path
                                })
    
    return pd.DataFrame(data)

df = load_dataset(train_dir)

def plot_label_distribution_per_subject(df):
    sns.countplot(data=df, x='subject', hue='label')
    plt.title("Distribusi Data per Subjek")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_sample_frames(df):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    for i, label in enumerate(["fall", "non_fall"]):
        sample = df[df['label'] == label].sample(1).iloc[0]
        img = Image.open(sample["path"])
        axs[i].imshow(img)
        axs[i].set_title(f"{label.upper()} - {sample['action']}")
        axs[i].axis("off")

    plt.suptitle("Contoh Frame Fall dan Non-Fall")
    plt.tight_layout()
    plt.show()

def check_image_sizes(df, sample_size=100):
    sizes = []
    errors = []
    sampled_paths = df['path'].sample(min(sample_size, len(df)))

    for path in sampled_paths:
        try:
            with Image.open(path) as img:
                sizes.append(img.size)
        except Exception as e:
            errors.append((path, str(e)))

    print("Ukuran unik gambar yang ditemukan:", set(sizes))
    if errors:
        print("\nGambar yang gagal dibuka:")
        for path, err in errors:
            print(f"{path} => {err}")


def plot_action_distribution(df):
    # Hitung distribusi aksi
    action_counts = df.groupby(['action', 'label']).size().unstack().fillna(0)
    
    # 1. Tampilkan nilai numerik
    print("="*50)
    print("DISTRIBUSI AKSI DETAIL:")
    print("-"*50)
    print(action_counts)
    print("="*50)
    
    # 2. Visualisasi
    plt.figure(figsize=(12, 8))
    
    # Plot stacked bar
    action_counts.plot(kind='barh', stacked=True, figsize=(12, 8))
    
    plt.title('Distribusi Aksi per Label (Fall vs Non-Fall)', pad=20)
    plt.xlabel('Jumlah Frame')
    plt.ylabel('Aksi')
    plt.legend(title='Label', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Tambah annotasi nilai
    for i, (idx, row) in enumerate(action_counts.iterrows()):
        total = row.sum()
        plt.text(total + 5, i, f"{int(total)}", va='center')
        
        for j, val in enumerate(row):
            if val > 0:
                plt.text(val/2 + action_counts.iloc[:, :j].sum(axis=1)[i], 
                         i, 
                         f"{int(val)}", 
                         va='center', 
                         ha='center',
                         color='white' if val > total/3 else 'black')
    
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

print(f"Total data loaded: {len(df)}")

sns.countplot(data=df, x='subject', hue='label')
plt.title("Distribusi Data per Subjek")
plt.show()

plot_sample_frames(df)

plot_action_distribution(df)

def augment_fall_images(df_minority, target_count, augment_dir="augmented_fall"):
    """
    Melakukan augmentasi gambar kelas minoritas (fall) hingga mencapai target_count.
    
    Parameters:
        df_minority (DataFrame): DataFrame berisi data kelas minoritas
        target_count (int): Jumlah total sampel yang diinginkan
        augment_dir (str): Direktori penyimpanan gambar augmentasi
    
    Returns:
        DataFrame: Berisi metadata gambar yang diaugmentasi
    """
    os.makedirs(augment_dir, exist_ok=True)
    augmented_data = []
    
    datagen = ImageDataGenerator(
        rotation_range=15,        # Rotasi +/- 15 derajat
        width_shift_range=0.1,   # Pergeseran horizontal 10%
        height_shift_range=0.1,  # Pergeseran vertikal 10%
        shear_range=0.1,         # Shear 10%
        zoom_range=0.1,          # Zoom +/- 10%
        horizontal_flip=True,    # Flip horizontal
        brightness_range=[0.9, 1.1],  # Variasi brightness
        fill_mode='nearest'      # Metode pengisian area kosong
    )
    
    # Hitung kebutuhan augmentasi
    samples_needed = target_count - len(df_minority)
    aug_per_image = max(1, samples_needed // len(df_minority) + 1)
    
    print(f"\nMemulai augmentasi {aug_per_image} variasi per gambar...")
    print(f"Direktori penyimpanan: {os.path.abspath(augment_dir)}")
    
    for idx, row in df_minority.iterrows():
        try:
            # Buka gambar dan konversi ke array
            img = Image.open(row['path'])
            img_array = np.array(img)
            
            # Reshape untuk augmentasi (1, height, width, channels)
            if img_array.ndim == 2:  # Jika grayscale
                img_array = img_array.reshape((1,) + img_array.shape + (1,))
            else:
                img_array = img_array.reshape((1,) + img_array.shape)
            
            # Generate augmented images
            aug_count = 0
            for batch in datagen.flow(img_array, batch_size=1,
                                    save_to_dir=augment_dir,
                                    save_prefix=f'aug_{idx}',
                                    save_format='jpg'):
                # Simpan metadata
                new_filename = f"aug_{idx}_{aug_count}.jpg"
                new_path = os.path.join(augment_dir, new_filename)
                
                augmented_data.append({
                    "subject": row['subject'],
                    "label": row['label'],
                    "action": row['action'],
                    "path": new_path
                })
                
                aug_count += 1
                if aug_count >= aug_per_image:
                    break
                    
        except Exception as e:
            print(f"Gagal memproses {row['path']}: {str(e)}")
    
    return pd.DataFrame(augmented_data).sample(min(samples_needed, len(augmented_data)))

majority_count = len(df[df['label'] == 'non_fall'])
minority_count = len(df[df['label'] == 'fall'])

if minority_count < majority_count:
        df_minority = df[df['label'] == 'fall']
        df_augmented = augment_fall_images(df_minority, majority_count, augment_dir)
        df_balanced = pd.concat([df, df_augmented])
        
        print("\nDistribusi kelas setelah augmentasi:")
        print(df_balanced['label'].value_counts())
else:
        print("\nDataset sudah seimbang, tidak perlu augmentasi")
        df_balanced = df.copy()

train_df, val_df = train_test_split(
    df_balanced, 
    test_size=0.2, 
    stratify=df_balanced['label'], 
    random_state=42
)

train_df['label'] = train_df['label'].apply(lambda x: 1 if x == 'fall' else 0)
val_df['label'] = val_df['label'].apply(lambda x: 1 if x == 'fall' else 0)

def calculate_class_weights(df):
    classes = np.unique(df['label'])
    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=df['label']
    )
    return dict(zip(classes, weights))

class_weights = calculate_class_weights(train_df)  

print("\nClass weights for training:")
print(class_weights)
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30, 
    width_shift_range=0.25,
    height_shift_range=0.25,
    shear_range=0.25,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,  
    brightness_range=[0.8, 1.2],  
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

def df_to_generator(df, generator, batch_size=32):
    return generator.flow_from_dataframe(
        dataframe=df,
        x_col="path",
        y_col="label",
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='raw',
        shuffle=True
    )

train_generator = df_to_generator(train_df, train_datagen)
val_generator = df_to_generator(val_df, val_datagen)

def build_model(hp):
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Tunable hyperparameters
    hp_trainable_layers = hp.Int('trainable_layers', min_value=0, max_value=100, step=10)
    hp_units1 = hp.Int('units1', min_value=256, max_value=1024, step=128)
    hp_units2 = hp.Int('units2', min_value=128, max_value=512, step=64)
    hp_dropout = hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-3, 5e-4, 1e-4, 5e-5, 1e-5])
    
    # Freeze layers based on tuning
    for layer in base_model.layers[:hp_trainable_layers]:
        layer.trainable = False
    
    # Model architecture
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    x = Dense(units=hp_units1, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(hp_dropout)(x)
    
    x = Dense(units=hp_units2, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(hp_dropout/2)(x)
    
    predictions = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(
        optimizer=Adam(learning_rate=hp_learning_rate),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    
    return model

tuner = kt.Hyperband(
    hypermodel=build_model,
    objective=kt.Objective("val_auc", direction="max"),
    max_epochs=5,       
    factor=3,            
    directory='hyperparameter_tuning',
    project_name='fall_detect',
    overwrite=True,
    executions_per_trial=1
)

early_stop = EarlyStopping(
    monitor='val_auc',
    patience=5,
    mode='max',
    restore_best_weights=True
)

print("Starting hyperparameter search...")
tuner.search(
    train_generator,
    epochs=30,
    validation_data=val_generator,
    callbacks=[early_stop],
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator),
    class_weight=class_weights
)

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"""
Best hyperparameters:
- Trainable layers: {best_hps.get('trainable_layers')}
- Dense layer 1 units: {best_hps.get('units1')}
- Dense layer 2 units: {best_hps.get('units2')}
- Dropout rate: {best_hps.get('dropout')}
- Learning rate: {best_hps.get('learning_rate')}
""")

# 4. Final Model Training
print("\nTraining final model with best hyperparameters...")
model = tuner.hypermodel.build(best_hps)

model_checkpoint = ModelCheckpoint(
    'best_fall_detection_model.h5',
    monitor='val_auc',
    mode='max',
    save_best_only=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=15,
    validation_data=val_generator,
    validation_steps=len(val_generator),
    callbacks=[model_checkpoint, early_stop, reduce_lr],
    class_weight=class_weights
)

def plot_training_history(history):
    plt.figure(figsize=(18, 6))
    
    metrics = {
        'accuracy': 'Accuracy',
        'loss': 'Loss',
        'auc': 'AUC',
        'precision': 'Precision',
        'recall': 'Recall'
    }
    
    for i, (metric, title) in enumerate(metrics.items(), 1):
        plt.subplot(2, 3, i)
        plt.plot(history.history[metric], label=f'Train {title}')
        if f'val_{metric}' in history.history:
            plt.plot(history.history[f'val_{metric}'], label=f'Validation {title}')
        plt.title(f'Model {title}')
        plt.ylabel(title)
        plt.xlabel('Epoch')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.show()

plot_training_history(history)

# Load and evaluate best model
print("\nEvaluating final model...")
model.load_weights('best_fall_detection_model.h5')

results = model.evaluate(val_generator)
metrics = {
    'Loss': results[0],
    'Accuracy': results[1]*100,
    'Precision': results[2]*100,
    'Recall': results[3]*100,
    'AUC': results[4]*100
}

print("\nValidation Metrics:")
for name, value in metrics.items():
    print(f"- {name}: {value:.2f}{'%' if name != 'Loss' else ''}")

# Save final model
model.save('final_fall_detection_model.h5')
print("\nModel saved as 'final_fall_detection_model.h5'")

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained model
model_1 = load_model('model.h5')

# Define the test folder path
test_folder = 'test'  # Change this to your test folder path
output_csv = 'predictions.csv'

# Get all image files in the test folder
image_files = [f for f in os.listdir(test_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Prepare the predictions
predictions = []

# Process each image
for img_file in image_files:
    # Load and preprocess the image
    img_path = os.path.join(test_folder, img_file)
    img = image.load_img(img_path, target_size=(model_1.input_shape[1], model_1.input_shape[2]))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Make prediction
    prediction = model_1.predict(img_array, verbose=0)
    predicted_class = np.argmax(prediction, axis=1)[0]  # Get the class with highest probability
    
    # Add to results
    predictions.append((img_file, predicted_class))

# Create DataFrame and save to CSV
df = pd.DataFrame(predictions, columns=['id', 'label'])
df.to_csv(output_csv, index=False)

print(f"Predictions saved to {output_csv}")