!pip install -U scikit-fuzzy

#import library
import os
import time
import glob
import shutil
from google.colab import drive
import zipfile

import cv2
import PIL
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('darkgrid')

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import tensorflow.image as tfi
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Conv2D, MaxPool2D, UpSampling2D,Multiply, Activation, SeparableConv2D, Conv2DTranspose, MaxPooling2D, concatenate, Layer, Input, Add, Dropout, BatchNormalization, MultiHeadAttention, LayerNormalization, Dense, Dropout
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import Sequential, initializers
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from sklearn.model_selection import KFold # Import KFold from sklearn.model_selection

from sklearn.metrics import jaccard_score
import skfuzzy as fuzz
import warnings
import logging
warnings.filterwarnings("ignore")

print ('modules loaded')

#Fuzzy Sigmoid Layer
class FuzzyLayer(Layer):
    def __init__(self, **kwargs):
        super(FuzzyLayer, self).__init__(**kwargs)

    def call(self, inputs):
        b = 0.5  # midpoint
        c = 1.5  # fuzzy scale
        return 1. / (1. + tf.exp(-c * (inputs - b)))

#Create Data
def create_data(data_dir):
    image_paths = []
    mask_paths = []

    folds = sorted(os.listdir(data_dir))
    for fold in folds:
        foldpath = os.path.join(data_dir, fold)
        if fold.lower() in ['image', 'images']:
            images = sorted(os.listdir(foldpath))
            for image in images:
                if image.lower().endswith(('.jpg', '.jpeg', '.png')):
                    fpath = os.path.join(foldpath, image)
                    image_paths.append(fpath)

        elif fold.lower() in ['mask', 'masks']:
            masks = sorted(os.listdir(foldpath))
            for mask in masks:
                if mask.lower().endswith(('.jpg', '.jpeg', '.png')):
                    fpath = os.path.join(foldpath, mask)
                    mask_paths.append(fpath)
        else:
            continue

    return image_paths, mask_paths


#Load Images
def load_image(image, grayscale=False):
    if grayscale:
        img = load_img(image, color_mode="grayscale")
    else:
        img = load_img(image)

    return np.round(tfi.resize(img_to_array(img) / 255., (512, 512)), 4)

def load_images(image_paths, mask=False, grayscale=False, trim=None):
    if trim is not None:
        image_paths = image_paths[:trim]

    if mask:
        images = np.zeros(shape=(len(image_paths), 512, 512, 1))
    else:
        images = np.zeros(shape=(len(image_paths), 512, 512, 3))

    for i, image in enumerate(image_paths):
        img = load_image(image, grayscale=grayscale)
        if mask:
            images[i] = img[:, :, :1]
        else:
            images[i] = img

    return images

#Show Images
def show_image(image, title=None, cmap=None, alpha=1):
    plt.imshow(image, cmap=cmap, alpha=alpha)
    if title is not None:
        plt.title(title)
    plt.axis('off')

def show_mask(image, mask, cmap=None, alpha=0.4):
    plt.imshow(image)
    plt.imshow(tf.squeeze(mask), cmap=cmap, alpha=alpha)
    plt.axis('off')

def show_images(imgs, msks):
    plt.figure(figsize=(13,8))

    for i in range(15):
        plt.subplot(3,5,i+1)
        id = np.random.randint(len(imgs))
        show_mask(imgs[id], msks[id], cmap='binary')

    plt.tight_layout()
    plt.show()
     
#Smooth Transformer Block
class SmoothingTransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(SmoothingTransformerBlock, self).__init__(**kwargs)
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation='relu'),
            Dense(embed_dim)
        ])
        self.norm1 = LayerNormalization()
        self.norm2 = LayerNormalization()
        self.dropout = Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.mha(inputs, inputs)
        attn_output = self.dropout(attn_output, training=training)
        out1 = self.norm1(inputs + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout(ffn_output, training=training)
        return self.norm2(out1 + ffn_output)


#Fuzzy Encoder Block
class FuzzyEncoderBlock(Layer):
    def __init__(self, filters, kernel_size=3, rate=0.05, pooling=True, **kwargs):
        super(FuzzyEncoderBlock, self).__init__(**kwargs)
        self.conv1 = Conv2D(filters, kernel_size=kernel_size, padding='same', kernel_regularizer=l2(0.0001))
        self.bn1 = BatchNormalization()
        self.conv2 = Conv2D(filters, kernel_size=kernel_size, padding='same', kernel_regularizer=l2(0.0001))
        self.bn2 = BatchNormalization()
        self.pool = MaxPooling2D(pool_size=(2, 2)) if pooling else None
        self.dropout = Dropout(rate)

        self.fuzzy = FuzzyLayer()

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = Activation('relu')(x)

        x = self.conv2(x)
        x = Activation('relu')(x)

        if self.pool:
            x = self.pool(x)
            x = self.fuzzy(x)

        return x, inputs

#Attention Gate
class AttentionGate(Layer):
    def __init__(self, filters, kernel_size=1, **kwargs):
        super(AttentionGate, self).__init__(**kwargs)
        self.W_g = Conv2D(filters, kernel_size=kernel_size, strides=1, padding='same')
        self.W_x = Conv2D(filters, kernel_size=kernel_size, strides=1, padding='same')
        self.psi = Conv2D(1, kernel_size=kernel_size, strides=1, padding='same', activation='sigmoid')
        self.relu = Activation('relu')
        self.bn = BatchNormalization()

    def call(self, g, x):
        # g: Decoder feature map
        # x: Skip connection from Encoder
        g1 = self.W_g(g)  # Linear transformation on g
        x1 = self.W_x(x)  # Linear transformation on x
        psi = self.relu(g1 + x1)  # Add and apply activation
        psi = self.bn(psi)
        psi = self.psi(psi)  # Sigmoid to get attention weights
        return Multiply()([x, psi])  # Apply attention to skip connection

#Loss Function
def tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1 - y_true) * y_pred)
    fn = tf.reduce_sum(y_true * (1 - y_pred))

    tversky_index = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return 1 - tversky_index

#Decoder Block
class DecoderBlock(Layer):
    def __init__(self, filters, rate=0.05, kernel_size=3, kernel_reg=0.0001, use_attention_gate=True, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.up = Conv2DTranspose(filters, kernel_size=kernel_size, strides=(2, 2), padding='same')

        # Change here: Initialize AttentionGate with the correct number of filters for the skip connection
        self.attention_gate = AttentionGate(filters) if use_attention_gate else None  # Updated filters to 'filters'

        self.conv1 = Conv2D(filters, kernel_size=kernel_size, padding='same', kernel_regularizer=l2(kernel_reg))
        self.bn1 = BatchNormalization()
        self.dropout1 = Dropout(rate)

        self.conv2 = Conv2D(filters, kernel_size=kernel_size, padding='same', kernel_regularizer=l2(kernel_reg))
        self.bn2 = BatchNormalization()
        self.dropout2 = Dropout(rate)

        self.spatial_attention = Conv2D(1, kernel_size=1, activation='sigmoid')

    def call(self, inputs, training=False):
        x, skip = inputs

        # Apply attention gate if enabled
        if self.attention_gate:
            # Adjust the number of filters in 'x' to match 'skip' before applying attention gate
            # x = Conv2D(skip.shape[-1], kernel_size=1, padding='same')(x)  # Added to adjust filters of 'x' #Removed to upsample x instead
            x = self.up(x) # Added to upsample x to match the spatial dimension of the skip connection
            skip = self.attention_gate(x, skip) # Updated

        # Upsample (Moved up for attention gate)
        #x = self.up(x)

        # Concatenate skip connection
        x = concatenate([x, skip], axis=3)

        # Apply spatial attention
        attention_weights = self.spatial_attention(x)
        x = Multiply()([x, attention_weights])

        # First convolution block
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = Activation('relu')(x)
        x = self.dropout1(x, training=training)

        # Second convolution block
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = Activation('relu')(x)
        x = self.dropout2(x, training=training)

        return x

#Metric Evaluation
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    intersection = tf.keras.backend.sum(y_true * y_pred)
    union = tf.keras.backend.sum(y_true) + tf.keras.backend.sum(y_pred)

    dice = (2. * intersection + smooth) / (union + smooth)
    return dice

def jaccard_index(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    intersection = tf.keras.backend.sum(y_true * y_pred)
    union = tf.keras.backend.sum(y_true) + tf.keras.backend.sum(y_pred) - intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou

def precision(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    true_positives = tf.keras.backend.sum(y_true * y_pred)
    predicted_positives = tf.keras.backend.sum(y_pred)

    precision = true_positives / (predicted_positives + smooth)
    return precision

def recall(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    true_positives = tf.keras.backend.sum(y_true * y_pred)
    actual_positives = tf.keras.backend.sum(y_true)

    recall = true_positives / (actual_positives + smooth)
    return recall

def combined_loss(y_true, y_pred):
    smooth = 1e-6
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    bce = tf.keras.losses.BinaryCrossentropy()
    bce_loss = bce(y_true, y_pred)

    intersection = tf.keras.backend.sum(y_true * y_pred)
    union = tf.keras.backend.sum(y_true) + tf.keras.backend.sum(y_pred)
    dice_loss = 1 - (2. * intersection + smooth) / (union + smooth)

    combined_loss = 0.5 * dice_loss + 0.5 * bce_loss

    return combined_loss

#Custom Callback
class MetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, x_val, y_val):
        super(MetricsCallback, self).__init__()
        self.x_val = x_val
        self.y_val = y_val

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        y_pred = self.model.predict(self.x_val)
        dice = dice_coefficient(self.y_val, y_pred)
        iou = jaccard_index(self.y_val, y_pred)
        prec = precision(self.y_val, y_pred)
        rec = recall(self.y_val, y_pred)

        print(f'Epoch {epoch + 1} - Dice Coefficient: {dice}, IoU: {iou}, Precision: {prec}, Recall: {rec}')
        logs['val_dice_coefficient'] = float(dice)
        logs['val_jaccard_index'] = float(iou)
        logs['val_precision'] = float(prec)
        logs['val_recall'] = float(rec)

#Dataset
#Mount Google Drive and Extract Dataset
import os
import zipfile
from sklearn.model_selection import train_test_split
import shutil
from google.colab import drive

drive.mount('/content/drive')
train_zip_path = '/path_to_dataset_train.zip'
test_zip_path = '/path_to_dataset_test.zip'

with zipfile.ZipFile(train_zip_path, 'r') as zip_ref:
    zip_ref.extractall('dataset/train')
with zipfile.ZipFile(test_zip_path, 'r') as zip_ref:
    zip_ref.extractall('dataset/test')

#Load and Show Train Dataset
train_dir = 'dataset/train'
image_paths, mask_paths = create_data(train_dir)

imgs = load_images(image_paths, mask=False, grayscale=False)
msks = load_images(mask_paths, mask=True)

show_images(imgs, msks)

#Create Model
def create_model(optimizer):
    inputs = Input(shape=(512, 512, 3))
    p1, c1 = FuzzyEncoderBlock(16, pooling=True)(inputs)
    p2, c2 = FuzzyEncoderBlock(32, pooling=True)(p1)
    p3, c3 = FuzzyEncoderBlock(64, pooling=True)(p2)
    p4, c4 = FuzzyEncoderBlock(128, pooling=True)(p3)

    transformer_output = SmoothingTransformerBlock(embed_dim=128, num_heads=4, ff_dim=256)(p4)

    d1 = DecoderBlock(64)([transformer_output, c4])
    d2 = DecoderBlock(32)([d1, c3])
    d3 = DecoderBlock(16)([d2, c2])
    d4 = DecoderBlock(8)([d3, c1])
    output_layer = Conv2D(1, kernel_size=(1, 1), activation='sigmoid')(d4)

    model = Model(inputs=[inputs], outputs=[output_layer])
    model.compile(optimizer=optimizer, loss=combined_loss, metrics=[dice_coefficient, jaccard_index, precision, recall])

    return model

optimizer = AdamW(learning_rate=0.0001, weight_decay=1e-4)
model = create_model(optimizer)
model.summary()

#FLOPS Calculate
def get_flops(model):
    input_signature = [tf.TensorSpec([1] + list(model.input_shape[1:]), tf.float32)]
    full_model = tf.function(model).get_concrete_function(input_signature)
    frozen_func = convert_variables_to_constants_v2(full_model)
    run_meta = tf.compat.v1.RunMetadata()

    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.compat.v1.profiler.profile(graph=frozen_func.graph, run_meta=run_meta, options=opts)

    return flops.total_float_ops

tf.get_logger().setLevel(logging.ERROR)
flops = get_flops(model)
print(f"FLOPs: {flops / 10 ** 9:.03} G")

#Plot Training
def plot_training(hist):
    tr_loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    dice = hist.history['dice_coefficient']
    iou = hist.history['jaccard_index']
    prec = hist.history['precision']
    rec = hist.history['recall']
    index_loss = np.argmin(val_loss)
    val_lowest = val_loss[index_loss]
    index_dice = np.argmax(dice)
    dice_highest = dice[index_dice]
    index_iou = np.argmax(iou)
    iou_highest = iou[index_iou]
    index_prec = np.argmax(prec)
    prec_highest = prec[index_prec]
    index_rec = np.argmax(rec)
    rec_highest = rec[index_rec]
    Epochs = [i+1 for i in range(len(dice))]
    loss_label = f'best epoch= {str(index_loss + 1)}'
    dice_label = f'best epoch= {str(index_dice + 1)}'
    iou_label = f'best epoch= {str(index_iou + 1)}'
    prec_label = f'best epoch= {str(index_prec + 1)}'
    rec_label = f'best epoch= {str(index_rec + 1)}'

    plt.figure(figsize= (20, 18))
    plt.style.use('fivethirtyeight')

    plt.subplot(3, 2, 1)
    plt.plot(Epochs, tr_loss, 'r', label= 'Training loss')
    plt.plot(Epochs, val_loss, 'g', label= 'Validation loss')
    plt.scatter(index_loss + 1, val_lowest, s= 150, c= 'blue', label= loss_label)
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(3, 2, 2)
    plt.plot(Epochs, dice, 'r', label= 'Dice Coefficient')
    plt.scatter(index_dice + 1 , dice_highest, s= 150, c= 'blue', label= dice_label)
    plt.title('Dice Coefficient')
    plt.xlabel('Epochs')
    plt.ylabel('Dice Coefficient')
    plt.legend()

    plt.subplot(3, 2, 3)
    plt.plot(Epochs, iou, 'r', label= 'IoU')
    plt.scatter(index_iou + 1 , iou_highest, s= 150, c= 'blue', label= iou_label)
    plt.title('Intersection over Union (IoU)')
    plt.xlabel('Epochs')
    plt.ylabel('IoU')
    plt.legend()

    plt.subplot(3, 2, 4)
    plt.plot(Epochs, prec, 'r', label= 'Precision')
    plt.scatter(index_prec + 1 , prec_highest, s= 150, c= 'blue', label= prec_label)
    plt.title('Precision')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.legend()

    plt.subplot(3, 2, 5)
    plt.plot(Epochs, rec, 'r', label= 'Recall')
    plt.scatter(index_rec + 1 , rec_highest, s= 150, c= 'blue', label= rec_label)
    plt.title('Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.legend()

    plt.tight_layout
    plt.show()

#Train Configuration
optimizer = AdamW(learning_rate=0.0001, weight_decay=1e-4)
model = create_model(optimizer)

batch_size = 8
epochs = 150

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
metrics_callback = MetricsCallback(imgs, msks)

#Training with Cross Validation
def train_with_cross_validation(imgs, msks, n_splits=5, epochs=150, batch_size=8, save_dir="/pathtosavemodel"):
    # Buat folder untuk menyimpan model
    os.makedirs(save_dir, exist_ok=True)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = {'dice': [], 'iou': [], 'precision': [], 'recall': []}
    best_dice = 0
    best_model_path = ""

    for fold, (train_index, val_index) in enumerate(kf.split(imgs)):
        print(f"Training Fold {fold + 1}/{n_splits}")

        X_train, X_val = imgs[train_index], imgs[val_index]
        y_train, y_val = msks[train_index], msks[val_index]

        optimizer = AdamW(learning_rate=0.0001, weight_decay=1e-4)
        model = create_model(optimizer)

        early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
        metrics_callback = MetricsCallback(X_val, y_val)

        start_time = time.time()
        history = model.fit(
            imgs,  # Input images
            msks,  # Ground truth masks
            validation_split=0.2,  # 20% data digunakan untuk validasi
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,  # Menampilkan progress training
            callbacks=[early_stopping, reduce_lr, metrics_callback]
        )
        end_time = time.time()
        training_time = end_time - start_time
        print(f"Training time for fold {fold + 1}: {training_time:.2f} seconds")

        # Predict on validation set
        y_pred = model.predict(X_val)
        y_pred_binary = (y_pred >= 0.5).astype(int)

        # Calculate metrics
        dice = dice_coefficient(y_val, y_pred_binary)
        iou = jaccard_index(y_val, y_pred_binary)
        precision_val = precision(y_val, y_pred_binary)
        recall_val = recall(y_val, y_pred_binary)

        # Append metrics to results
        fold_metrics['dice'].append(dice)
        fold_metrics['iou'].append(iou)
        fold_metrics['precision'].append(precision_val)
        fold_metrics['recall'].append(recall_val)

        print(f"Fold {fold + 1} - Dice: {dice:.4f}, IoU: {iou:.4f}, Precision: {precision_val:.4f}, Recall: {recall_val:.4f}")

        # Save model for this fold
        fold_model_path = os.path.join(save_dir, f"model_fold_{fold + 1}.h5")
        save_model(model, fold_model_path)
        print(f"Model for fold {fold + 1} saved at {fold_model_path}")

        # Check if this is the best model
        if dice > best_dice:
            best_dice = dice
            best_model_path = os.path.join(save_dir, "best_model.h5")
            save_model(model, best_model_path)
            print(f"Best model updated with Dice: {best_dice:.4f}")

    return fold_metrics, best_model_path

# Example usage (replace imgs, msks with your actual data)
fold_metrics, best_model_path = train_with_cross_validation(imgs, msks)

# Calculate mean and standard deviation
mean_metrics = {key: np.mean(values) for key, values in fold_metrics.items()}
std_metrics = {key: np.std(values) for key, values in fold_metrics.items()}

# Print results
print("\nCross-Validation Results:")
for metric in fold_metrics.keys():
    print(f"{metric.capitalize()} - Mean: {mean_metrics[metric]:.4f}, Std: {std_metrics[metric]:.4f}")

# Visualize metrics across folds
plt.figure(figsize=(10, 6))
for metric in fold_metrics.keys():
    plt.plot(range(1, len(fold_metrics[metric]) + 1), fold_metrics[metric], marker='o', label=metric.capitalize())

plt.title("Metrics Across Folds")
plt.xlabel("Fold")
plt.ylabel("Score")
plt.legend()
plt.grid()
plt.show()

print(f"\nBest model saved at: {best_model_path}")


# Load the best model
best_model = load_model(best_model_path, custom_objects={'dice_coefficient': dice_coefficient,
                                                        'jaccard_index': jaccard_index,
                                                        'precision': precision,
                                                        'recall': recall,
                                                        'tversky_loss': tversky_loss,
                                                        'AttentionGate': AttentionGate,
                                                        'DecoderBlock': DecoderBlock,
                                                        'FuzzyLayer': FuzzyLayer,
                                                        'FuzzyEncoderBlock': FuzzyEncoderBlock,  # Add FuzzyEncoderBlock
                                                        'SmoothingTransformerBlock': SmoothingTransformerBlock}) # Add SmoothingTransformerBlock

# Assuming you have your test data loaded as X_test and y_test
# Replace 'X_test' and 'y_test' with your actual test data
test_dir = '/content/dataset/test'
X_test, y_test = create_data(test_dir)
X_test = load_images(X_test, mask=False, grayscale=False)
y_test = load_images(y_test, mask=True)

# Predict on test data
y_pred = best_model.predict(X_test)
y_pred_thresholded = (y_pred > 0.7).astype(int)

# Evaluate the model
loss, dice, iou, precision_test, recall_test = best_model.evaluate(X_test, y_test, verbose=0)

print(f"Test Loss: {loss:.4f}")
print(f"Test Dice Coefficient: {dice:.4f}")
print(f"Test IoU: {iou:.4f}")
print(f"Test Precision: {precision_test:.4f}")
print(f"Test Recall: {recall_test:.4f}")

#Test with Test Dataset
def plot_predictions(X_test, y_test, y_pred):
    n_images = len(X_test)  # Number of images to display

    plt.figure(figsize=(20, 10 * n_images))
    for i in range(n_images):
        # Original Image
        plt.subplot(n_images, 4, i * 4 + 1)
        plt.imshow(X_test[i])
        plt.title(f"Original Image {i + 1}")
        plt.axis('off')

        # Original Mask
        plt.subplot(n_images, 4, i * 4 + 2)
        plt.imshow(np.squeeze(y_test[i]), cmap='gray')
        plt.title(f"Original Mask {i + 1}")
        plt.axis('off')

        # Predicted Mask
        plt.subplot(n_images, 4, i * 4 + 3)
        plt.imshow(np.squeeze(y_pred[i]), cmap='gray')
        plt.title(f"Predicted Mask {i + 1}")
        plt.axis('off')

        # Overlay
        overlay = np.zeros_like(X_test[i])
        overlay[:, :, 0] = np.squeeze(y_pred[i]) * 255 # Red channel for predicted mask
        overlay = cv2.addWeighted(X_test[i], 0.5, overlay, 0.5, 0) # Overlay on original image
        plt.subplot(n_images, 4, i * 4 + 4)
        plt.imshow(overlay)
        plt.title(f"Overlay {i+1}")
        plt.axis('off')
    plt.show()

# Call the function to display the results
plot_predictions(X_test, y_test, y_pred_thresholded)


#Check Error Prediction
def plot_low_dice_predictions(X_test, y_test, y_pred, threshold=0.7, num_images=5):
    # Calculate Dice coefficients for each prediction
    dice_scores = []
    for i in range(len(y_test)):
      dice_scores.append(dice_coefficient(y_test[i], y_pred[i]))

    # Find indices of images with Dice coefficients below the threshold
    low_dice_indices = np.argsort(dice_scores)[:num_images]

    plt.figure(figsize=(20, 10 * num_images))
    for i, index in enumerate(low_dice_indices):
        # Original Image
        plt.subplot(num_images, 4, i * 4 + 1)
        plt.imshow(X_test[index])
        plt.title(f"Original Image {index + 1} (Dice: {dice_scores[index]:.4f})")
        plt.axis('off')

        # Original Mask
        plt.subplot(num_images, 4, i * 4 + 2)
        plt.imshow(np.squeeze(y_test[index]), cmap='gray')
        plt.title(f"Original Mask {index + 1}")
        plt.axis('off')

        # Predicted Mask
        plt.subplot(num_images, 4, i * 4 + 3)
        plt.imshow(np.squeeze(y_pred[index]), cmap='gray')
        plt.title(f"Predicted Mask {index + 1}")
        plt.axis('off')

        # Overlay
        overlay = np.zeros_like(X_test[index])
        overlay[:, :, 0] = np.squeeze(y_pred[index]) * 255  # Red channel for predicted mask
        overlay = cv2.addWeighted(X_test[index], 0.5, overlay, 0.5, 0)  # Overlay on original image
        plt.subplot(num_images, 4, i * 4 + 4)
        plt.imshow(overlay)
        plt.title(f"Overlay {index + 1}")
        plt.axis('off')

    plt.show()


# Example usage:
plot_low_dice_predictions(X_test, y_test, y_pred_thresholded)


#Generate Feature Maps
model = load_model(best_model_path, custom_objects={
    'dice_coefficient': dice_coefficient,
    'jaccard_index': jaccard_index,
    'precision': precision,
    'recall': recall,
    'tversky_loss': tversky_loss,
    'AttentionGate': AttentionGate,
    'DecoderBlock': DecoderBlock,
    'FuzzyLayer': FuzzyLayer,
    'FuzzyEncoderBlock': FuzzyEncoderBlock,
    'SmoothingTransformerBlock': SmoothingTransformerBlock
})

# Visualisasi Feature Maps
# Get a list of all layer names
layer_names = [layer.name for layer in model.layers]

# Choose an image index to display feature maps for
image_index_to_display = 46  # Change this to the desired image index

# Define the number of rows and columns for the figure
num_rows = 3  # 2 rows for feature maps and 1 for the original image
num_cols = int(np.ceil(len(layer_names) / (num_rows - 1)))

# Create the figure
plt.figure(figsize=(20, 15))  # Adjust figure size as needed

# Display the original image
plt.subplot(num_rows, num_cols, 1)
plt.imshow(imgs[image_index_to_display])
plt.title("Original Image", fontsize=10)
plt.axis('off')

# Iterate through each layer to visualize feature maps
for i, layer_name in enumerate(layer_names):
    try:
        # Create a model that outputs the activations of the chosen layer
        intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

        # Prepare the image and get the feature maps
        image = np.expand_dims(imgs[image_index_to_display], axis=0)  # Add batch dimension
        feature_maps = intermediate_layer_model.predict(image)

        # Handle cases where feature_maps is a list
        if isinstance(feature_maps, list):
            feature_maps = feature_maps[0]  # Use the first element of the list

        # Visualize the first feature map if it exists
        if feature_maps.shape[-1] > 0:
            plt.subplot(num_rows, num_cols, i + 2)  # Subplot starting from index 2
            plt.imshow(feature_maps[0, :, :, 0], cmap='gray')  # Display the first feature map
            plt.title(layer_name, fontsize=10)  # Set the title for each layer
            plt.axis('off')

    except Exception as e:
        print(f"Error processing layer {layer_name}: {e}")

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()

#Get Inference Time
inference_times = []
resized_test_images = []

for img in test_images:
  # Resize the image
  resized_img = cv2.resize(img, (224, 224))
  resized_test_images.append(resized_img)

resized_test_images = np.array(resized_test_images)

for i in range(len(resized_test_images)):
    start_time = time.time()
    # Perform inference
    model.predict(np.expand_dims(resized_test_images[i], axis=0))
    end_time = time.time()
    inference_time = end_time - start_time
    inference_times.append(inference_time)

# Calculate and print the average inference time
avg_inference_time = sum(inference_times) / len(inference_times)
print(f"Average inference time: {avg_inference_time:.4f} seconds")
