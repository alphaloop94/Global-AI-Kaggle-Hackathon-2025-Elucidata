# kaggle Score: 0.46112 edited
import h5py
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model, Input, callbacks
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB2, ResNet50
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random
import os


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# 1. Data loading & patch extraction
def load_data(path):
    with h5py.File(path, 'r') as f:
        train_imgs = {k: f['images/Train'][k][()] for k in f['images/Train']}
        train_spots = {k: pd.DataFrame(f['spots/Train'][k][()]) for k in f['spots/Train']}
        test_img = f['images/Test']['S_7'][()]
        test_spots = pd.DataFrame(f['spots/Test']['S_7'][()])
    return train_imgs, train_spots, test_img, test_spots


def extract_patch(img, x, y, scales=(128, 224), out=224):
    h, w = img.shape[:2]
    patches = []
    for s in scales:
        half = s // 2
        x1, x2 = int(x - half), int(x + half)
        y1, y2 = int(y - half), int(y + half)
        pad_x1, pad_x2 = max(0, -x1), max(0, x2 - w)
        pad_y1, pad_y2 = max(0, -y1), max(0, y2 - h)
        patch = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        patch = cv2.copyMakeBorder(patch, pad_y1, pad_y2, pad_x1, pad_x2, cv2.BORDER_REFLECT)
        patch = cv2.resize(patch, (out, out))
        if patch.ndim == 2:
            patch = cv2.cvtColor(patch, cv2.COLOR_GRAY2RGB)
        patches.append(patch.astype('float32') / 255.)
    return np.mean(patches, axis=0)


def prepare_dataset(imgs, spots):
    Xp, Xc, Y = [], [], []
    for slide, img in imgs.items():
        df = spots[slide].dropna()
        for _, r in df.iterrows():
            x, y = r['x'], r['y']
            Xp.append(extract_patch(img, x, y))
            Xc.append([x / img.shape[1], y / img.shape[0]])
            Y.append(r.values[2:].astype('float32'))
    return np.stack(Xp), np.stack(Xc), np.stack(Y)


# 2. Build model with diverse backbones
def build_model(seed=0, variant='B0', lr=1e-5):
    tf.keras.utils.set_random_seed(seed)

    img_input = Input((224, 224, 3), name='img')

    # Data Augmentation
    augment = tf.keras.Sequential([
        layers.RandomFlip("horizontal", seed=seed),
        layers.RandomRotation(0.1, seed=seed),
        layers.RandomZoom(0.1, seed=seed),
        layers.RandomContrast(0.2, seed=seed),
        layers.RandomBrightness(factor=0.1, seed=seed),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1, seed=seed)
    ])
    x = augment(img_input)

    # Backbone Selection
    if variant == 'B1':
        base_model = EfficientNetB1(include_top=False, weights='imagenet', input_tensor=x, pooling='avg')
    elif variant == 'B2':
        base_model = EfficientNetB2(include_top=False, weights='imagenet', input_tensor=x, pooling='avg')
    elif variant == 'B0':
        base_model = EfficientNetB0(include_top=False, weights='imagenet', input_tensor=x, pooling='avg')
    elif variant == 'R50':
        base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=x, pooling='avg')
    else:
        raise ValueError(f"Unknown variant '{variant}'")

    for layer in base_model.layers[:100]:
        layer.trainable = False

    # Coordinate Input
    coord_input = Input((2,), name='coord')
    coord_x = layers.Dense(64, activation='relu')(coord_input)
    coord_x = layers.BatchNormalization()(coord_x)
    coord_x = layers.Dense(64, activation='relu')(coord_x)

    # Combined features
    combined = layers.Concatenate()([base_model.output, coord_x])

    # Additional Dense Layers
    x = layers.Dense(256, activation='relu')(combined)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    output = layers.Dense(35, activation=None)(x)

    model = Model(inputs=[img_input, coord_input], outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss='mae',
        metrics=['mae']
    )
    return model


# 3. Train single model
def train_model(Xp, Xc, Y, seed, variant='B0'):
    print(f"▶ Training seed={seed} with variant={variant}")
    tf.keras.utils.set_random_seed(seed)
    Xp_tr, Xp_val, Xc_tr, Xc_val, y_tr, y_val = train_test_split(
        Xp, Xc, Y, test_size=0.15, random_state=seed)

    model = build_model(seed=seed, variant=variant)

    os.makedirs('models', exist_ok=True)  # call it before the callbacks
    cb = [
        callbacks.EarlyStopping('val_mae', patience=10, restore_best_weights=True),
        callbacks.ModelCheckpoint(f'models/best_{variant}_{seed}.keras', save_best_only=True, monitor='val_mae'),
        callbacks.ReduceLROnPlateau('val_mae', factor=0.5, patience=5, min_lr=1e-6)
    ]

    model.fit([Xp_tr, Xc_tr], y_tr,
              validation_data=([Xp_val, Xc_val], y_val),
              epochs=50, batch_size=32,
              callbacks=cb, verbose=2)
    return model


def train_ensemble(Xp, Xc, Y):
    configs = [
        (0, 'B0'),
        (17, 'B1'),
        (42, 'B2'),
        (77, 'R50')  # ResNet50 with seed 77
    ]
    return [train_model(Xp, Xc, Y, seed=s, variant=v) for s, v in configs]


# 5. Predict & save
def predict_and_save(models, test_img, test_spots, scaler, out='submission.csv'):
    patches, coords = [], []
    test_spots = test_spots.dropna()
    for _, r in test_spots.iterrows():
        patches.append(extract_patch(test_img, r['x'], r['y']))
        coords.append([r['x'] / test_img.shape[1], r['y'] / test_img.shape[0]])
    Xp_test, Xc_test = np.stack(patches), np.stack(coords)
    preds = np.mean([m.predict([Xp_test, Xc_test], batch_size=64, verbose=0)
                     for m in models], axis=0)
    preds = scaler.inverse_transform(preds)
    df = pd.DataFrame(preds, columns=[f'C{i + 1}' for i in range(35)])
    df.insert(0, 'ID', np.arange(len(df)))
    df.to_csv(out, index=False)
    print(f"✅ Saved {out} ({df.shape[0]} rows)")


# 6. Run all
if __name__ == '__main__':
    h5 = '/kaggle/input/el-hackathon-2025/elucidata_ai_challenge_data.h5'
    print("Loading data…")
    imgs, spots, timg, tspots = load_data(h5)
    print("Preparing patches & coords…")
    Xp, Xc, Y = prepare_dataset(imgs, spots)
    print("Scaling targets…")
    scaler = StandardScaler()
    Ys = scaler.fit_transform(Y)
    print("Training ensemble…")
    ensemble = train_ensemble(Xp, Xc, Ys)
    print("Generating & saving submission…")
    predict_and_save(ensemble, timg, tspots, scaler)
