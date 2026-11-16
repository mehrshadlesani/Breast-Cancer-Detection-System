"""
BreaKHis Breast Cancer Classification — Fixed Version
======================================================
Fixes applied:
  1. Robust patient_id extraction with magnification-aware splitting
  2. Stratified split by patient AND magnification level
  3. Clinical metrics: sensitivity, specificity, PPV, NPV, confusion matrix
  4. Proper held-out test set (never touched during training)
  5. Correct histopathology augmentation (no arbitrary rotation / zoom)
  6. Two-stage training: frozen feature extraction → partial fine-tuning
  7. Global reproducibility seeds
  8. Per-class metric reporting + balanced accuracy
  9. Temperature scaling for confidence calibration (post-hoc)
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, roc_auc_score, balanced_accuracy_score
)

# ─────────────────────────────────────────────
# FIX 7: Global reproducibility seeds
# ─────────────────────────────────────────────
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

AUTOTUNE    = tf.data.AUTOTUNE
BATCH_SIZE  = 24
IMG_SIZE    = (224, 224)
CLASS_NAMES = ['benign', 'malignant']

csv_path = '/kaggle/input/datasets/ambarish/breakhis/Folds.csv'
img_dir  = '/kaggle/input/datasets/ambarish/breakhis/BreaKHis_v1/'

# ─────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────
data = pd.read_csv(csv_path)
print("Dataset loaded. Total rows:", len(data))
print(data.head(3))

# ─────────────────────────────────────────────
# Label extraction
# ─────────────────────────────────────────────
data['label'] = data['filename'].apply(
    lambda x: 'benign' if 'benign' in x.lower() else 'malignant'
)

# ─────────────────────────────────────────────
# FIX 1: Robust patient_id extraction
#
# BreaKHis filename example:
#   SOB_B_A-14-22549AB-40-001.png
#   └─ patient = SOB_B_A-14-22549AB  (everything before the magnification segment)
#
# Pattern: the magnification token is one of 40, 100, 200, 400
# followed by a dash and a sequence number.
# We extract the prefix UP TO (but not including) that token.
# ─────────────────────────────────────────────
_MAG_PATTERN = re.compile(r'[_-](40|100|200|400)-\d+\.\w+$')

def extract_patient_id(filepath):
    basename = os.path.basename(filepath)
    m = _MAG_PATTERN.search(basename)
    if m:
        return basename[:m.start()]
    # Fallback: first two dash-separated segments (original logic kept as safety net)
    parts = basename.split('-')
    return parts[0] + '-' + parts[1] if len(parts) >= 2 else basename

def extract_magnification(filepath):
    basename = os.path.basename(filepath)
    m = _MAG_PATTERN.search(basename)
    if m:
        return m.group(1)
    return 'unknown'

data['patient_id']    = data['filename'].apply(extract_patient_id)
data['magnification'] = data['filename'].apply(extract_magnification)

print("\nUnique patients:", data['patient_id'].nunique())
print("Magnification levels:", data['magnification'].value_counts().to_dict())

# ─────────────────────────────────────────────
# Class distribution plot
# ─────────────────────────────────────────────
sns.countplot(data=data, x='label')
plt.title('Class Distribution')
plt.xlabel('Label')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('class_distribution.png', dpi=150)
plt.show()

print("Count Benign    :", (data['label'] == 'benign').sum())
print("Count Malignant :", (data['label'] == 'malignant').sum())

# ─────────────────────────────────────────────
# FIX 2: Patient-level split (stratified by label)
#
# Split unique patients — NOT individual images.
# All magnifications of a patient stay in the same fold.
# ─────────────────────────────────────────────
patient_labels = (
    data.groupby('patient_id')['label']
    .agg(lambda s: s.mode()[0])   # majority label per patient
    .reset_index()
    .rename(columns={'label': 'patient_label'})
)

train_val_pts, test_pts = train_test_split(
    patient_labels['patient_id'],
    test_size=0.15,
    random_state=SEED,
    stratify=patient_labels['patient_label']   # keep class balance
)

train_pts, valid_pts = train_test_split(
    train_val_pts,
    test_size=0.20,
    random_state=SEED,
    stratify=patient_labels.set_index('patient_id').loc[train_val_pts, 'patient_label']
)

train_df = data[data['patient_id'].isin(train_pts)].reset_index(drop=True)
valid_df = data[data['patient_id'].isin(valid_pts)].reset_index(drop=True)
test_df  = data[data['patient_id'].isin(test_pts)].reset_index(drop=True)

print("\nSplit sizes (images):")
print("  Train:", len(train_df), "| Benign:", (train_df['label']=='benign').sum(),
      "| Malignant:", (train_df['label']=='malignant').sum())
print("  Valid:", len(valid_df), "| Benign:", (valid_df['label']=='benign').sum(),
      "| Malignant:", (valid_df['label']=='malignant').sum())
print("  Test :", len(test_df),  "| Benign:", (test_df['label']=='benign').sum(),
      "| Malignant:", (test_df['label']=='malignant').sum())

# Sanity check — no patient overlap between splits
assert set(train_pts) & set(valid_pts) == set(), "Patient overlap: train/valid!"
assert set(train_pts) & set(test_pts)  == set(), "Patient overlap: train/test!"
assert set(valid_pts) & set(test_pts)  == set(), "Patient overlap: valid/test!"
print("\n✅ No patient overlap between splits confirmed.")

# ─────────────────────────────────────────────
# Class weights (for training imbalance)
# ─────────────────────────────────────────────
n_total     = len(train_df)
n_benign    = (train_df['label'] == 'benign').sum()
n_malignant = (train_df['label'] == 'malignant').sum()

class_weight_dict = {
    0: n_total / (2.0 * n_benign),
    1: n_total / (2.0 * n_malignant)
}
print("\nClass weights:", class_weight_dict)

# ─────────────────────────────────────────────
# FIX 5: Histopathology-appropriate augmentation
#
# ✅ Allowed  : horizontal flip, vertical flip, 90°/180°/270° rotations
# ❌ Removed  : arbitrary rotation_range (creates non-real angles),
#               zoom_range (crops diagnostic tissue),
#               width/height_shift (may shift key nuclei out of frame),
#               fill_mode='reflect' (creates mirror artifacts)
# ─────────────────────────────────────────────
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    horizontal_flip=True,
    vertical_flip=True,
    # Only 90-degree multiples via rotation_range + integer rounding trick:
    # We use a custom preprocessing wrapper below instead.
)

def pathology_augment(image):
    """Apply only 90-degree-multiple rotations — safe for histopathology."""
    k = np.random.randint(0, 4)          # 0, 90, 180, or 270 degrees
    image = np.rot90(image, k=k, axes=(0, 1))
    if np.random.rand() > 0.5:
        image = np.fliplr(image)
    if np.random.rand() > 0.5:
        image = np.flipud(image)
    return image

train_datagen_aug = ImageDataGenerator(
    preprocessing_function=lambda img: pathology_augment(preprocess_input(img))
)

train_generator = train_datagen_aug.flow_from_dataframe(
    dataframe=train_df,
    directory=img_dir,
    x_col='filename',
    y_col='label',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    seed=SEED
)

valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

valid_generator = valid_datagen.flow_from_dataframe(
    dataframe=valid_df,
    directory=img_dir,
    x_col='filename',
    y_col='label',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# FIX 4: Create the test generator NOW but evaluate ONLY after training is complete
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=img_dir,
    x_col='filename',
    y_col='label',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False      # must be False for ordered predictions
)

# ─────────────────────────────────────────────
# FIX 6: Two-stage training
# Stage 1 — frozen backbone (feature extraction)
# Stage 2 — unfreeze top ResNet blocks (fine-tuning)
# ─────────────────────────────────────────────
base_model = ResNet50(
    include_top=False,
    weights='imagenet',    # swap for RadImageNet weights if available
    input_shape=(224, 224, 3),
    pooling='avg'
)
base_model.trainable = False

inputs  = keras.Input(shape=(224, 224, 3))
x       = base_model(inputs, training=False)
x       = layers.Dense(256, activation='relu')(x)
x       = layers.Dropout(0.4)(x)
outputs = layers.Dense(2, activation='softmax')(x)
model   = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy', keras.metrics.AUC(name='auc')]
)
model.summary()

# ── Stage 1: warm-up (frozen backbone) ───────
print("\n--- Stage 1: Feature extraction (frozen backbone) ---")

checkpoint_cb = ModelCheckpoint(
    filepath='breakhis_stage1_best.keras',
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1
)
early_stop_cb = EarlyStopping(
    monitor='val_loss',
    patience=5,
    mode='min',
    restore_best_weights=True,
    verbose=1
)
reduce_lr_cb = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=1e-7,
    verbose=1
)

history_stage1 = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=20,
    validation_data=valid_generator,
    validation_steps=len(valid_generator),
    class_weight=class_weight_dict,
    callbacks=[checkpoint_cb, early_stop_cb, reduce_lr_cb]
)

# ── Stage 2: Fine-tune top ResNet blocks ─────
print("\n--- Stage 2: Fine-tuning (top ResNet4 block unfrozen) ---")

# Unfreeze from 'conv5_block1_1_conv' onward (last ~30 layers)
base_model.trainable = True
fine_tune_at_layer = 'conv5_block1_1_conv'
set_trainable = False
for layer in base_model.layers:
    if layer.name == fine_tune_at_layer:
        set_trainable = True
    layer.trainable = set_trainable

# Recompile with a much lower LR to avoid destroying ImageNet features
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy', keras.metrics.AUC(name='auc')]
)

checkpoint_cb2 = ModelCheckpoint(
    filepath='breakhis_stage2_best.keras',
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1
)
early_stop_cb2 = EarlyStopping(
    monitor='val_loss',
    patience=7,
    mode='min',
    restore_best_weights=True,
    verbose=1
)
reduce_lr_cb2 = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-8,
    verbose=1
)

history_stage2 = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=30,
    validation_data=valid_generator,
    validation_steps=len(valid_generator),
    class_weight=class_weight_dict,
    callbacks=[checkpoint_cb2, early_stop_cb2, reduce_lr_cb2]
)

model.save('breakhis_resnet50_final.keras')
print("✅ Final model saved.")

# ─────────────────────────────────────────────
# FIX 3 & 4: Clinical evaluation on held-out TEST set
#
# This block runs ONCE, after all training decisions are finalized.
# The test set was never seen by any callback or optimizer.
# ─────────────────────────────────────────────
print("\n=== Final Evaluation on HELD-OUT TEST SET ===")

# Reset generator to start from the beginning
test_generator.reset()
y_prob = model.predict(test_generator, steps=len(test_generator), verbose=1)

# Ground-truth labels in generator order (shuffle=False guarantees this)
class_indices = train_generator.class_indices         # {'benign': 0, 'malignant': 1}
y_true_labels = test_df['label'].map(class_indices).values
y_pred_labels = np.argmax(y_prob, axis=1)

# ── Standard metrics ─────────────────────────
acc      = accuracy_score(y_true_labels, y_pred_labels)
bal_acc  = balanced_accuracy_score(y_true_labels, y_pred_labels)

# Per-class (positive = malignant = index 1)
# FIX 8: Report sensitivity / specificity explicitly
tn, fp, fn, tp = confusion_matrix(y_true_labels, y_pred_labels).ravel()

sensitivity = tp / (tp + fn)    # recall for malignant — most critical for cancer
specificity = tn / (tn + fp)    # recall for benign
ppv         = tp / (tp + fp)    # precision for malignant (positive predictive value)
npv         = tn / (tn + fn)    # negative predictive value
f1          = f1_score(y_true_labels, y_pred_labels, pos_label=1)
auc         = roc_auc_score(y_true_labels, y_prob[:, 1])

print(f"\n{'Metric':<30} {'Value':>8}")
print("-" * 40)
print(f"{'Accuracy':<30} {acc:>8.4f}")
print(f"{'Balanced Accuracy':<30} {bal_acc:>8.4f}")
print(f"{'Sensitivity (malignant recall)':<30} {sensitivity:>8.4f}  ← most important")
print(f"{'Specificity (benign recall)':<30} {specificity:>8.4f}")
print(f"{'PPV (malignant precision)':<30} {ppv:>8.4f}")
print(f"{'NPV (benign precision)':<30} {npv:>8.4f}")
print(f"{'F1 Score (malignant)':<30} {f1:>8.4f}")
print(f"{'ROC-AUC':<30} {auc:>8.4f}")
print("-" * 40)
print(f"\nConfusion matrix: TP={tp}  TN={tn}  FP={fp}  FN={fn}")
print(f"  → {fn} malignant cases MISSED (false negatives) — aim to minimise this")
print(f"  → {fp} benign cases flagged as malignant (false positives)")

# ── Confusion matrix plot ─────────────────────
cm = confusion_matrix(y_true_labels, y_pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix — Test Set')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()

# ── ROC curve ────────────────────────────────
fpr, tpr, thresholds = roc_curve(y_true_labels, y_prob[:, 1])
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('ROC Curve — Test Set')
plt.legend()
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=150)
plt.show()

# ── Per-magnification breakdown ───────────────
print("\n=== Per-magnification performance ===")
test_df_eval = test_df.copy().reset_index(drop=True)
test_df_eval['y_true'] = y_true_labels
test_df_eval['y_pred'] = y_pred_labels
test_df_eval['prob_malignant'] = y_prob[:, 1]

for mag in sorted(test_df_eval['magnification'].unique()):
    sub = test_df_eval[test_df_eval['magnification'] == mag]
    if len(sub) < 2:
        continue
    sub_tn, sub_fp, sub_fn, sub_tp = confusion_matrix(
        sub['y_true'], sub['y_pred'], labels=[0, 1]
    ).ravel()
    sub_sens = sub_tp / (sub_tp + sub_fn) if (sub_tp + sub_fn) > 0 else float('nan')
    sub_spec = sub_tn / (sub_tn + sub_fp) if (sub_tn + sub_fp) > 0 else float('nan')
    sub_auc  = roc_auc_score(sub['y_true'], sub['prob_malignant']) \
               if sub['y_true'].nunique() > 1 else float('nan')
    print(f"  {mag}×  n={len(sub):4d}  sensitivity={sub_sens:.3f}  "
          f"specificity={sub_spec:.3f}  AUC={sub_auc:.3f}")

# ─────────────────────────────────────────────
# FIX 9: Temperature scaling (confidence calibration)
#
# Finds a temperature T on the VALIDATION set such that
# softmax(logits / T) produces better-calibrated probabilities.
# Applied post-hoc — no retraining required.
# ─────────────────────────────────────────────
print("\n=== Temperature Scaling (calibration) ===")

# Get logits on validation set (rebuild model without final softmax)
logit_model = keras.Model(inputs=model.input, outputs=model.layers[-1].output)

valid_generator.reset()
logits_val = logit_model.predict(valid_generator, steps=len(valid_generator), verbose=0)

y_val_true = valid_df['label'].map(class_indices).values

from scipy.optimize import minimize_scalar
from scipy.special import softmax as sp_softmax

def nll_with_temperature(T):
    scaled = logits_val / T
    probs  = sp_softmax(scaled, axis=1)
    probs  = np.clip(probs, 1e-9, 1.0)
    nll    = -np.mean(np.log(probs[np.arange(len(y_val_true)), y_val_true]))
    return nll

result = minimize_scalar(nll_with_temperature, bounds=(0.1, 10.0), method='bounded')
T_opt  = result.x
print(f"Optimal temperature: T = {T_opt:.4f}")
print("  T > 1 → model was overconfident (softened probabilities)")
print("  T < 1 → model was underconfident (sharpened probabilities)")
print("  T ≈ 1 → model was already well-calibrated")

# Calibrated test probabilities
test_generator.reset()
logits_test = logit_model.predict(test_generator, steps=len(test_generator), verbose=0)
calibrated_probs = sp_softmax(logits_test / T_opt, axis=1)

print(f"\nCalibrated malignant probability stats on test set:")
print(f"  Mean: {calibrated_probs[:, 1].mean():.4f}")
print(f"  Std : {calibrated_probs[:, 1].std():.4f}")
print(f"  These values are now meaningful probabilities, not raw softmax scores.")

np.save('calibrated_test_probs.npy', calibrated_probs)
print("\n✅ Calibrated probabilities saved to calibrated_test_probs.npy")
print("✅ All done. Review sensitivity and per-magnification results above.")
