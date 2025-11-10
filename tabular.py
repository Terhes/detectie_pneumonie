import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA

# ----------------------
# Load CSV
data = pd.read_csv("Data_Entry_2017.csv")
data = data.dropna(subset=['Patient Gender', 'Finding Labels'])

# Encode gender
data['Patient Gender'] = LabelEncoder().fit_transform(data['Patient Gender'])

# Convert 'Finding Labels' to binary columns
data['Finding Labels'] = data['Finding Labels'].str.split('|')
mlb = MultiLabelBinarizer()
diseases = pd.DataFrame(mlb.fit_transform(data['Finding Labels']),
                        columns=mlb.classes_,
                        index=data.index)
data = pd.concat([data, diseases], axis=1)

# ----------------------
# Classification Task: Predict Pneumonia using MLP
if 'Pneumonia' not in data.columns:
    raise ValueError("Pneumonia column not found in your dataset")

# CRITICAL: Remove target from features to prevent data leakage
y_clf = data['Pneumonia']
feature_columns = [col for col in mlb.classes_ if col in data.columns and col != 'Pneumonia'] + ['Patient Gender']

print(f"Number of features: {len(feature_columns)}")
print(f"Target distribution - No Pneumonia: {(y_clf == 0).sum()}, Pneumonia: {(y_clf == 1).sum()}")
print(f"Pneumonia prevalence: {(y_clf == 1).sum() / len(y_clf) * 100:.2f}%\n")

X_clf = data[feature_columns]

# Scale features
scaler = StandardScaler()
X_clf_scaled = scaler.fit_transform(X_clf)

X_train, X_test, y_train, y_test = train_test_split(
    X_clf_scaled, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

# ----------------------
# Train Multilayer Perceptron (Neural Network)
print("Training Multilayer Perceptron...")

mlp_model = MLPClassifier(
    hidden_layer_sizes=(64, 32),  # Two hidden layers: 64 neurons, then 32 neurons
    activation='relu',
    max_iter=1000,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1
)
mlp_model.fit(X_train, y_train)

y_pred_mlp = mlp_model.predict(X_test)
acc_mlp = accuracy_score(y_test, y_pred_mlp)

print(f"\nMultilayer Perceptron Accuracy: {acc_mlp:.4f}")

# ----------------------
# Create Figure with 2 Subplots
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# ----------------------
# PLOT 1: MLP "Regression View" (Probability Predictions)
mlp_proba = mlp_model.predict_proba(X_test)[:, 1]  # Probability of Pneumonia

axes[0].scatter(y_test, mlp_proba, alpha=0.6, c=y_test, cmap='coolwarm', 
                edgecolors='k', s=50)
axes[0].axhline(0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold (0.5)')
axes[0].set_xlabel("True Pneumonia Label (0=No, 1=Yes)", fontsize=12, fontweight='bold')
axes[0].set_ylabel("Predicted Probability of Pneumonia", fontsize=12, fontweight='bold')
axes[0].set_title("MLP: Probability Predictions vs True Labels", fontsize=14, fontweight='bold')
axes[0].set_ylim(-0.05, 1.05)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Add accuracy text
axes[0].text(0.02, 0.98, f'Accuracy: {acc_mlp:.2%}', 
             transform=axes[0].transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

# ----------------------
# PLOT 2: MLP Decision Boundary in 2D (using PCA)
# Reduce to 2D for visualization
pca = PCA(n_components=2)
X_test_2d = pca.fit_transform(X_test)

# Create mesh grid for decision boundary
h = 0.02  # step size
x_min, x_max = X_test_2d[:, 0].min() - 1, X_test_2d[:, 0].max() + 1
y_min, y_max = X_test_2d[:, 1].min() - 1, X_test_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Train MLP on 2D PCA data for visualization
mlp_2d = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    max_iter=1000,
    random_state=42,
    early_stopping=True
)
X_train_2d = pca.transform(X_train)
mlp_2d.fit(X_train_2d, y_train)

# Predict on mesh
Z = mlp_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundary
axes[1].contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
axes[1].contour(xx, yy, Z, colors='black', linewidths=2, linestyles='--')

# Plot test points
scatter = axes[1].scatter(X_test_2d[:, 0], X_test_2d[:, 1], 
                          c=y_test, cmap='coolwarm', 
                          edgecolors='k', s=50, alpha=0.7)
axes[1].set_xlabel("First Principal Component", fontsize=12, fontweight='bold')
axes[1].set_ylabel("Second Principal Component", fontsize=12, fontweight='bold')
axes[1].set_title("MLP: Non-Linear Decision Boundary (PCA 2D Projection)", fontsize=14, fontweight='bold')

# Add colorbar
cbar = plt.colorbar(scatter, ax=axes[1])
cbar.set_label('Pneumonia (0=No, 1=Yes)', fontsize=10)

# Calculate accuracy on 2D projection
y_pred_2d = mlp_2d.predict(X_test_2d)
accuracy_2d = accuracy_score(y_test, y_pred_2d)
axes[1].text(0.02, 0.98, f'2D Accuracy: {accuracy_2d:.2%}', 
             transform=axes[1].transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

plt.tight_layout()
plt.show()

# ----------------------
# Print Model Details
print("\n" + "="*60)
print("MULTILAYER PERCEPTRON - Confusion Matrix:")
print("="*60)
print(confusion_matrix(y_test, y_pred_mlp))

print("\n" + "="*60)
print("Classification Report:")
print("="*60)
print(classification_report(y_test, y_pred_mlp, target_names=['No Pneumonia', 'Pneumonia']))

print("\n" + "="*60)
print("MODEL ARCHITECTURE:")
print("="*60)
print(f"Input Layer:    {len(feature_columns)} features")
print(f"Hidden Layer 1: 64 neurons (ReLU activation)")
print(f"Hidden Layer 2: 32 neurons (ReLU activation)")
print(f"Output Layer:   1 neuron (probability)")
print(f"Total Parameters: ~{len(feature_columns)*64 + 64*32 + 32*1:,}")
print("="*60)