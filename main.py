import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from rdkit import Chem
from rdkit.Chem.Draw import MolsToGridImage

# Suppress TensorFlow logs and warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
np.random.seed(42)
tf.random.set_seed(42)

# Import modules from src
from src.graph_utils import graphs_from_smiles
from src.dataset import MPNNDataset
from src.model import MPNNModel

# Load dataset
csv_path = keras.utils.get_file("BBBP.csv", "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv")
df = pd.read_csv(csv_path, usecols=[1, 2, 3])

# Shuffle and split indices into train (80%), valid (19%), and test (1%)
permuted_indices = np.random.permutation(np.arange(df.shape[0]))
train_index = permuted_indices[: int(df.shape[0] * 0.8)]
valid_index = permuted_indices[int(df.shape[0] * 0.8) : int(df.shape[0] * 0.99)]
test_index = permuted_indices[int(df.shape[0] * 0.99):]

x_train = graphs_from_smiles(df.iloc[train_index].smiles)
y_train = df.iloc[train_index].p_np

x_valid = graphs_from_smiles(df.iloc[valid_index].smiles)
y_valid = df.iloc[valid_index].p_np

x_test = graphs_from_smiles(df.iloc[test_index].smiles)
y_test = df.iloc[test_index].p_np

# Initialize model using dimensions from training data
atom_dim = x_train[0][0][0].shape[0]
bond_dim = x_train[1][0][0].shape[0]
mpnn = MPNNModel(atom_dim, bond_dim)

mpnn.compile(
    loss=keras.losses.BinaryCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=5e-4),
    metrics=[keras.metrics.AUC(name="AUC")]
)

# (Optional) Visualize the model architecture
keras.utils.plot_model(mpnn, show_dtype=True, show_shapes=True)

# Create datasets
train_dataset = MPNNDataset(x_train, y_train)
valid_dataset = MPNNDataset(x_valid, y_valid)
test_dataset = MPNNDataset(x_test, y_test)

# Train model
history = mpnn.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=50,
    verbose=2,
    class_weight={0: 2.0, 1: 0.5}
)

# Plot training AUC
plt.figure(figsize=(10, 6))
plt.plot(history.history["AUC"], label="train AUC")
plt.plot(history.history["val_AUC"], label="valid AUC")
plt.xlabel("Epochs", fontsize=16)
plt.ylabel("AUC", fontsize=16)
plt.legend(fontsize=16)
plt.show()

# Uncertainty Prediction & Visualization on Test Set
molecules = [Chem.MolFromSmiles(df.smiles.values[index]) for index in test_index]
y_true = [df.p_np.values[index] for index in test_index]
y_pred = tf.squeeze(mpnn.predict(test_dataset), axis=1)

legends = [f"y_true/y_pred = {y_true[i]}/{y_pred[i]:.2f}" for i in range(len(y_true))]
grid_image = MolsToGridImage(molecules, molsPerRow=5, legends=legends)
grid_image.save("uncertainty_grid.png")
print("Saved uncertainty grid image as 'uncertainty_grid.png'.")

