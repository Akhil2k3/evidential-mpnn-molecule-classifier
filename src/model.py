import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class EdgeNetwork(layers.Layer):
    def build(self, input_shape):
        self.atom_dim = input_shape[0][-1]
        self.bond_dim = input_shape[1][-1]
        self.kernel = self.add_weight(
            shape=(self.bond_dim, self.atom_dim * self.atom_dim),
            initializer="glorot_uniform",
            name="kernel",
        )
        self.bias = self.add_weight(
            shape=(self.atom_dim * self.atom_dim),
            initializer="zeros",
            name="bias",
        )
        super().build(input_shape)

    def call(self, inputs):
        atom_features, bond_features, pair_indices = inputs
        bond_features = tf.matmul(bond_features, self.kernel) + self.bias
        bond_features = tf.reshape(bond_features, (-1, self.atom_dim, self.atom_dim))
        atom_features_neighbors = tf.gather(atom_features, pair_indices[:, 1])
        atom_features_neighbors = tf.expand_dims(atom_features_neighbors, axis=-1)
        transformed_features = tf.matmul(bond_features, atom_features_neighbors)
        transformed_features = tf.squeeze(transformed_features, axis=-1)
        aggregated_features = tf.math.unsorted_segment_sum(
            transformed_features,
            pair_indices[:, 0],
            num_segments=tf.shape(atom_features)[0],
        )
        return aggregated_features

class MessagePassing(layers.Layer):
    def __init__(self, units, steps=4, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.steps = steps

    def build(self, input_shape):
        self.atom_dim = input_shape[0][-1]
        self.message_step = EdgeNetwork()
        self.pad_length = max(0, self.units - self.atom_dim)
        self.update_step = layers.GRUCell(self.atom_dim + self.pad_length)
        super().build(input_shape)

    def call(self, inputs):
        atom_features, bond_features, pair_indices = inputs
        atom_features_updated = tf.pad(atom_features, [(0, 0), (0, self.pad_length)])
        for i in range(self.steps):
            atom_features_aggregated = self.message_step(
                [atom_features_updated, bond_features, pair_indices]
            )
            atom_features_updated, _ = self.update_step(
                atom_features_aggregated, atom_features_updated
            )
        return atom_features_updated

class PartitionPadding(layers.Layer):
    def __init__(self, batch_size, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size

    def call(self, inputs):
        atom_features, molecule_indicator = inputs
        atom_features_partitioned = tf.dynamic_partition(
            atom_features, molecule_indicator, self.batch_size
        )
        num_atoms = [tf.shape(f)[0] for f in atom_features_partitioned]
        max_num_atoms = tf.reduce_max(num_atoms)
        atom_features_stacked = tf.stack(
            [tf.pad(f, [(0, max_num_atoms - n), (0, 0)]) for f, n in zip(atom_features_partitioned, num_atoms)],
            axis=0,
        )
        gather_indices = tf.where(tf.reduce_sum(atom_features_stacked, axis=(1, 2)) != 0)
        gather_indices = tf.squeeze(gather_indices, axis=-1)
        return tf.gather(atom_features_stacked, gather_indices, axis=0)

class TransformerEncoderReadout(layers.Layer):
    def __init__(self, num_heads=8, embed_dim=64, dense_dim=512, batch_size=32, **kwargs):
        super().__init__(**kwargs)
        self.partition_padding = PartitionPadding(batch_size)
        self.attention = layers.MultiHeadAttention(num_heads, embed_dim)
        self.dense_proj = keras.Sequential(
            [
                layers.Dense(dense_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.average_pooling = layers.GlobalAveragePooling1D()

    def call(self, inputs):
        x = self.partition_padding(inputs)
        padding_mask = tf.reduce_any(tf.not_equal(x, 0.0), axis=-1)
        padding_mask = padding_mask[:, tf.newaxis, tf.newaxis, :]
        attention_output = self.attention(x, x, attention_mask=padding_mask)
        proj_input = self.layernorm_1(x + attention_output)
        proj_output = self.layernorm_2(proj_input + self.dense_proj(proj_input))
        return self.average_pooling(proj_output)

def MPNNModel(atom_dim, bond_dim, batch_size=32, message_units=64, message_steps=4, num_attention_heads=8, dense_units=512):
    atom_features_in = keras.Input((atom_dim,), dtype="float32", name="atom_features")
    bond_features_in = keras.Input((bond_dim,), dtype="float32", name="bond_features")
    pair_indices_in = keras.Input((2,), dtype="int32", name="pair_indices")
    molecule_indicator_in = keras.Input((), dtype="int32", name="molecule_indicator")
    
    x = MessagePassing(message_units, message_steps)(
        [atom_features_in, bond_features_in, pair_indices_in]
    )
    x = TransformerEncoderReadout(num_attention_heads, message_units, dense_units, batch_size)(
        [x, molecule_indicator_in]
    )
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dense(1, activation="sigmoid")(x)
    
    model = keras.Model(
        inputs=[atom_features_in, bond_features_in, pair_indices_in, molecule_indicator_in],
        outputs=[x]
    )
    return model

