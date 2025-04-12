import tensorflow as tf

def prepare_batch(x_batch, y_batch):
    """
    Merges (sub)graphs of batch into a single global (disconnected) graph.
    """
    atom_features, bond_features, pair_indices = x_batch
    num_atoms = atom_features.row_lengths()
    num_bonds = bond_features.row_lengths()
    molecule_indices = tf.range(len(num_atoms))
    molecule_indicator = tf.repeat(molecule_indices, num_atoms)
    
    # Compute increment for global graph merging
    gather_indices = tf.repeat(molecule_indices[:-1], num_bonds[1:])
    increment = tf.cumsum(num_atoms[:-1])
    increment = tf.pad(tf.gather(increment, gather_indices), [(num_bonds[0], 0)])
    
    pair_indices = pair_indices.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
    pair_indices = pair_indices + increment[:, tf.newaxis]
    atom_features = atom_features.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
    bond_features = bond_features.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
    return (atom_features, bond_features, pair_indices, molecule_indicator), y_batch

def MPNNDataset(X, y, batch_size=32, shuffle=False):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        dataset = dataset.shuffle(1024)
    return (
        dataset.batch(batch_size)
               .map(prepare_batch, num_parallel_calls=tf.data.AUTOTUNE)
               .prefetch(tf.data.AUTOTUNE)
    )
