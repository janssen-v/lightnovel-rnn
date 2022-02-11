import tensorflow as tf
import numpy as np
import os
import time

# Download Dataset
filepath = tf.keras.utils.get_file('light_novel_titles_clean_v2.txt', 'https://drive.google.com/uc?export=download&id=13ExvJcOr0l8LZD1gHCiYJn2C_pKx8v1D')

# Import Data
text = open(filepath, 'rb').read().decode(encoding='utf-16')
vocab = sorted(set(text))

# Text Processing
chars_to_ids = tf.keras.layers.StringLookup(vocabulary=list(vocab), mask_token=None)
ids_to_chars = tf.keras.layers.StringLookup(vocabulary=chars_to_ids.get_vocabulary(), invert=True, mask_token=None)

def text_from_ids(ids):
  return tf.strings.reduce_join(ids_to_chars(ids), axis =-1)

# Training Examples
all_ids = chars_to_ids(tf.strings.unicode_split(text,'UTF-8'))
ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)
seq_length = 100
examples_per_epoch = len(text)//(seq_length)+1
sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)

def split_input_target(sequence):
  input_text = sequence[:-1]
  target_text = sequence[1:]
  return input_text, target_text

dataset = sequences.map(split_input_target)

# Create Training Batch
# Batch size
BATCH_SIZE = 64
# Buffer size for dataset shuffle
BUFFER_SIZE = 10000

dataset = (
    dataset
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.experimental.AUTOTUNE))

# Build Model

vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024

class MyModel(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, rnn_units):
    super().__init__(self)
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(rnn_units,
                                   return_sequences=True,
                                   return_state=True)
    self.dense = tf.keras.layers.Dense(vocab_size)
                                    
  def call(self, inputs, states=None, return_state=False, training=False):
    x = inputs
    x = self.embedding(x, training=training)
    if states is None:
      states = self.gru.get_initial_state(x)
    x, states = self.gru(x, initial_state=states, training=training)
    x = self.dense(x, training=training)

    if return_state:
      return x, states
    else:
      return x
  
model = MyModel(
    # Be sure vocabulary size matches the 'StringLookup' layers.
    vocab_size=len(chars_to_ids.get_vocabulary()),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units)
  

# Model Training
loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
for input_example_batch, target_example_batch in dataset.take(1):
  example_batch_predictions = model(input_example_batch)
example_batch_mean_loss = loss(target_example_batch, example_batch_predictions)
model.compile(optimizer='adam', loss=loss)

# Checkpoint save directory
checkpoint_dir = './training_checkpoints'
# Name of checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

# EPOCHS
EPOCHS=100
history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

# TEMPERATURE
TEMP = 0.5

# One Step Predictor
class OneStep(tf.keras.Model):
  def __init__(self, model, ids_to_chars, chars_to_ids, temperature=TEMP):
    super().__init__()
    self.temperature = temperature
    self.model = model
    self.ids_to_chars = ids_to_chars
    self.chars_to_ids = chars_to_ids

    # Create a mask to prevent "[UNK]" from being generated.
    skip_ids = self.chars_to_ids(['[UNK]'])[:, None]
    sparse_mask = tf.SparseTensor(
        # Put a -inf at each bad index.
        values=[-float('inf')]*len(skip_ids),
        indices=skip_ids,
        # Match the shape to the vocabulary
        dense_shape=[len(chars_to_ids.get_vocabulary())])
    self.prediction_mask = tf.sparse.to_dense(sparse_mask)

  @tf.function
  def generate_one_step(self, inputs, states=None):
    # Convert strings to token IDs.
    input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
    input_ids = self.chars_to_ids(input_chars).to_tensor()

    # Run model
    # predicted_logits.shape is [batch, char, next_char_logits]
    predicted_logits, states = self.model(inputs=input_ids, states=states,
                                          return_state=True)
    # Only use the last prediction.
    predicted_logits = predicted_logits[:, -1, :]
    predicted_logits = predicted_logits/self.temperature
    # Apply prediction mask: prevent UNK generation.
    predicted_logits = predicted_logits + self.prediction_mask

    # Sample output logits to generate token IDs.
    predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
    predicted_ids = tf.squeeze(predicted_ids, axis=-1)

    # Convert token ids to characters
    predicted_chars = self.ids_to_chars(predicted_ids)

    # Return characters and model state.
    return predicted_chars, states

one_step_model = OneStep(model, ids_to_chars, chars_to_ids)

SEED = "Is this a light novel title?"

# Output

start = time.time()
states = None
# Start String
next_char = tf.constant([SEED])
result = [next_char]

for n in range(1000):
  next_char, states = one_step_model.generate_one_step(next_char, states=states)
  result.append(next_char)

result = tf.strings.join(result)
result = result[0].numpy().decode('utf-8')
# TODO: Check if generated results are in source dataset, if yes -> remove

f = open("results.txt", "x")
f.write(result)
f.close()
end = time.time()

print('\nRun time:', end - start)