import numpy as np
from tensorflow.keras.models import load_model
import pickle
import tensorflow as tf
# import sys

model = load_model('tag_correction_model.keras')# Load model

# Load tokenizer and parameters
with open('tag_tokenizer.pkl', 'rb') as f:
    data = pickle.load(f)
char2idx = data['char2idx']
idx2char = data['idx2char']
max_len = data['max_len']

# Define encoding/decoding functions
def encode(s):
    return [char2idx.get(c, 0) for c in s] + [0] * (max_len - len(s))

def decode(seq):
    return ''.join(idx2char.get(i, '') for i in seq if i != 0)

# ---- Predict on any tag ----

new_tag = sys.argv[1]

encoded_input = np.array([encode(new_tag)])
pred = model.predict(encoded_input)
pred_seq = np.argmax(pred[0], axis=-1)

predicted_tag = decode(pred_seq)
print("Input Tag:    ", new_tag)
print("Predicted Tag:", predicted_tag)
