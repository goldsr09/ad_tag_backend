import pandas as pd
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense

# === 1. Load CSV Data ===
df = pd.read_csv('/Users/rgoldstein/ad_tag_backend/tag_pairs_export.csv')

# Ensure the columns are named exactly as in your CSV
broken_tags = df['original_tag'].astype(str).tolist()
fixed_tags = df['transformed_tag'].astype(str).tolist()

# === 2. Build Character Vocabulary ===
all_text = ''.join(broken_tags + fixed_tags)
chars = sorted(list(set(all_text)))
char2idx = {c: i + 1 for i, c in enumerate(chars)}  # index 0 is padding
idx2char = {i + 1: c for i, c in enumerate(chars)}

max_len = max(max(map(len, broken_tags)), max(map(len, fixed_tags)))

def encode(s):
    return [char2idx[c] for c in s] + [0] * (max_len - len(s))

def decode(seq):
    return ''.join(idx2char.get(i, '') for i in seq if i != 0)

X = np.array([encode(tag) for tag in broken_tags])
y = np.array([encode(tag) for tag in fixed_tags])
y = np.expand_dims(y, -1)

vocab_size = len(chars) + 1

# === 3. Build Seq2Seq Model ===
input_layer = Input(shape=(max_len,))
embed = Embedding(input_dim=vocab_size, output_dim=32, mask_zero=True)(input_layer)
encoded = LSTM(128, return_sequences=True)(embed)
output_layer = Dense(vocab_size, activation='softmax')(encoded)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

# === 4. Train Model ===
model.fit(X, y, epochs=100, batch_size=2)

# After model.fit(...)
model.save('tag_correction_model.keras')

# Save vocab and parameters
import pickle
with open('tag_tokenizer.pkl', 'wb') as f:
    pickle.dump({
        'char2idx': char2idx,
        'idx2char': idx2char,
        'max_len': max_len
    }, f)


# === 5. Test Prediction ===
test_idx = 0
test_tag = broken_tags[test_idx]
encoded_test = np.array([encode(test_tag)])
pred = model.predict(encoded_test)
pred_seq = np.argmax(pred[0], axis=-1)
print("Input:     ", test_tag)
print("Predicted: ", decode(pred_seq))
print("Expected:  ", fixed_tags[test_idx])
