from tensorflow.keras.models import load_model


model = load_model('tag_correction_model.keras')

print(model.summary())
print(model.get_config())

for layer in model.layers:
    print(layer.name, layer)

lstm_weights = model.get_layer('lstm').get_weights()
print("LSTM kernel shape:", lstm_weights[0].shape)  # kernel (input weights)
print("LSTM recurrent kernel shape:", lstm_weights[1].shape)  # recurrent weights
print("LSTM bias shape:", lstm_weights[2].shape)

dense_weights = model.get_layer('dense').get_weights()
print("Dense kernel shape:", dense_weights[0].shape)
print("Dense bias shape:", dense_weights[1].shape)
