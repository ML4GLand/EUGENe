from EUGENE import EUGENE
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import random
import torch


def random_dna_sequence(length):
    return ''.join(random.choice('ACTG') for _ in range(length))

sequences = [random_dna_sequence(64) for i in range(8)]

# One-hot encode Sequences
integer_encoder = LabelEncoder()
one_hot_encoder = OneHotEncoder()
input_features = []
for sequence in sequences:
  integer_encoded = integer_encoder.fit_transform(list(sequence))
  integer_encoded = np.array(integer_encoded).reshape(-1, 1)
  one_hot_encoded = one_hot_encoder.fit_transform(integer_encoded)
  input_features.append(one_hot_encoded.toarray())

np.set_printoptions(threshold=40)
input_features = np.stack(input_features)
print("Example sequence\n-----------------------")
print('DNA Sequence #1:\n',sequences[0][:10],'...',sequences[0][-10:])
print('One hot encoding of Sequence #1:\n',input_features[0].T)
input_tensor = torch.tensor(input_features).float()
print(input_tensor.size())

# Instantiate a model
eugene = EUGENE()
print(eugene)
input_tensor = input_tensor.reshape(8, 4, 64)
print(input_tensor.size())
print(input_tensor)
print(eugene(input_tensor))
