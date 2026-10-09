# 05_load_split.py
# Goal: load the saved embeddings and split them into train and test sets.
# Using the is_training column (1 = train, 0 = test).

import numpy as np  # to load the .npz file and work with the arrays.

# Load the saved file. np.load opens the .npz and it works like a dictionary:
# We get each array back by the name we saved it under (X, y, gender, ...).
data = np.load("features/facenet_embeddings.npz", allow_pickle=True)

X = data["X"]                       # embeddings, shape (3954, 512)
y = data["y"]                       # bmi values, shape (3954, )
gender = data["gender"]             # 1 male, 0 female
is_training = data["is_training"]   # 1 train, 0 test
names = data["names"]               # image file names

# Sanity check: print the shapes so we confirm everything loaded correctly.
print("X:", X.shape, "| y:", y.shape, "| gender:", gender.shape)

# Make a boolean mask: True where this row belongs to the training set.
# is_training == 1 turns the array into True/False (True where value is 1).
train_mask = is_training == 1
test_mask = is_training == 0

# Use the mask to pick only the matching rows.
# X[train_mask] keeps the rows where train_mask is True.
X_train = X[train_mask]
y_train = y[train_mask]
X_test = X[test_mask]
y_test = y[test_mask]

# Print how many ended up in each group.
print("train:", X_train.shape[0], "faces")
print("test:", X_test.shape[0], "faces") 

