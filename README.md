# trainingCat-Dog

A convolutional neural network (CNN) built with Keras/TensorFlow to classify images of cats vs. dogs.

## Model

`model.py` defines and trains a Sequential CNN:

- 3 convolutional blocks (`Conv2D` + `MaxPooling2D`), with 32 → 64 → 128 filters
- A dense head (`Flatten` → `Dense(512)` → `Dropout(0.5)` → `Dense(2, softmax)`)
- Trained with data augmentation (rotation, zoom, shift, shear, horizontal flip)
- Compiled with the Adam optimizer and categorical cross-entropy loss

After training, the script saves the model to `cat_dog_model.keras`, evaluates it on the test set, and runs a sample prediction.

## Expected data layout

The script expects two local directories, each containing one subfolder per class (`cats/`, `dogs/`):

```
training_data/
├── cats/
└── dogs/
testing_data/
├── cats/
└── dogs/
```

These directories (and the resulting model files) are not versioned — see `.gitignore` — since the dataset and trained weights are too large for the repository. Update the `TRAIN_DIR` and `TEST_DIR` constants at the top of `model.py` to point to your local dataset location.

## Usage

```bash
pip install tensorflow numpy
python3 model.py
```

## License

Distributed under the MIT License — see [LICENSE](LICENSE).
