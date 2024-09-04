# Pneumonia Detection Using DenseNet-201

This project uses a DenseNet-201 model to classify chest X-ray images into two categories: "NORMAL" (healthy) and "PNEUMONIA" (infected). The model is trained using the Keras library and includes data augmentation to improve generalization.

## Project Structure

- `train/`: Directory containing training images, organized into subfolders `NORMAL/` and `PNEUMONIA/`.
- `test/`: Directory containing test images, organized into subfolders `NORMAL/` and `PNEUMONIA/`.
- `val/`: Directory containing validation images, organized into subfolders `NORMAL/` and `PNEUMONIA/`.
- `cnn-example-pneumonia-densenet201.h5`: File where the trained model's weights are saved.

## Model Architecture

The model is based on the pre-trained DenseNet-201 architecture, which is modified for binary classification. The architecture is as follows:

1. **DenseNet-201 Backbone**: Pre-trained on ImageNet, used for feature extraction.
2. **Global Average Pooling**: Reduces the spatial dimensions of the feature maps.
3. **Dropout Layer**: Used to prevent overfitting.
4. **Batch Normalization**: Normalizes the output of the previous layer.
5. **Dense Layer**: A fully connected layer with a sigmoid activation for binary classification.

## Data Augmentation

Data augmentation is applied to the training data to artificially increase the diversity of the dataset. The following transformations are used:

- Rotation (up to 40 degrees)
- Width and height shifts (up to 20%)
- Shear transformations (up to 20%)
- Zoom (up to 20%)
- Horizontal flips

## Training

The model is trained using the following parameters:

- **Optimizer**: Adam with a learning rate of `1e-4`
- **Loss Function**: Binary cross-entropy
- **Metrics**: Accuracy
- **Epochs**: 25
- **Steps per Epoch**: 100
- **Validation Steps**: 10

### Training Command

```python
history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=25,
    validation_data=test_generator,
    validation_steps=10
)
# pneumonia_detection
