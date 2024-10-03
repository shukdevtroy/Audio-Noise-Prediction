# Audio Classification Using MFCCs and LSTM

This repository contains a Python implementation of an audio classification model using Mel-frequency cepstral coefficients (MFCCs) and a Long Short-Term Memory (LSTM) neural network. The model classifies audio into three categories: clean audio, Gaussian noise, and impulse noise.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Data Structure](#data-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Training the Model](#training-the-model)
- [Model Prediction](#model-prediction)
- [License](#license)

## Features

- Loads audio data and computes MFCCs.
- Builds and trains a neural network for audio classification.
- Supports model saving and loading.
- Predicts class labels for new audio files.

## Requirements

Make sure you have the following libraries installed:

- `numpy`
- `librosa`
- `tensorflow`
- `scikit-learn`
- `google.colab` (if using Google Colab)

You can install the required packages using pip:

```bash
pip install numpy librosa tensorflow scikit-learn
```

## Data Structure

Your data directory should be structured as follows:

```
Data Directory/
│
├── clean_audio/
│   ├── audio_file_1.wav
│   ├── audio_file_2.wav
│   └── ...
│
├── gaussian_noise/
│   ├── audio_file_1.wav
│   ├── audio_file_2.wav
│   └── ...
│
└── impulse_noise/
    ├── audio_file_1.wav
    ├── audio_file_2.wav
    └── ...
```

Each subfolder contains `.wav` files corresponding to its label.

## Installation

Clone the repository:

```bash
git clone https://github.com/your_username/audio_classification.git
cd audio_classification
```

## Usage

1. Mount your Google Drive (if using Google Colab):

   ```python
   from google.colab import drive
   drive.mount('/content/drive/')
   ```

2. Load your data:

   ```python
   X, y = load_data('/content/drive/MyDrive/Data Directory/')
   ```

3. Build and train the model:

   ```python
   model = build_model(X_train.shape[1:])
   model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
   model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30)
   ```

4. Save the model:

   ```python
   model.save('/content/drive/MyDrive/Data Directory/audio_classification_model.h5')
   ```

## Training the Model

The model is trained using the following architecture:

- Convolutional layer
- MaxPooling layer
- LSTM layer
- Dense layers for classification

You can customize the number of epochs and other hyperparameters as needed.

## Model Prediction

To make predictions on new audio files, use the following function:

```python
result = predict_audio('/path/to/your/audio.wav', loaded_model)
print(f'Predicted class: {result}')
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or issues, please contact:

- **Email**: shukdevdatta@gmail.com
- **GitHub**: [Click to here to access the Github Profile](https://github.com/shukdevtroy)
- **WhatsApp**: [Click here to chat](https://wa.me/+8801719296601)

## Reference

[Click to here to access the Github Profile](https://github.com/shukdevtroy/Audio-Noise-Generator/)

---

