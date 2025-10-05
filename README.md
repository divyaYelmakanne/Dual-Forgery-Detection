# Image Forgery Detection with Heatmap Visualization

This project is a Streamlit-based web application for detecting image forgeries, specifically focusing on two common types: **Inpainting Forgery** and **Copy-Move Forgery**. It uses deep learning models to classify images as real or fake and provides Grad-CAM heatmaps to visualize the regions contributing to the forgery detection.

## Features

- **Inpainting Forgery Detection**: Uses a custom SimpleCNN model trained to detect images altered by inpainting techniques.
- **Copy-Move Forgery Detection**: Employs a fine-tuned ResNet18 model to identify copy-move manipulations.
- **Grad-CAM Heatmaps**: Generates visual heatmaps highlighting suspicious regions in fake images.
- **User-Friendly Interface**: Simple Streamlit web app for uploading images and viewing results.
- **Model Training Scripts**: Includes scripts to train both models on custom datasets.

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd Image-Forgery-Detection--inpainting-and-copy-move-forgery-with-heatmap-main
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

   Additional dependencies (not in requirements.txt, install manually):
   ```
   pip install streamlit torch torchvision pytorch-grad-cam scikit-learn matplotlib seaborn tqdm
   ```

3. Ensure you have the pre-trained models in the `models/` directory:
   - `forgery_classifier.pth` (for inpainting detection)
   - `best_model.pth` (for copy-move detection)

## Usage

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Open your browser and navigate to the provided local URL (usually `http://localhost:8501`).

3. Upload an image (JPG, JPEG, or PNG) using the file uploader.

4. View the detection results:
   - Prediction for each forgery type (Real or Fake) with confidence scores.
   - If fake, a Grad-CAM heatmap will be displayed showing the manipulated regions.

## Training the Models

### Inpainting Forgery Detection Model

1. Prepare your dataset with `real/` and `fake/` subdirectories containing images.

2. Run the training script:
   ```
   python train.py
   ```

   This will train a SimpleCNN model and save it as `forgery_classifier.pth`.

### Copy-Move Forgery Detection Model

1. Prepare your dataset (see `train_cm.py` for data preparation details).

2. Run the training script:
   ```
   python train_cm.py
   ```

   This will train a ResNet18 model and save it as `best_model.pth`.

## File Descriptions

- `app.py`: Main Streamlit application for the web interface.
- `model.py`: Defines the SimpleCNN architecture for inpainting detection.
- `predictor.py`: Functions to load models and make predictions.
- `preprocess.py`: Image preprocessing utilities.
- `gradcam.py`: Grad-CAM implementation for heatmap generation.
- `utils.py`: Additional utility functions for Grad-CAM.
- `train.py`: Training script for the inpainting detection model.
- `train_cm.py`: Training script for the copy-move detection model.
- `generate_copy_move.py`: Script to generate copy-move forgery images from real images.
- `generate_masks_and_masked.py`: Script to create masked images for inpainting forgery simulation.
- `grad_cam.py`: Standalone script to generate Grad-CAM for a single image.
- `resize_and_copy.py`: Utility to resize and copy images to the dataset directory.
- `requirements.txt`: List of basic dependencies.

## Dataset Preparation

- For inpainting: Use `generate_masks_and_masked.py` to create fake images from real ones.
- For copy-move: Use `generate_copy_move.py` to generate forged images.
- Organize datasets into `train/` and `val/` directories with `real/` and `fake/` subfolders.

## Model Architecture

- **Inpainting Model**: SimpleCNN with convolutional layers followed by fully connected layers.
- **Copy-Move Model**: ResNet18 with a modified final layer for binary classification.

## Contributing

Feel free to contribute by improving the models, adding more forgery types, or enhancing the UI.

## License

This project is open-source. Please check the license file for details.
