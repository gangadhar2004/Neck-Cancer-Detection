# Neck Cancer Detection

## Project Overview

This project is designed to detect and classify images of neck cancer. It uses machine learning models to predict whether an image depicts a normal case or oral squamous cell carcinoma (OSCC).

## Features

- Image Classification: Classifies uploaded images into 'normal' or 'OSCC'.
- Deep Learning Models: Utilizes InceptionV3, VGG16, and ResNet50 models with an ensemble approach for accurate predictions.
- Web Interface: Provides a user-friendly interface for uploading images and receiving predictions.
- 
## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/gangadhar2004/Neck-Cancer-Detection.git
    cd Neck-Cancer-Detection
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the web application:
    ```bash
    python app.py
    ```

## Models Used
- InceptionV3
- ResNet50
- VGG16

## Project Structure
- `app.py`: Main Flask app to handle uploads and predictions.
- `models/`: Directory containing saved models.
- `static/` and `templates/`: Web assets for the interface.



