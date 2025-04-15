# Number-Classification-PyTorch

A short and simple project that uses a Convolutional Neural Network (CNN) in PyTorch to classify hand-drawn digits (0–9) from a canvas input.

## Description

This project implements a lightweight 2D CNN to recognize digits drawn by the user in a browser-based canvas. It is trained on the MNIST dataset and can predict a digit based on real-time input from the canvas. The drawing is processed, resized to 28x28 grayscale, and passed to the trained model for inference.

The goal was to explore basic digit classification with PyTorch and create a minimal interactive demo using a notebook interface like Google Colab. 

The model is saved as "model.pth"

## Features

- Interactive drawing canvas
- Real-time prediction on digit drawings
- Simple CNN with 2 convolutional layers
- Trained on MNIST dataset
- Lightweight and beginner-friendly

## Tech Stack

- Python
- PyTorch
- IPython Widgets (Canvas)
- Matplotlib

<img src="https://github.com/user-attachments/assets/7c0320d6-98fa-49ce-9f49-8621aa9ee2bf/prediction2" width="300"/> <img src="https://github.com/user-attachments/assets/07c2a3c4-7785-4186-81a6-21e7e587c5b0/prediction" width="300"/>
