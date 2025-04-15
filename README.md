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