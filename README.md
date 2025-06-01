# MNIST Digit Classification

This project demonstrates handwritten digit classification using the MNIST dataset. Two different classification models are explored: a Binary Stochastic Gradient Descent (SGD) Classifier for identifying the digit '5', and a K-Nearest Neighbors (KNN) classifier for multi-class classification of all digits (0-9).

## Dataset

The project uses the MNIST dataset, a large database of handwritten digits that is commonly used for training various image processing systems. The dataset contains 70,000 images, each being a 28x28 pixel image.

## Models

1.  **Binary SGD Classifier**: A linear model trained to classify whether an image represents the digit '5' or not.
2.  **K-Nearest Neighbors (KNN) Classifier**: A non-parametric model used for multi-class classification of all ten digits. Hyperparameter tuning is performed using `GridSearchCV` to find the best number of neighbors and weighting scheme.

## Analysis and Results

The notebook includes steps for:

*   Loading and exploring the MNIST dataset.
*   Visualizing an example digit.
*   Training a binary classifier for the digit '5' using SGD.
*   Evaluating the binary classifier using a confusion matrix, precision, recall, and F1-score.
*   Comparing the ROC curve of the SGD classifier with a Random Forest classifier.
*   Performing multi-class classification using a KNN classifier.
*   Tuning hyperparameters for the KNN classifier using `GridSearchCV`.
*   Evaluating the performance of the best KNN model on the test set using accuracy and a classification report.

The results of the model evaluations, including the confusion matrix, precision, recall, F1-score, ROC curves, and the final test set accuracy for the KNN classifier, are presented in the notebook.

## Dependencies

The following libraries are required to run this notebook:

*   `sklearn`
*   `matplotlib`
*   `numpy`
*   `pandas`
