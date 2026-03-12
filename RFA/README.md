# Pattern Recognition and Machine Learning
This projects includes the evaluation of differents techniques for pattern recognition and machine learning, including:
- Convolutional Neural Networks
- No Parametric Methods
- Lineal Models
- Multilayer Perceptron
- Pretrained Models (ResNet 50)

# Data Processing in all Notebooks
The data processing steps are common for all the notebooks and include:
1. Loading the dataset
2. Preprocessing the data (normalization, resizing, etc.)
3. Splitting the data into training and testing sets
4. Data augmentation (if applicable)

# conv Notebook
Includes the implementation of a Convolutional Neural Network for image classification.
It uses a RandomSearch for hyperparameter tuning and evaluates the model using accuracy and confusion matrix.

# metodos no parametricos Notebook (Non-Parametric Methods)
This notebook explores non-parametric methods for classification, such as K-Nearest Neighbors (KNN), Random Forest and Decision Trees.

# mlp notebook
This notebook focuses on the implementation of a Multilayer Perceptron (MLP) for classification. It includes the architecture of the MLP, the training process and the evaluation of the model using accuracy and confusion matrix.

It also includes a section for hyperparameter tuning using RandomSearch from keras tuner.

# modelos lineales Notebook (Lineal Models)
This notebook focuses on linear models for classification, including Linear Discriminant Analysis (LDA), Quadratic Discriminant Analysis (QDA) and Logistic Regression and Naive Bayes.
In order to apply these models, it was necessary to use PCA for dimensionality reduction.

# modelos preentrenados Notebook (Pretrained Models)
This notebook explores the use of pretrained models for image classification, specifically ResNet 50. It includes the process of fine-tuning the pretrained model and evaluating its performance using accuracy and confusion matrix.

It also includes the evaluation of the model EfficientNet B2.