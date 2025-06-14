# Titanic Death Predictor

This project is a machine learning model designed to predict passenger survival on the Titanic based on demographic and passenger information. The model uses data from the classic Titanic dataset and is implemented in Python.

## Features

- Reads and processes Titanic passenger data from a CSV file
- Applies preprocessing (e.g., handling missing values, encoding categorical variables)
- Trains a classification model to predict survival
- Outputs prediction results for evaluation

## Files

- `Main.py`: Main script that loads data, preprocesses it, trains the model, and makes predictions.
- `Titanic-Dataset.csv`: Dataset used for training and evaluation. Contains passenger information including:
  - `PassengerId`, `Survived`, `Pclass`, `Name`, `Sex`, `Age`, `SibSp`, `Parch`, `Ticket`, `Fare`, `Cabin`, `Embarked`

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Dushmilan/titanic-death-predictor.git
   cd titanic-death-predictor
