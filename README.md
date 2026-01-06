# Titanic Survival Prediction

This project demonstrates a **Machine Learning pipeline** for predicting Titanic survival using multiple models and automated feature engineering.

## 🔹 Features
- Pclass, Sex, Age, Fare, Embarked
- FamilySize, IsAlone (engineered features)

## 🔹 Models
- Logistic Regression
- Random Forest
- Gradient Boosting

## 🔹 Pipeline
- Handling missing values
- Scaling numeric features
- One-hot encoding categorical features
- Hyperparameter tuning with GridSearchCV
- F1-score evaluation

## 🔹 Usage
1. Clone the repository
2. Place `titanic.csv` in the `data/` folder
3. Run:

```bash
python src/train_models.py
