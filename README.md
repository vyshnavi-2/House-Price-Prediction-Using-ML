# House Price Prediction using Machine Learning

## Overview
This project predicts house prices using the **California Housing Dataset** and the **XGBoost Regressor** model. The dataset is imported directly from the `sklearn.datasets` library, so no manual downloads are required.

## Dataset
- The dataset used is **California Housing Data**, which contains housing data from California, USA.
- Features include:
  - **MedInc**: Median income in block
  - **HouseAge**: Median house age in block
  - **AveRooms**: Average number of rooms per household
  - **AveBedrms**: Average number of bedrooms per household
  - **Population**: Block population
  - **AveOccup**: Average household occupancy
  - **Latitude**: Latitude of block
  - **Longitude**: Longitude of block
- The target variable (`price`) represents the **median house value** in the area.

## Installation & Setup
### **1. Install Dependencies**
Ensure you have the required libraries installed:
```sh
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

## Data Preprocessing
1. **Load the dataset**: The dataset is imported using `fetch_california_housing()`.
2. **Convert to DataFrame**: The dataset is loaded into a Pandas DataFrame.
3. **Check for missing values**: Ensure data integrity.
4. **Statistical analysis**: Compute summary statistics of the dataset.
5. **Correlation Analysis**: Use a heatmap to visualize feature relationships.

## Model Training
- **Algorithm**: XGBoost Regressor (`xgboost.XGBRegressor`)
- **Training**:
```python
from xgboost import XGBRegressor
model = XGBRegressor()
model.fit(X_train, Y_train)
```

## Model Evaluation
- **R-squared Score (R²)**: Measures how well the model explains variance in house prices.
- **Mean Absolute Error (MAE)**: Measures prediction errors.
```python
from sklearn import metrics
r2_score = metrics.r2_score(Y_test, model.predict(X_test))
mae = metrics.mean_absolute_error(Y_test, model.predict(X_test))
print(f'R² Score: {r2_score:.4f}')
print(f'Mean Absolute Error: {mae:.4f}')
```

## Results
- **Training Performance**:
  - R² Score: **~0.9** (varies based on dataset split)
  - MAE: **Low error values indicate good performance**
- **Test Performance**:
  - R² Score: **~0.7-0.8**
  - MAE: **Comparable to training performance**

## Conclusion
- The **XGBoost Regressor** provides strong predictive power for house prices.
- The model performs well with an **R² score above 0.7**.
- Further improvements can be made using **feature engineering** and **hyperparameter tuning**.

## Future Enhancements
- Try different models like **Random Forest** or **Neural Networks**.
- Tune hyperparameters for better accuracy.
- Explore additional features to improve predictions.

## Repository Structure
```
House-Price-Prediction
│── house_price_prediction.ipynb  # Jupyter Notebook with full code
│── README.md                      # Project documentation
```


