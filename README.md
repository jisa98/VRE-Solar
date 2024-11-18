# VRE-Solar

### Variable Renewable Energy (VRE) assessment and forecast
Assessment and forecasting of the variable reneweable energy pv capacity factor using different climate variables

## Research Question: 
1. How accurately can we predict the PV capacity factor using climate variables?
2. Which climate variables are most influential in determining PV capacity factor across different regions in France?

## Project objectives
<div class="alert alert-block alert-info">

- Assess the solar photovoltaic hourly production over in metropolitan France regions using climate data and capacity factor observations

- Predict the VRE power (capacity factor) ahead of time in the next few days/a week


## Motivation and Description
### Methodology Type:

- Using supervised learning since the task involves predicting a numerical target variable (PV capacity factor) based on known input variables (climate data)

- Utilizing and learning 2/3 methods of Machine Learning (starting from the easiest to more complex)


### Steps and Strategies:

#### Data Preprocessing:

1. Handle Missing Data: Replace or remove missing values (e.g., NaNs) in the dataset to ensure no errors during modeling.

2. Feature Engineering: Compute regional averages of climate variables (if necessary) and remove redundant or non-informative features (e.g., constant columns or zeros).

3. Scaling: Normalize or standardize the climate variable values to ensure all features have comparable scales.

#### Exploratory Data Analysis:

Analyze relationships between climate variables and the PV capacity factor using correlation matrices and scatter plots.
Perform feature importance analysis using initial linear regression models to identify potential key drivers.

#### Model Selection:

Start with simple models and progressively increase complexity to improve results:

- Baseline Model: Linear Regression with OLS between Climate Variables and Capacity Factor to establish a baseline performance.
- Advanced Models: Using Lasso Regression to find R2/ bagging Regressor, Feature Importances using MDI, Choose between the stacking regressor, AdaBoost regressor, and  voting regressor (compare between DecisionTree, Random Forest and Lasso in each regressor method)
- Neural Networks: 3. RNN -LTSM (If we have time)

## Feature Selection Models:
- Apply Lasso Regression to select the most significant features.
- Use mutual information and correlation matrix to rank the predictive power of each climate variable.

### Evaluation Methods:

- Train-Test Split: Divide the data into training and test sets (e.g., 80%-20%).
- Cross-Validation: Use k-fold cross-validation (e.g., 5-fold) to evaluate model robustness. --> Nested Cross Validation (bonus)

Metrics: Evaluate the models using:
- RMSE (Root Mean Squared Error): To quantify the difference between predicted and actual values.
- R2 (Coefficient of Determination): To assess how well the model explains the variance in the target variable.
- MAE (Mean Absolute Error): To capture prediction error robustness.

### Feature Importance and Interpretability:

- Very useful for tree-based models, use of feature importance scores to identify which climate variables influence PV capacity the most.
- Analyze results regionally to understand spatial variability.

## Complexity Management:

- Start simple (e.g. linear regression) and introduce complexity incrementally.
- Apply regularization techniques (L1 or L2 penalties) in linear models to reduce overfitting.
- Use hyperparameter tuning (e.g. grid search or Bayesian optimization) for tree-based models to balance model complexity with predictive performance.

## Final Model Validation:

Evaluate the best-performing model on the test set to assess generalizability.
Perform robustness checks: 
- Testing on subsets of regions to see if relationships hold consistently.
- Testing on unseen climate variables or time ranges (if data allows).

## Description of Data
- Inputs: 
1. Climate Variables (netcfd) corresponding to each region in France (surface downward radiation, surface temperature, surface density, surface specific humidity) --> polynomials degree 2

- Target Output: The data that we are going to try to predict
1. Capacity Factor(Monthly Capacity Factors)








</div>
