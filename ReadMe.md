# Salary Data Analysis Project

## Overview
This project provides a comprehensive analysis of salary data, exploring factors that influence compensation in the job market. The analysis includes exploratory data visualization, outlier detection, correlation analysis, and predictive modeling.

## Implementation
The analysis is available in two programming languages:

- **R Script** (`Group.R`): Complete analysis using R's statistical and visualization libraries
- **Python Notebook** (`Group.ipynb`): Equivalent analysis implemented in Python using pandas, scikit-learn, and matplotlib

## Key Features

### Data Preparation
- Cleaning and transformation of salary data
- Categorization of experience levels and employment types
- Creation of income categories (low, middle, high)

### Exploratory Analysis
- Descriptive statistics of salary distributions
- Visualization of salary patterns across different factors
- Identification of top-paying job titles

### Advanced Analysis
- Outlier detection using IQR method
- Correlation analysis between numeric variables
- Normalized salary distributions

### Modeling
- Linear Regression: Basic predictive model
- Random Forest: Advanced model with feature importance
- GAM/Polynomial Regression: Capturing non-linear relationships
- Multinomial Logistic Regression: Classification of income categories

## Visualizations
All visualizations are saved to the `images/` directory, including:
- Salary distributions
- Boxplots by experience level, remote work ratio, and company size
- Top 10 highest-paying job titles
- Outlier detection plots
- Correlation matrices
- Model comparison charts

## Usage
- To run the R analysis: Open and execute `Group.R` in RStudio or R console
- To run the Python analysis: Open `Group.ipynb` in Jupyter Notebook or JupyterLab

## Requirements
### R Dependencies
- ggplot2, dplyr, tidyr, corrplot, caret, mgcv, randomForest, e1071

### Python Dependencies
- pandas, numpy, matplotlib, seaborn, scikit-learn

## Results
The analysis provides insights into salary determinants, identifies outliers, and compares the performance of different predictive models to understand which factors most strongly influence compensation.