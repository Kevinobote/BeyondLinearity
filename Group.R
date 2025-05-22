# Comprehensive Analysis of Salary Data
# Load necessary libraries
library(ggplot2)
library(dplyr)
library(tidyr)
library(corrplot)
library(caret)
library(mgcv)
library(randomForest)
library(e1071)

# Set working directory if needed
# setwd("/home/obote/Documents/Strathmore DSA/Module 3/Linear Models/R")

# Create images directory if it doesn't exist
dir.create("images", showWarnings = FALSE)

# Load data
df <- read.csv("data/salaries.csv")

# Display the structure of the data
str(df)

# Display the first few rows
head(df, 5)

# Summary of the data
summary(df)

# ---------------------------------------------------------
# 1. DATA PREPARATION AND CLEANING
# ---------------------------------------------------------

# Check for missing values
missing_values <- colSums(is.na(df))
print("Missing values per column:")
print(missing_values)

# Convert categorical variables to factors
df$experience_level <- factor(df$experience_level, 
                             levels = c("EN", "MI", "SE", "EX"),
                             labels = c("Entry", "Mid", "Senior", "Executive"))

df$employment_type <- factor(df$employment_type, 
                            levels = c("FT", "PT", "CT"),
                            labels = c("Full-time", "Part-time", "Contract"))

df$company_size <- factor(df$company_size, 
                         levels = c("S", "M", "L"),
                         labels = c("Small", "Medium", "Large"))

# Convert remote_ratio to factor for better visualization
df$remote_ratio_factor <- factor(df$remote_ratio, 
                               levels = c(0, 50, 100),
                               labels = c("On-site", "Hybrid", "Remote"))

# Create income categories (low, middle, high)
# Using quantiles to divide into three equal groups
income_quantiles <- quantile(df$salary_in_usd, probs = c(0, 1/3, 2/3, 1))
df$income_category <- cut(df$salary_in_usd, 
                         breaks = income_quantiles,
                         labels = c("Low", "Middle", "High"),
                         include.lowest = TRUE)

# Print the income category thresholds
cat("Income category thresholds:\n")
print(income_quantiles)

# ---------------------------------------------------------
# 2. EXPLORATORY DATA ANALYSIS
# ---------------------------------------------------------

# Basic statistics of salary
salary_stats <- data.frame(
  Mean = mean(df$salary_in_usd),
  Median = median(df$salary_in_usd),
  Min = min(df$salary_in_usd),
  Max = max(df$salary_in_usd),
  SD = sd(df$salary_in_usd),
  Q1 = quantile(df$salary_in_usd, 0.25),
  Q3 = quantile(df$salary_in_usd, 0.75)
)

print("Salary statistics (USD):")
print(salary_stats)

# ---------------------------------------------------------
# 3. VISUALIZATION
# ---------------------------------------------------------

# Histogram of salaries
p1 <- ggplot(df, aes(x = salary_in_usd)) +
  geom_histogram(bins = 30, fill = "skyblue", color = "black") +
  labs(title = "Distribution of Salaries",
       x = "Salary (USD)",
       y = "Frequency") +
  theme_minimal()
print(p1)
ggsave("images/salary_distribution.png", p1, width = 8, height = 6)

# Boxplot of salaries by experience level
p2 <- ggplot(df, aes(x = experience_level, y = salary_in_usd, fill = experience_level)) +
  geom_boxplot() +
  labs(title = "Salary by Experience Level",
       x = "Experience Level",
       y = "Salary (USD)") +
  theme_minimal() +
  theme(legend.position = "none")
print(p2)
ggsave("images/salary_by_experience.png", p2, width = 8, height = 6)

# Boxplot of salaries by remote work ratio
p3 <- ggplot(df, aes(x = remote_ratio_factor, y = salary_in_usd, fill = remote_ratio_factor)) +
  geom_boxplot() +
  labs(title = "Salary by Remote Work Ratio",
       x = "Remote Work Ratio",
       y = "Salary (USD)") +
  theme_minimal() +
  theme(legend.position = "none")
print(p3)
ggsave("images/salary_by_remote_ratio.png", p3, width = 8, height = 6)

# Boxplot of salaries by company size
p4 <- ggplot(df, aes(x = company_size, y = salary_in_usd, fill = company_size)) +
  geom_boxplot() +
  labs(title = "Salary by Company Size",
       x = "Company Size",
       y = "Salary (USD)") +
  theme_minimal() +
  theme(legend.position = "none")
print(p4)
ggsave("images/salary_by_company_size.png", p4, width = 8, height = 6)

# Top 10 job titles by average salary
job_avg <- df %>%
  group_by(job_title) %>%
  summarise(avg_salary = mean(salary_in_usd),
            count = n()) %>%
  filter(count >= 5) %>%  # Only include job titles with at least 5 entries
  arrange(desc(avg_salary)) %>%
  head(10)

p5 <- ggplot(job_avg, aes(x = reorder(job_title, avg_salary), y = avg_salary)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Top 10 Highest Paying Job Titles",
       x = "Job Title",
       y = "Average Salary (USD)") +
  theme_minimal()
print(p5)
ggsave("images/top_10_job_titles.png", p5, width = 10, height = 6)

# Distribution of income categories
p6 <- ggplot(df, aes(x = income_category, fill = income_category)) +
  geom_bar() +
  labs(title = "Distribution of Income Categories",
       x = "Income Category",
       y = "Count") +
  theme_minimal() +
  theme(legend.position = "none")
print(p6)
ggsave("images/income_categories.png", p6, width = 8, height = 6)

# ---------------------------------------------------------
# 4. OUTLIER DETECTION
# ---------------------------------------------------------

# Function to detect outliers using IQR method
detect_outliers <- function(x) {
  q1 <- quantile(x, 0.25)
  q3 <- quantile(x, 0.75)
  iqr <- q3 - q1
  lower_bound <- q1 - 1.5 * iqr
  upper_bound <- q3 + 1.5 * iqr
  
  outliers <- x < lower_bound | x > upper_bound
  return(list(
    outliers = outliers,
    count = sum(outliers),
    percentage = mean(outliers) * 100,
    lower_bound = lower_bound,
    upper_bound = upper_bound
  ))
}

# Detect outliers in salary_in_usd
salary_outliers <- detect_outliers(df$salary_in_usd)

cat("Salary outliers:\n")
cat("Count:", salary_outliers$count, "\n")
cat("Percentage:", round(salary_outliers$percentage, 2), "%\n")
cat("Lower bound:", salary_outliers$lower_bound, "\n")
cat("Upper bound:", salary_outliers$upper_bound, "\n")

# Visualize outliers with a boxplot
p7 <- ggplot(df, aes(y = salary_in_usd)) +
  geom_boxplot(fill = "lightblue") +
  labs(title = "Boxplot of Salaries with Outliers",
       y = "Salary (USD)") +
  theme_minimal()
print(p7)
ggsave("images/salary_outliers_boxplot.png", p7, width = 8, height = 6)

# Create normalized version of salary for better visualization
df$salary_normalized <- scale(df$salary_in_usd)

# Histogram of normalized salaries
p8 <- ggplot(df, aes(x = salary_normalized)) +
  geom_histogram(bins = 30, fill = "lightgreen", color = "black") +
  labs(title = "Distribution of Normalized Salaries",
       x = "Normalized Salary (z-score)",
       y = "Frequency") +
  theme_minimal()
print(p8)
ggsave("images/normalized_salary_distribution.png", p8, width = 8, height = 6)

# ---------------------------------------------------------
# 5. CORRELATION ANALYSIS
# ---------------------------------------------------------

# Select numeric variables for correlation analysis
numeric_vars <- df %>% select(salary_in_usd, remote_ratio, work_year)

# Calculate correlation matrix
cor_matrix <- cor(numeric_vars, use = "complete.obs")
print("Correlation matrix:")
print(cor_matrix)

# Visualize correlation matrix
corrplot(cor_matrix, method = "circle", type = "upper", 
         tl.col = "black", tl.srt = 45)

# Save correlation plot
png("images/correlation_matrix.png", width = 800, height = 600)
corrplot(cor_matrix, method = "circle", type = "upper", 
         tl.col = "black", tl.srt = 45)
dev.off()

# ---------------------------------------------------------
# 6. MODELING
# ---------------------------------------------------------

# Prepare data for modeling
model_data <- df %>%
  select(salary_in_usd, experience_level, employment_type, 
         remote_ratio, company_size, work_year)

# Split data into training and testing sets (70% training, 30% testing)
set.seed(123)
train_index <- createDataPartition(model_data$salary_in_usd, p = 0.7, list = FALSE)
train_data <- model_data[train_index, ]
test_data <- model_data[-train_index, ]

# 6.1 Linear Regression Model
lm_model <- lm(salary_in_usd ~ experience_level + employment_type + 
                remote_ratio + company_size + work_year, data = train_data)
summary(lm_model)

# Predict on test data
lm_predictions <- predict(lm_model, test_data)

# Calculate RMSE for linear model
lm_rmse <- sqrt(mean((test_data$salary_in_usd - lm_predictions)^2, na.rm = TRUE))
cat("Linear Model RMSE:", lm_rmse, "\n")


# 6.2 Random Forest Model
# First, check for missing values in the training data
missing_train <- colSums(is.na(train_data))
cat("Missing values in training data:\n")
print(missing_train)

# Remove rows with missing values or impute them
train_data_clean <- na.omit(train_data)
test_data_clean <- na.omit(test_data)

# Now fit the Random Forest model on the clean data
rf_model <- randomForest(salary_in_usd ~ experience_level + employment_type + 
                         remote_ratio + company_size + work_year, 
                         data = train_data_clean, ntree = 100)
print(rf_model)

# Predict on clean test data
rf_predictions <- predict(rf_model, test_data_clean)

# Calculate RMSE for random forest model
rf_rmse <- sqrt(mean((test_data_clean$salary_in_usd - rf_predictions)^2))
cat("Random Forest Model RMSE:", rf_rmse, "\n")


# Variable importance
var_importance <- importance(rf_model)
print("Variable importance:")
print(var_importance)

# Plot variable importance
png("images/variable_importance.png", width = 800, height = 600)
varImpPlot(rf_model, main = "Variable Importance")
dev.off()

# 6.3 GAM Model for Salary Prediction
# Check unique values in smoothing variables
cat("Unique values in work_year:", length(unique(train_data$work_year)), "\n")
cat("Unique values in remote_ratio:", length(unique(train_data$remote_ratio)), "\n")

# Use clean data and specify lower k values for the smooth terms
gam_model <- gam(salary_in_usd ~ s(work_year, k=3) + s(remote_ratio, k=3) + 
                experience_level + employment_type + company_size, 
                data = train_data_clean)
summary(gam_model)

# Predict on test data
gam_predictions <- predict(gam_model, test_data_clean)

# Calculate RMSE for GAM model
gam_rmse <- sqrt(mean((test_data_clean$salary_in_usd - gam_predictions)^2))
cat("GAM Model RMSE:", gam_rmse, "\n")


# Plot GAM model smooths
png("images/gam_smooths.png", width = 1000, height = 800)
par(mfrow = c(2, 1))
plot(gam_model, select = 1, shade = TRUE, col = "blue", 
     main = "GAM: Effect of Work Year on Salary")
plot(gam_model, select = 2, shade = TRUE, col = "blue", 
     main = "GAM: Effect of Remote Ratio on Salary")
dev.off()

# 6.4 Multinomial Logistic Regression for Income Category
# Load required package
library(nnet)

# Prepare data for classification
class_data <- df %>%
  select(income_category, experience_level, employment_type, 
         remote_ratio, company_size, work_year)

# Clean the classification data
class_data_clean <- na.omit(class_data)

# Split data for classification
set.seed(123)
train_index_class <- createDataPartition(class_data_clean$income_category, p = 0.7, list = FALSE)
train_data_class <- class_data_clean[train_index_class, ]
test_data_class <- class_data_clean[-train_index_class, ]

# Train multinomial model
multinom_model <- multinom(income_category ~ experience_level + employment_type + 
                          remote_ratio + company_size + work_year, 
                          data = train_data_class)
summary(multinom_model)

# Predict on test data
multinom_predictions <- predict(multinom_model, test_data_class)

# Confusion matrix
conf_matrix <- confusionMatrix(multinom_predictions, test_data_class$income_category)
print("Confusion Matrix for Income Category Classification:")
print(conf_matrix)


# Save confusion matrix
png("images/confusion_matrix.png", width = 800, height = 600)
fourfoldplot(conf_matrix$table, color = c("#CC6666", "#99CC99", "#6666CC"),
             conf.level = 0, margin = 1, main = "Confusion Matrix")
dev.off()

# ---------------------------------------------------------
# 7. MODEL COMPARISON AND EVALUATION
# ---------------------------------------------------------

# Compare RMSE of different regression models
model_comparison <- data.frame(
  Model = c("Linear Regression", "Random Forest", "GAM"),
  RMSE = c(lm_rmse, rf_rmse, gam_rmse)
)

print("Model Comparison (RMSE):")
print(model_comparison)

# Plot model comparison
p9 <- ggplot(model_comparison, aes(x = Model, y = RMSE, fill = Model)) +
  geom_bar(stat = "identity") +
  labs(title = "Model Comparison by RMSE",
       x = "Model",
       y = "RMSE (Lower is Better)") +
  theme_minimal() +
  theme(legend.position = "none")
print(p9)
ggsave("images/model_comparison.png", p9, width = 8, height = 6)

# ---------------------------------------------------------
# 8. CONCLUSION
# ---------------------------------------------------------

cat("\n=== ANALYSIS SUMMARY ===\n")
cat("Dataset contains", nrow(df), "salary records\n")
cat("Salary range: $", min(df$salary_in_usd), "to $", max(df$salary_in_usd), "\n")
cat("Income categories: Low (<$", income_quantiles[2], 
    "), Middle ($", income_quantiles[2], "-$", income_quantiles[3], 
    "), High (>$", income_quantiles[3], ")\n", sep="")
cat("Best performing model: ", model_comparison$Model[which.min(model_comparison$RMSE)], 
    " (RMSE = ", min(model_comparison$RMSE), ")\n", sep="")
cat("Outliers detected: ", salary_outliers$count, " (", 
    round(salary_outliers$percentage, 2), "%)\n", sep="")
cat("Most important salary predictor: ", 
    rownames(var_importance)[which.max(var_importance)], "\n")
cat("Classification accuracy: ", round(conf_matrix$overall["Accuracy"] * 100, 2), "%\n", sep="")