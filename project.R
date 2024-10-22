read.csv("C:/Users/Admin/Documents/employeedata/Employee.csv")
Employees<-read.csv("C:/Users/Admin/Documents/employeedata/Employee.csv")
library(tidyverse)
library(modelr)
library(caret)
library(rsample)
library(corrplot)
library(plotly)
library(gridExtra)
library(randomForest)
library(xgboost)



  
Employees$Education <- factor(Employees$Education)
Employees$City <- factor(Employees$City)
Employees$Gender <- factor(Employees$Gender)
Employees$LeaveOrNot <- factor(Employees$LeaveOrNot)
Employees$PaymentTier <- factor(Employees$PaymentTier)
Employees$ExperienceInCurrentDomain <- factor(Employees$ExperienceInCurrentDomain)


ggplot(Employees, aes(x = City, fill = PaymentTier)) +
  geom_bar(position = "dodge") +
  theme_minimal() +
  labs(title = "Distribution of Employees by City and Payment Tier")

ggplot(Employees, aes(x = Age, fill = LeaveOrNot)) +
  geom_histogram(bins = 20, alpha = 0.7) +
  theme_minimal() +
  labs(title = "Age Distribution by Leave or Not")




simple_lm <- lm(as.numeric(LeaveOrNot) ~ Age, data = Employees)
summary(simple_lm)


ggplot(Employees, aes(x = Age, y = as.numeric(LeaveOrNot))) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "lm", col = "blue") +
  theme_minimal() +
  labs(title = "Simple Linear Regression: Age vs LeaveOrNot")




multiple_lm <- lm(as.numeric(LeaveOrNot) ~ Age + Education + Gender, data = Employees)
summary(multiple_lm)



predictions <- Employees
predictions$predicted_values <- predict(multiple_lm, newdata = Employees)



ggplot(predictions, aes(x = Age, y = predicted_values, color = Education, linetype = Gender)) +
  geom_point(alpha = 0.3) +
  geom_smooth(method = "lm", se = FALSE) +
  theme_minimal() +
  labs(title = "Predicted LeaveOrNot by Age, Education, and Gender", x = "Age", y = "Predicted LeaveOrNot")


logistic_model <- glm(LeaveOrNot ~ Age + Education + Gender, data = Employees, family = binomial)


summary(logistic_model)



new_data <- expand.grid(
  Age = seq(min(Employees$Age), max(Employees$Age), length.out = 100),
  Education = levels(Employees$Education),
  Gender = levels(Employees$Gender)
)

new_data$predicted_probabilities <- predict(logistic_model, newdata = new_data, type = "response")

ggplot(new_data, aes(x = Age, y = predicted_probabilities, color = Education, linetype = Gender)) +
  geom_line() +
  theme_minimal() +
  labs(title = "Predicted Probability of Leaving by Age, Education, and Gender",
       x = "Age", y = "Predicted Probability of Leaving") +
  scale_color_manual(values = c("Bachelors" = "blue", "Masters" = "green", "PHD" = "red")) +
  scale_linetype_manual(values = c("Female" = "solid", "Male" = "dashed"))


set.seed(123)
train_index <- createDataPartition(Employees$LeaveOrNot, p = 0.7, list = FALSE)
train_data <- Employees[train_index, ]
test_data <- Employees[-train_index, ]


set.seed(123)
rf_model <- randomForest(LeaveOrNot ~ ., data = train_data, importance = TRUE)


rf_predictions <- predict(rf_model, test_data)

confusionMatrix(rf_predictions, test_data$LeaveOrNot)

varImpPlot(rf_model)

summary(rf_model)

evaluate_feature_importance <- function(feature_name, data, target) {
 
  reduced_data <- data[, !names(data) %in% feature_name]
  
  set.seed(123)
  train_index <- createDataPartition(data[[target]], p = 0.7, list = FALSE)
  train_data <- reduced_data[train_index, ]
  test_data <- reduced_data[-train_index, ]
  
 
  rf_model_reduced <- randomForest(as.formula(paste(target, "~ .")), data = train_data, importance = TRUE)
  

  rf_predictions_reduced <- predict(rf_model_reduced, test_data)
  

  accuracy_reduced <- confusionMatrix(rf_predictions_reduced, test_data[[target]])$overall['Accuracy']
  
  return(accuracy_reduced)
}

set.seed(123)
train_index <- createDataPartition(Employees$LeaveOrNot, p = 0.7, list = FALSE)
train_data <- Employees[train_index, ]
test_data <- Employees[-train_index, ]
rf_model <- randomForest(LeaveOrNot ~ ., data = train_data, importance = TRUE)
rf_predictions <- predict(rf_model, test_data)
original_accuracy <- confusionMatrix(rf_predictions, test_data$LeaveOrNot)$overall['Accuracy']

accuracy_without_joiningyear <- evaluate_feature_importance("JoiningYear", Employees, "LeaveOrNot")


decrease_in_accuracy <- original_accuracy - accuracy_without_joiningyear
print(decrease_in_accuracy)








employee_data$LeaveOrNot <- as.numeric(as.character(employee_data$LeaveOrNot)) +1

employee_data$LeaveOrNot <- ifelse(employee_data$LeaveOrNot == 1, 0, 1)

dummies <- dummyVars(LeaveOrNot ~ ., data = employee_data)
employee_data_transformed <- predict(dummies, newdata = employee_data)
employee_data_transformed <- as.data.frame(employee_data_transformed)


employee_data_transformed$LeaveOrNot <- employee_data$LeaveOrNot


set.seed(123)
train_index <- createDataPartition(employee_data_transformed$LeaveOrNot, p = 0.7, list = FALSE)
train_data <- employee_data_transformed[train_index, ]
test_data <- employee_data_transformed[-train_index, ]

train_matrix <- xgb.DMatrix(data = as.matrix(train_data[, -which(names(train_data) == "LeaveOrNot")]), label = train_data$LeaveOrNot)
test_matrix <- xgb.DMatrix(data = as.matrix(test_data[, -which(names(test_data) == "LeaveOrNot")]), label = test_data$LeaveOrNot)


params <- list(
  booster = "gbtree",
  objective = "binary:logistic",
  eta = 0.1, Q
  max_depth = 6,
  min_child_weight = 1,
  subsample = 0.8,
  colsample_bytree = 0.8,
  eval_metric = "error"
)


set.seed(123)
xgb_model <- xgb.train(
  params = params,
  data = train_matrix,
  nrounds = 100,
  watchlist = list(train = train_matrix, eval = test_matrix),
  early_stopping_rounds = 10,
  print_every_n = 10
)


xgb_predictions <- predict(xgb_model, newdata = test_matrix)
xgb_predictions <- ifelse(xgb_predictions > 0.5, 1, 0)


conf_matrix <- confusionMatrix(as.factor(xgb_predictions), as.factor(test_data$LeaveOrNot))

conf_matrix_data <- as.data.frame(conf_matrix$table)

importance_matrix <- xgb.importance(model = xgb_model)


print(importance_matrix)


xgb.plot.importance(importance_matrix, top_n = 10, measure = "Gain")

importance_df <- as.data.frame(importance_matrix)


ggplot(importance_df, aes(x = reorder(Feature, Gain), y = Gain)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "Feature Importance (Gain)", x = "Feature", y = "Importance (Gain)") +
  theme_minimal()