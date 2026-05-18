# Assignment STAT606 #

# Load library(s)
library(dplyr) # for data manipulation and preprocessing
library(caTools) # for splitting data into train and test sets
library(caret) # for performance metric functions
library(ggplot2) # for visualisation
library(pROC)

library(h2o)
library(ROSE)

# -----------------------------------------------------------------------------#
# Set user-specified parameters --
#------------------------------------------------------------------------------#
seed = 606
train_distribution <- 0.7 # 70% of data goes to training and 30% into test set
metric <- "Recall" # Used to maximise detection of intentional shopper
folds <- 5


# Make the output easier to understand
options(scipen = 999)


# -----------------------------------------------------------------------------#
# Import and Explore the Data
# -----------------------------------------------------------------------------#

# Import the Online Shoppers dataset into R
library(readxl)
Online_Shoppers_Purchasing_Intention_dataset <- read_excel("C:/Users/qndlovu_DS/OneDrive - University of KwaZulu-Natal/Desktop/PGDip Data Science UKZN/STAT606-Binary Classification and Matching/Assignment 1/Online Shoppers Purchasing Intention dataset.xlsx")

# Ensure the imported data is a data frame
Online_Shoppers_Purchasing_Intention_dataset <- data.frame(Online_Shoppers_Purchasing_Intention_dataset)

View(Online_Shoppers_Purchasing_Intention_dataset)

# convert target to a factor
Online_Shoppers_Purchasing_Intention_dataset$Revenue <- as.factor(Online_Shoppers_Purchasing_Intention_dataset$Revenue)


# Convert other type(e.g logical) variables to a factor
Online_Shoppers_Purchasing_Intention_dataset$Month <- as.factor(Online_Shoppers_Purchasing_Intention_dataset$Month)
Online_Shoppers_Purchasing_Intention_dataset$VisitorType <- as.factor(Online_Shoppers_Purchasing_Intention_dataset$VisitorType)
Online_Shoppers_Purchasing_Intention_dataset$Weekend <- as.factor(Online_Shoppers_Purchasing_Intention_dataset$Weekend)
Online_Shoppers_Purchasing_Intention_dataset$Browser <- as.factor(Online_Shoppers_Purchasing_Intention_dataset$Browser)
Online_Shoppers_Purchasing_Intention_dataset$Region <- as.factor(Online_Shoppers_Purchasing_Intention_dataset$Region)
Online_Shoppers_Purchasing_Intention_dataset$TrafficType <- as.factor(Online_Shoppers_Purchasing_Intention_dataset$TrafficType)
Online_Shoppers_Purchasing_Intention_dataset$OperatingSystems <- as.factor(Online_Shoppers_Purchasing_Intention_dataset$OperatingSystems)

# view properties of dataset
summary(Online_Shoppers_Purchasing_Intention_dataset)

# Inspect the variable BounceRates - suspect strings and scientific notations
View(Online_Shoppers_Purchasing_Intention_dataset$BounceRates)
Online_Shoppers_Purchasing_Intention_dataset$BounceRates <- as.numeric(Online_Shoppers_Purchasing_Intention_dataset$BounceRates)

# check cardinality for all factor variables
sapply(Filter(is.factor, Online_Shoppers_Purchasing_Intention_dataset), nlevels)

# Summary: 
# Likely candidates to simplify later: TrafficType, Browser/Operating System, Region due to high cardinality

# check for missing values
colSums(is.na(Online_Shoppers_Purchasing_Intention_dataset))

# -----------------------------------------------------------------------------#
#         Exploratory Data Analysis
# -----------------------------------------------------------------------------#


# Target(Revenue) distribution
table(Online_Shoppers_Purchasing_Intention_dataset$Revenue) # --- FALSE=10422, TRUE = 1908, Imbalanced class

# Check how behavior differs between purchasers and non-purchasers

# Check distribution of pageValues
boxplot(Online_Shoppers_Purchasing_Intention_dataset$PageValues) # results: a few users contribute very large revenue signals, most of users generate no page value(median = 0)

#how are bounce rates distributed?
boxplot(Online_Shoppers_Purchasing_Intention_dataset$BounceRates)

# Pagevalue relationship with Revenue
ggplot(Online_Shoppers_Purchasing_Intention_dataset,aes(x=Revenue, y=PageValues, fill=Revenue)) +
  geom_boxplot()  # results: Users with higher PageValues are much more likely to purchase

# BounceRates relationship with Revenue
ggplot(Online_Shoppers_Purchasing_Intention_dataset,aes(x=Revenue, y=BounceRates, fill=Revenue)) +
  geom_boxplot()  # results: users with low bounce rates are more likely to purchase

#VisitorType relationship with Revenue
table(Online_Shoppers_Purchasing_Intention_dataset$VisitorType,
      Online_Shoppers_Purchasing_Intention_dataset$Revenue) 
#------results/intepretation: 
# ----example (a) New Visitors Total = 1272 + 422 = 1694, therefore purchase rate = 422/1694 = 24.9%
#             (b) Returning Visitor purchase rate = 13.9% and Other= 18.8%
# Conclusion: New visitors have a higher purchase rate than returning visitors

# add other variables likeExitRates, Product, month, TrafficType

# Correlation for numeric variables
cor(Online_Shoppers_Purchasing_Intention_dataset[, sapply(Online_Shoppers_Purchasing_Intention_dataset, is.numeric)])


# Use new variables for whole data and target
data <- Online_Shoppers_Purchasing_Intention_dataset
target <- "Revenue"


# ---------------------------------------------------------------------------- #
#                           Train & Split step
# ----------------------------------------------------------------------------#

# Convert the True and False of target to 0 and 1:
table(data$Revenue)

# forlogistic regression, for DT just use TRUE FALSE
# data$Revenue <- as.integer(data$Revenue)


set.seed(seed)

#it looks at how data is distributed in target, then the splitting will follow the same structure(0:70, 1:30), that's why we pass "target" into the below function.
split = sample.split(data[[target]], SplitRatio = train_distribution)

train_set =  subset(data,split==TRUE)
test_set =  subset(data,split==FALSE)


# -----------------------------------------------------------------------------#
#          Feature Engineering on Train set 
# -----------------------------------------------------------------------------#

train_set$Total_timeSpent <- train_set$Administrative_Duration + train_set$Informational_Duration + train_set$ProductRelated_Duration
train_set$Engagement_Rate <- train_set$ProductRelated / (train_set$ProductRelated_Duration + 1) # +1 avoids division by zero

# Apply the same transformation to test set
test_set$Total_timeSpent <- test_set$Administrative_Duration + test_set$Informational_Duration + test_set$ProductRelated_Duration
test_set$Engagement_Rate <- test_set$ProductRelated / (test_set$ProductRelated_Duration + 1)


# H2O gives error when predicting using test set(factor level mismatch), fix factor levels(for test set) to avoid this.
test_set$Revenue <- factor(test_set$Revenue,
                           levels = levels(train_set$Revenue))

test_set$Month <- factor(test_set$Month,
                         levels = levels(train_set$Month))

test_set$VisitorType <- factor(test_set$VisitorType,
                               levels = levels(train_set$VisitorType))

test_set$Weekend <- factor(test_set$Weekend,
                           levels = levels(train_set$Weekend))

test_set$Browser <- factor(
  test_set$Browser,
  levels = unique(train_set$Browser) 
                            # this is exact variable with warnings
)

test_set$Region <- factor(test_set$Region,
                          levels = levels(train_set$Region))

test_set$TrafficType <- factor(
  test_set$TrafficType,
  levels = unique(train_set$TrafficType)  # this is exact variable with warnings
)

test_set$OperatingSystems <- factor(test_set$OperatingSystems,
                                    levels = levels(train_set$OperatingSystems))


###### Balancing Using Over-Sampling on the training set, the minority class is not tiny

# Count the majority class observations (FALSE)
total_over <- nrow(train_set[train_set[[target]] == "FALSE",])

train_over <- ovun.sample(
  as.formula(paste(target, "~ .")),
  data = train_set,
  method = "over",
  N = 2 * total_over, # multiply by 2 for the two classes
  seed = seed
)

# Extract and save the resulting under-sampled data:
train_over_data <- train_over$data
summary(train_over_data[[target]])



#initialise h2o
h2o.init()


# Convert train and test sets to H2o frame
train_h2o_lr <- as.h2o(train_over_data)
test_h2o_lr <- as.h2o(test_set)

# Define predictors and target variable(done earlier)
y <- target
x <- setdiff(names(train_h2o_lr), y)

# No hyperparameter tuning required!!

# ----------------------------- ----------------------------------------------#
############# ---------------Fit a decision tree using h20 -------------------#

# Fit a model
dt_model <- h2o.gbm(
  x = x, # predictors
  y = y,
  training_frame = train_h2o,
  ntrees = 1,
  max_depth = 5,
  #minsplit # not required by h20, only rpart
  learn_rate = 1.0,
  seed = seed
)

# Make predictions, and save them
preds_dt_train <- h2o.predict(dt_model, train_h2o)
preds_dt_test <- h2o.predict(dt_model, test_h20)

# Convert predictions back to R data.frames 
preds_dt_train <- as.data.frame(preds_dt_train)
preds_dt_test <- as.data.frame(preds_dt_test)

# Append column 3 (predicted probabilities for class label = 1) to original training and test sets
train_dt_pred <- cbind(train_set,
                       setNames(preds_dt_train[, 3, drop = FALSE], "pred_prob")) 

test_dt_pred <- cbind(test_set, 
                      setNames(preds_dt_test[, 3, drop = FALSE], "pred_prob"))

#####  -----------------DT Model evaluation  ----------------------------------#

#specify the threshold
threshold <- 0.5

# Determine the predicted class labels:

train_dt_pred$pred_class <- factor(
  ifelse(train_dt_pred$pred_prob > threshold, "TRUE", "FALSE"),
  levels = levels(train_dt_pred[[target]])
) # training set


test_dt_pred$pred_class <- factor(
  ifelse(test_dt_pred$pred_prob > threshold, "TRUE", "FALSE"),
  levels = levels(test_dt_pred[[target]])
) # test set 

# Confusion Matrix for training set
confusionMatrix(
  train_dt_pred$pred_class,
  train_dt_pred[[target]],
  positive = "TRUE",
  mode = "everything"
)

# actual classes first then predicted probabilities for the training set
roc_dt_train <- roc(train_dt_pred[[target]], train_dt_pred$pred_prob)
auc(roc_dt_train)
plot(roc_dt_train)

# predicted classes first then actual classes for the test set
confusionMatrix(
  test_dt_pred$pred_class,
  test_dt_pred[[target]],
  positive = "TRUE",
  mode = "everything"
)

# actual classes first then predicted probabilities for the test set
roc_dt_test <- roc(test_dt_pred[[target]], test_dt_pred$pred_prob)
auc(roc_dt_test)
plot(roc_dt_test)



######## Fit a Logistic Regression(& tune the threshold <- 0.3) ---------------#

# Check correlation first
h2o.cor(train_h2o_lr)

Log_Regression <- h2o.glm(
  x = x,
  y = y,
  training_frame = train_h2o_lr,
  family = "binomial", 
  compute_p_values = TRUE, 
  remove_collinear_columns = TRUE
)

# Extract p-values
Log_regression_results <- Log_Regression@model[["coefficients_table"]]

# create odds ratios from the regression coefficient estimates
Log_regression_results$OddsRatio <- exp(Log_regression_results[,2])

# round the p-values off to 5 decimal places
Log_regression_results$p_value <- round(Log_regression_results$p_value,5)
Log_regression_results$OddsRatio <- round(Log_regression_results$OddsRatio,5)

# Make predictions
predictions_LR_train <- h2o.predict(Log_Regression, train_h2o_lr)
predictions_LR_test <- h2o.predict(Log_Regression, test_h2o_lr) ##=== factor level issue here !!!

# Convert predictions to R data.frames to extract from H2O environment:
predictions_LR_train <- as.data.frame(predictions_LR_train)
predictions_LR_test <- as.data.frame(predictions_LR_test)

# View the prediction output
View(predictions_LR_train)

# Take True probability column and combine with original data set
train_LogR_pred <- cbind(
  train_over_data,
  predicted_prob = predictions_LR_train[,3]
)

test_LogR_pred <- cbind(
  test_set,
  predicted_prob = predictions_LR_test[,3]
)


## ---------------------------Set the threshold ---

# Due to imbalance, adjust threshold from 0.5 to 0.3 to improve detection of positive purchasers
threshold <- 0.3

# Determine predicted class labels

train_LogR_pred$pred_class <- factor(
  ifelse(train_LogR_pred$predicted_prob > threshold,"TRUE","FALSE"), levels = levels(train_LogR_pred[[target]]))

test_LogR_pred$pred_class <- factor(
  ifelse(test_LogR_pred$predicted_prob > threshold, "TRUE", "FALSE"), levels = levels(test_LogR_pred[[target]]))


# ---------- Model Evaluation  -----------------------------------------------

# ROC + AUC( LR Model Overall)
roc_LR_train <- roc(train_LogR_pred[[target]], train_LogR_pred$predicted_prob)
auc(roc_LR_train)
plot(roc_LR_train)

# For Test.
roc_LR_test <- roc(test_LogR_pred[[target]], test_LogR_pred$predicted_prob)
auc(roc_LR_test)
plot(roc_LR_test)

# For training, confusion matrix
confusionMatrix(
  train_LogR_pred$pred_class,
  train_LogR_pred$Revenue,
  positive = "TRUE",
  mode = "everything"
)

# test confusion matrix
confusionMatrix(
  test_LogR_pred$pred_class,
  test_LogR_pred[[target]],
  positive = "TRUE",
  mode = "everything"
)


###### Fit Naive Bayes Classifier  ######   





























