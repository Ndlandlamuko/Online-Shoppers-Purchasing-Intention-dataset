#############################################################################
# : Naive Bayes, Decision trees and Logistic regression  #
#############################################################################

# ----------------------------------------------------------------------------- #
# 0. Load Libraries -----------------------------------------------------------
# ----------------------------------------------------------------------------- #

library(dplyr)    # for data manipulation and preprocessing
library(caTools)  # for splitting into train and test sets
library(caret)    # used for performance metric functions
library(pROC)     # used for obtaining AUC
library(h2o)
library(readxl)   # package used to import Excel files

# for fitting  DT in rprart
library(rpart)
library(rpart.plot)

# this is to turn scientific notation off so the output is easier to interpret:
options(scipen = 999) # turn back on by changing to 0

# ----------------------------------------------------------------------------- #
# 1. Setup & User Parameters --------------------------------------------------
# ----------------------------------------------------------------------------- #

# User-specified parameters
seed = 606     # Seed for reproducibility (students can change)
train_frac <- 0.7  # Proportion of data in training set
metric <- "F1" # "auc", "aucpr" (Area Under Precision-Recall Curve), "logloss", "Accuracy", "Specificity", "Precision", "Recall", "F1"
folds <- 5 # for 5-fold CV, or change to 10


# ----------------------------------------------------------------------------- #
# 2. Load, Inspect and Format Data ------------------------------------------------------
# ----------------------------------------------------------------------------- #

# recall: data from different sources/files can be imported into R. We will use data on online shoppers purchasing intention in an Excel file (download from Moodle and ensure it is in the same folder as this project)

SHOPPERS <- read_excel("datasets/Online Shoppers Purchasing Intention dataset.xlsx")

SHOPPERS <- data.frame(SHOPPERS)

# Excel can read numeric columns as character if any cell is text-formatted; explicitly
# coerce the originally-numeric columns here so they are never converted to factor by
# the mutate(across(where(is.character), as.factor)) call below (which would create
# float-string factor levels that break h2o.decision_tree's LabelEncoder).
SHOPPERS <- SHOPPERS %>%
  mutate(across(c(Administrative, Administrative_Duration,
                  Informational, Informational_Duration,
                  ProductRelated, ProductRelated_Duration,
                  BounceRates, ExitRates, PageValues, SpecialDay,
                  OperatingSystems, Browser, Region, TrafficType), as.numeric))

# look at the properties of the data
summary(SHOPPERS)

# ──  Missing values 
cat("Missing values per column:\n")
print(colSums(is.na(SHOPPERS)))

#  Remove exact duplicate rows 
before <- nrow(SHOPPERS)
SHOPPERS <- distinct(SHOPPERS)
cat(sprintf("\nDropped %d duplicate rows. Remaining: %d\n", before - nrow(SHOPPERS), nrow(SHOPPERS)))

#  Encode Month as an ordered integer so models can learn seasonality 
month_order <- c(Feb=2, Mar=3, May=5, June=6, Jul=7, Aug=8, Sep=9, Oct=10, Nov=11, Dec=12)
SHOPPERS$Month_Num <- as.integer(month_order[as.character(SHOPPERS$Month)])

#  Encode VisitorType as an integer (New=0, Returning=1, Other=2) 
visitor_map <- c(New_Visitor=0, Returning_Visitor=1, Other=2)
SHOPPERS$VisitorType_Enc <- as.integer(visitor_map[as.character(SHOPPERS$VisitorType)])

#  Convert boolean flags to integer for compatibility 
SHOPPERS$Revenue_Int <- as.integer(SHOPPERS$Revenue)
SHOPPERS$Weekend_Int <- as.integer(SHOPPERS$Weekend)

cat("\nClass distribution (Revenue):\n")
print(table(SHOPPERS$Revenue_Int))
cat(sprintf("\nDataset shape after cleaning: %d rows x %d columns\n", nrow(SHOPPERS), ncol(SHOPPERS)))

# We need to structure all categorical variables into factors (instead of characters which is default when a flat file/CSV/Excel/text file is imported):

# This next step tells R to find all the character variables and change them to factors:
SHOPPERS <- SHOPPERS %>%
  mutate(across(where(is.character), as.factor))

# some other categorical variables have been treated as numeric, let's convert them to factor as well:
SHOPPERS$OperatingSystems <- factor(SHOPPERS$OperatingSystems)
SHOPPERS$Browser          <- factor(SHOPPERS$Browser)
SHOPPERS$Region           <- factor(SHOPPERS$Region)
SHOPPERS$TrafficType      <- factor(SHOPPERS$TrafficType)
SHOPPERS$Revenue          <- factor(SHOPPERS$Revenue_Int) # this is actually the target
SHOPPERS$Weekend          <- factor(SHOPPERS$Weekend_Int)

# check new factors variables again for sparsity and high cardinality:
summary(SHOPPERS)

# use the following to determine the number of levels (cardinality) of the factor variables:
sapply(Filter(is.factor, SHOPPERS), nlevels)

# The target ('Revenue') is somewhat imbalanced - note the class labels are 0 and 1:
summary(SHOPPERS$Revenue)


# ----------------------------------------------------------------------------- #
# 2a. Feature Engineering  ----------------------------------------------------
# ----------------------------------------------------------------------------- #

# The features below are grouped by the concept they capture:
# 1. Session breadth      - how many pages and how much time overall
# 2. Product focus        - how concentrated the visit was on product pages (avg duration per admin, info, and product page; focus ratios)
# 3. Engagement quality   - combining page value with bounce behaviour
# 4. Temporal signals     - seasonality and proximity to special shopping days
# 5. Visitor intent flags - binary indicators derived from visitor type
# 6. Log-transformed skewed columns - reduces right-skew for all duration columns and PageValues
# 7. Revenue flags        - combined Revenue + context signals (weekend, visitor type, session length)
# 8. Categorical device / traffic signals - dominant-category flags for TrafficType, Browser, OperatingSystems, and Region

# # ----------------------------------------------------------------------------- #
# 1. SESSION BREADTH
## ----------------------------------------------------------------------------- #

# Total distinct pages visited across all three page categories
SHOPPERS$Total_Pages <- SHOPPERS$Administrative + SHOPPERS$Informational + SHOPPERS$ProductRelated

# Total seconds spent on the site during the session
SHOPPERS$Total_Duration <- SHOPPERS$Administrative_Duration + SHOPPERS$Informational_Duration + SHOPPERS$ProductRelated_Duration

# Average seconds per page visited; zero-guards against sessions with no pages
SHOPPERS$Avg_Duration_Per_Page <- ifelse(SHOPPERS$Total_Pages > 0, SHOPPERS$Total_Duration / SHOPPERS$Total_Pages, 0)

# ══════════════════════════════════════════════════════════════════
# 2. PRODUCT FOCUS
# ══════════════════════════════════════════════════════════════════

# Average time per administrative page (symmetric with product & info equivalents below)
SHOPPERS$Avg_Administrative_Duration <- ifelse(SHOPPERS$Administrative > 0, SHOPPERS$Administrative_Duration / SHOPPERS$Administrative, 0)

# Average time per informational page
SHOPPERS$Avg_Informational_Duration <- ifelse(SHOPPERS$Informational > 0, SHOPPERS$Informational_Duration / SHOPPERS$Informational, 0)

# Average time spent per individual product page.
# High values suggest the user is carefully evaluating items -> purchase intent.
SHOPPERS$Avg_ProductRelated_Duration <- ifelse(SHOPPERS$ProductRelated > 0, SHOPPERS$ProductRelated_Duration / SHOPPERS$ProductRelated, 0)

# Fraction of total session time spent specifically on product pages (0-1).
# A ratio near 1 means the visitor skipped admin/info and went straight to products.
SHOPPERS$Product_Focus_Ratio <- ifelse(SHOPPERS$Total_Duration > 0, SHOPPERS$ProductRelated_Duration / SHOPPERS$Total_Duration, 0)

# Number of non-product pages; high count may indicate browsing vs. buying intent
SHOPPERS$Non_Product_Pages <- SHOPPERS$Administrative + SHOPPERS$Informational

# Share of pages that were non-product (admin + informational)
SHOPPERS$Non_Product_Ratio <- ifelse(SHOPPERS$Total_Pages > 0, SHOPPERS$Non_Product_Pages / SHOPPERS$Total_Pages, 0)

# ══════════════════════════════════════════════════════════════════
# 3. ENGAGEMENT QUALITY
# ══════════════════════════════════════════════════════════════════

# Engagement Score = PageValues x (1 - BounceRate).
# PageValues measures monetary contribution of visited pages; multiplying by
# (1 - BounceRate) penalises sessions where users immediately left.
SHOPPERS$Engagement_Score <- SHOPPERS$PageValues * (1 - SHOPPERS$BounceRates)

# ExitRate is always >= BounceRate. A small gap means most exits happen on the
# entry page (bad UX). A large gap means users browse multiple pages before leaving.
SHOPPERS$Exit_Minus_Bounce <- SHOPPERS$ExitRates - SHOPPERS$BounceRates

# Sessions where PageValues > 0 visited at least one page that contributed to a
# recorded transaction - this is one of the strongest single predictors of Revenue.
SHOPPERS$Has_PageValue <- as.integer(SHOPPERS$PageValues > 0)

# ══════════════════════════════════════════════════════════════════
# 4. TEMPORAL SIGNALS
# ══════════════════════════════════════════════════════════════════

# Season bucket derived from Month_Num (created during preprocessing)
month_to_season <- function(m) {
  ifelse(m %in% c(12, 1, 2), "Winter",
    ifelse(m %in% c(3, 4, 5), "Spring",
      ifelse(m %in% c(6, 7, 8), "Summer", "Autumn")))
}
SHOPPERS$Season <- factor(month_to_season(SHOPPERS$Month_Num))

# Q4 flag (November-December): holiday shopping peak -> typically highest conversion
SHOPPERS$Is_Q4 <- as.integer(SHOPPERS$Month_Num %in% c(11, 12))

# SpecialDay = 0 means no proximity to a holiday sale event.
# Any positive value means the visit was close to a promotional date (e.g. Valentine's).
SHOPPERS$Near_Special_Day <- as.integer(SHOPPERS$SpecialDay > 0)

# ══════════════════════════════════════════════════════════════════
# 5. VISITOR INTENT FLAGS
# ══════════════════════════════════════════════════════════════════

# One-hot style flags for visitor type - new visitors and returning visitors
# exhibit very different conversion patterns and need separate signals.
SHOPPERS$Is_New_Visitor       <- as.integer(SHOPPERS$VisitorType == "New_Visitor")
SHOPPERS$Is_Returning_Visitor <- as.integer(SHOPPERS$VisitorType == "Returning_Visitor")

# High-intent composite flag: visited product pages AND has meaningful page value
# AND did not immediately bounce -> likely to convert
SHOPPERS$High_Intent <- as.integer(
  SHOPPERS$ProductRelated > 0 &
  SHOPPERS$PageValues > 0 &
  SHOPPERS$BounceRates < 0.05
)

# ══════════════════════════════════════════════════════════════════
# 6. LOG-TRANSFORMED SKEWED COLUMNS
# ══════════════════════════════════════════════════════════════════

# All duration columns and PageValues are heavily right-skewed.
# log1p(x) = log(1+x) compresses the long tail and helps linear models
# and distance-based algorithms (KNN, SVM) converge better.
SHOPPERS$Log_Administrative_Duration <- log1p(SHOPPERS$Administrative_Duration)
SHOPPERS$Log_Informational_Duration  <- log1p(SHOPPERS$Informational_Duration)
SHOPPERS$Log_ProductRelated_Duration <- log1p(SHOPPERS$ProductRelated_Duration)
SHOPPERS$Log_Total_Duration          <- log1p(SHOPPERS$Total_Duration)
SHOPPERS$Log_PageValues              <- log1p(SHOPPERS$PageValues)

# ══════════════════════════════════════════════════════════════════
# 7. REVENUE FLAGS
# ══════════════════════════════════════════════════════════════════

# True only when the session converted AND occurred on a weekend
SHOPPERS$Weekend_Revenue <- as.integer(SHOPPERS$Revenue_Int == 1 & SHOPPERS$Weekend_Int == 1)

# True only when the session converted AND the visitor was not a returning visitor
SHOPPERS$Non_Returning_Visitor_Revenue <- as.integer(SHOPPERS$Revenue_Int == 1 & as.character(SHOPPERS$VisitorType) != "Returning_Visitor")

# True only when the session converted AND product-related browsing was under 2 hours (7200 s)
SHOPPERS$Revenue_Under_2_Hours_ProductRelated <- as.integer(SHOPPERS$Revenue_Int == 1 & SHOPPERS$ProductRelated_Duration <= 7200)

# ══════════════════════════════════════════════════════════════════
# 8. CATEGORICAL DEVICE / TRAFFIC SIGNALS
# ══════════════════════════════════════════════════════════════════

# Flag the dominant traffic acquisition channel (most frequent TrafficType)
top_traffic <- names(which.max(table(SHOPPERS$TrafficType)))
SHOPPERS$Is_Top_TrafficType <- as.integer(as.character(SHOPPERS$TrafficType) == top_traffic)

# Flag the dominant browser (largest user base in this dataset)
top_browser <- names(which.max(table(SHOPPERS$Browser)))
SHOPPERS$Is_Top_Browser <- as.integer(as.character(SHOPPERS$Browser) == top_browser)

# Flag the dominant operating system
top_os <- names(which.max(table(SHOPPERS$OperatingSystems)))
SHOPPERS$Is_Top_OS <- as.integer(as.character(SHOPPERS$OperatingSystems) == top_os)

# Flag the dominant geographic region
top_region <- names(which.max(table(SHOPPERS$Region)))
SHOPPERS$Is_Top_Region <- as.integer(as.character(SHOPPERS$Region) == top_region)

# check again:
summary(SHOPPERS)

# drop columns that are not needed for modelling (equivalent to PyCaret's ignore_features):
# Revenue_Int and Weekend_Int are helper encoding columns already represented by their factor versions;
# Revenue flags (Weekend_Revenue, Non_Returning_Visitor_Revenue, Revenue_Under_2_Hours_ProductRelated)
# are derived from the target and would cause data leakage if used as predictors.

SHOPPERS <- SHOPPERS %>%
  dplyr::select(
    -Revenue_Int,
    -Weekend_Int,
    -Weekend_Revenue,
    -Non_Returning_Visitor_Revenue,
    -Revenue_Under_2_Hours_ProductRelated
  )

# check again:
summary(SHOPPERS)

# The target ('Revenue') does not suffer from severe class imbalance:

summary(SHOPPERS$Revenue) # take note the class labels are 0 and 1

# We are now ready to split the data into the training and test sets.

# ----------------------------------------------------------------------------- #
# 3. Specify df and target ---------------------------------------------------
# ----------------------------------------------------------------------------- #

# Instead of writing code that only works for one dataset, we define named objects here that act as settings/place-holders. Then we reuse those throughout the code (rather than hard-coding):

df <- SHOPPERS

target <- "Revenue"


# ----------------------------------------------------------------------------- #
# 4. Train/Test Split ---------------------------------------------------------
# ----------------------------------------------------------------------------- #

set.seed(seed)

# stratified sampling is used to maintain the proportion of class labels in your training and test sets:
split=sample.split(df[[target]],SplitRatio = train_frac) # train_frac was specified under the setup above

training_set=subset(df,split==TRUE)
test_set=subset(df,split==FALSE)

# ----------------------------------------------------------------------------- #
# 4. Data preprocessing  ------------------------------------------------------
# ----------------------------------------------------------------------------- #

# Note that in this demonstration, we will be fitting a Naive Bayes classifier, decision tree and a logistic regression model. None of these algorithms require data preprocessing of the attributes (normalization and dummmy variable encoding). However, for other algorithms (SVM, KNN, NN and various ensemble methods), this is the stage is applied (to both the training and test sets).


# ----------------------------------------------------------------------------- #
# 5. Initialize H2O -----------------------------------------------------------
# ----------------------------------------------------------------------------- #

h2o.init()

# To use data in H2O functions/models, it needs to be an H2O data frame. The following converts the built-in R data frame into an H2O frame and stores it in the H2O memory space. Now, all future processing (modeling, predictions) happens in H2O's memory space (inside the Java engine, not R's memory). H2O is great at handling big datasets relative to RAM size due to its optimized data structures.

# Convert  data to H2O dataframe
train_h2o <- as.h2o(training_set)
test_h2o  <- as.h2o(test_set)

# ----------------------------------------------------------------------------- #
# 6. Specify the attribute names ----------------------------------------------
# ----------------------------------------------------------------------------- #

# The target was already specified in step 3, so now we need to specify the names of the columns that would be considered as the attributes.

# Rather than writing out the column names manually, we use setdiff function, such as setdiff(x,y) to return the elements in x that are not in y. This removes the target from the list to only leave the predictors:

predictors <- setdiff(names(training_set), target)


# ----------------------------------------------------------------------------- #
# 7. Fit Naive Bayes Classifier -----------------------------------------------
# ----------------------------------------------------------------------------- #

# Build and train the Naive Bayes Classifier (https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/naive-bayes.html):

# No hyperparameter tuning is required so we can go straight into fitting the model:

########## --> Fit the model ----

nb <- h2o.naiveBayes(
  x = predictors,
  y = target,
  training_frame = train_h2o,
  laplace = 0, # a smoothing parameter for categories with 0 observations to avoid zero probabilities
  nfolds = folds, # this is based on 5 or 10 folds as specified in the setup (section 1)
  seed = seed # based on the seed set
)

# check performance of model on the training set (which is based on optimal threshold from F1 score):
h2o.performance(nb)

# we will extract the predicted probabilities of class label = 1, append it to the original training and test sets to determine the predicted class for each based on the threshold and then create our confusion matrix and obtain the performance measures:

########## --> Extract predicted probabilities ----

# Save predicted probabilities
preds_nb_train <- h2o.predict(nb, train_h2o)
preds_nb_test <- h2o.predict(nb, test_h2o)

# Convert predictions to R data.frames to extract from H2O environment:
preds_nb_train <- as.data.frame(preds_nb_train)
preds_nb_test <- as.data.frame(preds_nb_test)

# view the extracted predictions to see what we actually obtained:

View(preds_nb_train)

# Column 3 contained the predicted probabilities for out positive class (class 1).

# Append column 3 (predicted probabilities for class label = 1) to original training and test sets:

train_nb_pred <- cbind(training_set,
                       setNames(preds_nb_train[, 3, drop = FALSE], "pred_prob")) # extract the predicted probabilities in column 3 of preds_nb_train, combine it with the original training set and call the column "pred_prob", this all is saved in a new dataframe called train_nb_pred

test_nb_pred <- cbind(test_set,
                      setNames(preds_nb_test[, 3, drop = FALSE], "pred_prob")) # the same is done with the test set

# view the above:

View(train_nb_pred)

########## --> Specify threshold ----

# We will look at how to find the optimal threshold in practical 5

threshold <- 0.5

########## --> Determine the predicted class labels ----

# This following converts predicted probabilities into class labels by applying a threshold: observations with predicted probability above the threshold are classified as "1" (the positive class), and those below as "0", with the result stored as a factor.

# training
train_nb_pred$pred_class <- factor(ifelse(train_nb_pred$pred_prob > threshold,"1","0"))

# test
test_nb_pred$pred_class <- factor(ifelse(test_nb_pred$pred_prob > threshold, "1", "0"))

########## --> Obtain confusion matrix and model performance ----

# training set

# predicted classes first then actual classes for the training set
confusionMatrix(
  train_nb_pred$pred_class,
  train_nb_pred[[target]],
  positive = "1",
  mode = "everything"
)

# actual classes first then predicted probabilities for the training set
roc_nb_train <- roc(train_nb_pred[[target]], train_nb_pred$pred_prob)
auc(roc_nb_train)
plot(roc_nb_train)


# test set

# predicted classes first then actual classes for the test set
confusionMatrix(
  test_nb_pred$pred_class,
  test_nb_pred[[target]],
  positive = "1",
  mode = "everything"
)

# actual classes first then predicted probabilities for the test set
roc_nb_test <- roc(test_nb_pred[[target]], test_nb_pred$pred_prob)
auc(roc_nb_test)
plot(roc_nb_test)


# ----------------------------------------------------------------------------- #
# 8. Fit a decision tree using h2o --------------------------------------------
# ----------------------------------------------------------------------------- #

# We will demonstrate how to fit a DT using H2O which uses pre-pruning, as well as RPART which uses post-pruning

########## --> Hyperparameter tuning ----

# we will start by tuning the hyperparameter. The possible hyperparameters are:

# max_depth: Maximum depth of the tree
# min_rows: Minimum number of observations in a leaf
# min_split_improvement: Minimum reduction in error required to make a split


# Set up the hyperparameter search space (we will do an example for max_depth and min_rows)
hyper_params <- list(
  max_depth = seq(3, 21, by = 2), # from 3 to 21 in increments of 2
  min_rows = c(1, 5, 10, 20, 50)

)

# Define search criteria:
search_criteria <- list(
  strategy = "Cartesian" # Try "RandomDiscrete" for random search
)

# we use the h2o.grid function for cross validation and hyperparameter tuning. This function requires a grid_id which is a label for the entire grid search run so that H2O can store, retrieve, and reference it.

#NB: Each grid search is saved as a named experiment in grid_id. Once a grid_id is used, it cannot be overwritten - it is permanently stored in the H2O session until we explicitly remove it.

# Therefore, if we update the grid search or hyperparameter list and re-run the h2o.grid function, we need to remove the anything store in the grid_id first:

h2o.rm("dtree_grid")

# Run the grid search using a single decision tree, GBM (gradient boosting method) with ntrees = 1)
grid <- h2o.grid(
  algorithm = "gbm",
  grid_id = "dtree_grid", # this is just the ID we are giving the grid search
  x = predictors,
  y = target,
  training_frame = train_h2o,
  hyper_params = hyper_params,
  search_criteria = search_criteria,
  ntrees = 1,
  learn_rate = 1.0, # Full weight per tree (since it's only one)
  sample_rate = 1.0,  # use 100% of the training data
  col_sample_rate = 1.0, # use 100% of the attributes
  stopping_rounds = 0,
  seed = seed
)

# Next we order the grid search results according to the best CV performance based on our selected metric and save it in model_results_dt

model_results_dt <- h2o.getGrid("dtree_grid", sort_by = metric, decreasing = TRUE)

# Let's view the CV results
print(model_results_dt)

# Extract the best model ID _which is in the first row of model_results_dt
best_model_id <- model_results_dt@model_ids[[1]]

# Retrieve hyperparameter values associated with the best model
best_model <- h2o.getModel(best_model_id)

# Step 1: Automatically identify which hyperparameters were tuned
# (this pulls the names directly from the hyper_params list - no hardcoding)
tuned_param_names <- names(hyper_params)

# Step 2: Extract the actual tuned values that were used in the best model
# (best_model@parameters stores everything that was actually applied)
best_tuned_values <- lapply(tuned_param_names, function(param_name) {
  best_model@parameters[[param_name]]
})

# append the hyperparameter names to the tuned values:
names(best_tuned_values) <- tuned_param_names

########## --> Fit the model ----

# Step 3: Build and train the final decision tree using h2o.decision_tree()
# We use do.call so the extracted hyperparameters are passed automatically
# (you can add any other fixed parameters you want here)
# categorical_encoding = "LabelEncoder" converts factor columns to integers before
# splitting, which avoids an assertion error in h2o.decision_tree's CategoricalSplittingRule

final_dt_model <- do.call(h2o.decision_tree, c(
  list(
    x = predictors,
    y = target,
    training_frame = train_h2o,
    categorical_encoding = "LabelEncoder", # prevents CategoricalSplittingRule assertion error
    seed = seed                       # the seed
  ),
  best_tuned_values     # <- automatically inserts max_depth, min_rows, etc.
))

########## --> Extract predicted probabilities ----

# Save predicted probabilities
preds_dt_train <- h2o.predict(final_dt_model, train_h2o)
preds_dt_test <- h2o.predict(final_dt_model, test_h2o)

# Convert predictions to R data.frames to extract from H2O environment:
preds_dt_train <- as.data.frame(preds_dt_train)
preds_dt_test <- as.data.frame(preds_dt_test)

#the structure of the predicted output is the same as that from the Naive Bayes:
View(preds_dt_train)

# Append column 3 (predicted probabilities for class label = 1) to original training and test sets in the exact same manner as that for the Naive bayes:

train_dt_pred <- cbind(training_set,
                       setNames(preds_dt_train[, 3, drop = FALSE], "pred_prob"))

test_dt_pred <- cbind(test_set,
                      setNames(preds_dt_test[, 3, drop = FALSE], "pred_prob"))

# view the above:

View(train_dt_pred)

########## --> Specify threshold ----

# We will look at how to find the optimal threshold in practical 5

threshold <- 0.5

########## --> Determine the predicted class labels ----

# training
train_dt_pred$pred_class <- factor(ifelse(train_dt_pred$pred_prob > threshold,"1","0"))

# test
test_dt_pred$pred_class <- factor(ifelse(test_dt_pred$pred_prob > threshold, "1", "0"))

########## --> Obtain confusion matrix and model performance ----

# training set

# predicted classes first then actual classes for the training set
confusionMatrix(
  train_dt_pred$pred_class,
  train_dt_pred[[target]],
  positive = "1",
  mode = "everything"
)

# actual classes first then predicted probabilities for the training set
roc_dt_train <- roc(train_dt_pred[[target]], train_dt_pred$pred_prob)
auc(roc_dt_train)
plot(roc_dt_train)


# test set

# predicted classes first then actual classes for the test set
confusionMatrix(
  test_dt_pred$pred_class,
  test_dt_pred[[target]],
  positive = "1",
  mode = "everything"
)

# actual classes first then predicted probabilities for the test set
roc_dt_test <- roc(test_dt_pred[[target]], test_dt_pred$pred_prob)
auc(roc_dt_test)
plot(roc_dt_test)



# ----------------------------------------------------------------------------- #
# 9. Fit a decision tree using rpart --------------------------------------------
# ----------------------------------------------------------------------------- #

# we will use another package that fits a DT that allows for a visualization. This package grows the tree by implementing Cost-Complexity pruning, where pre and post pruning is implemented using a complexity parameter (cp). The complexity parameter (cp) is used to control the size of the decision tree and to select the optimal tree size.

# cp sets the minimum improvement a split must provide to be worth making.
# Any split that does not decrease the overall lack-of-fit (error) by at least a factor of cp is not attempted.
# Higher cp -> smaller, simpler trees (more pruning).
# Lower cp -> larger, more complex trees (less pruning, higher risk of overfitting).

#rpart does not have functionality for specifying grid searches, only single values.

########## --> Fit the DT using Rpart ----

# NOTE: We use the training and test sets that were originally split, not the ones from H2O.

set.seed(seed)

DT_rpart <- rpart(
                as.formula(paste(target, "~ .")),
                data = training_set,
                method = "class", # for classification
                xval = folds, # CV
                control = rpart.control(
                  #cp = 0.01,             # complexity parameter for pruning
                  minsplit = 20,         # minimum observations to attempt a split
                  maxdepth = 5           # maximum depth
                )
              ) # default attribute selection measure is Gini Index

DT_rpart # run this to see information about the fitted tree

# rpart automatically searches over values for cp to be tuned.

# rpart grows a large tree first (with a very small cp), then considers a whole sequence of smaller sub-trees by increasing the effective cp.

# --------------------------------------------  #
# While the tree is being grown, a samll cp acts as a stopping rule:

# A split is only made if it improves the model fit by at least cp amount.
# More precisely: the reduction in impurity must exceed cp.

# So if cp is large -> fewer splits -> smaller tree
# If cp is very small -> tree grows much deeper

# This is pre-pruning (early stopping).
# -------------------------------------------- #
# After growing a large tree, rpart also uses cp in a cost-complexity pruning framework:

# It computes a sequence of nested subtrees
# Each subtree corresponds to a different cp value
# These are stored in the complexity parameter table (cptable)
# -------------------------------------------- #

printcp(DT_rpart)
plotcp(DT_rpart)

# The cp table includes:

# The cp penalty value associated with each subtree

# nsplit: Number of splits in the tree (more splits = more complex tree)

# The rel error is the total error of the model divided by the error of the initial model (a model with just the root node, predicting the most frequent class). It's a measure of the error relative to the simplest possible model.

# The xerror is the cross-validation error of the model relative to te root node model(0 splits). It is computed during the tree-building process if cross-validation is enabled (e.g., using the xval argument in rpart()). This error is estimated by applying the decision tree to each of the cross-validation folds used during tree construction. It provides a measure of how well the tree is likely to perform on unseen data, hence an estimate of the model's generalization error. Typically, it helps identify if the model is overfitting. If xerror starts to increase as the complexity of the model increases (more splits in the tree), it may suggest that simpler models are preferable.

# The xstd is the standard error of the cross-validation error (xerror). This value provides an indication of the variability of the cross-validation error estimate. A high standard error suggests that the cross-validation error might not be a reliable estimate of the model's error on new data, possibly due to the model being unstable across different subsets of the training data or due to a small number of cross-validation folds.

# rpart() automatically computes the optimal tree size (considering complexity cost) using these metrics. Specifically, xerror and xstd are used to determine the smallest tree that is within one standard error of the minimum cross-validation error (xerror + xstd). This criterion helps to balance model accuracy with complexity, aiming to avoid overfitting while maintaining sufficient explanatory power.


########## --> Extract predicted probabilities ----

### Extract predicted probabilities:
# Note: this provides TWO columns - the predicted probabilities for "0" in column 1 and "1" in column 2.

pred_prob_DT_train <- predict(DT_rpart, newdata = training_set, type = "prob")

train_DT_rpart <- cbind(training_set,
                        setNames(data.frame(pred_prob_DT_train[, 2]), "pred_prob")) # We only want the probs in column 2 (for "1")

# View the results:

View(train_DT_rpart)


pred_prob_DT_test <- predict(DT_rpart, newdata = test_set, type = "prob")

test_DT_rpart <- cbind(test_set,
                        setNames(data.frame(pred_prob_DT_test[, 2]), "pred_prob")) # We only want the probs in column 2 (for "1")

########## --> Specify threshold ----

# We will look at how to find the optimal threshold in practical 5

threshold <- 0.5

########## --> Determine the predicted class labels ----

# training set
train_DT_rpart$pred_class <- factor(ifelse(train_DT_rpart$pred_prob > threshold, "1","0"))

# test
test_DT_rpart$pred_class <- factor(ifelse(test_DT_rpart$pred_prob > threshold,"1","0"))

########## --> Obtain confusion matrix and model performance ----

# training set

# predicted classes first then actual classes for the training set
confusionMatrix(
  train_DT_rpart$pred_class,
  train_DT_rpart[[target]],
  positive = "1",
  mode = "everything"
)

# actual classes first then predicted probabilities for the training set
roc_DT_train_rpart <- roc(train_DT_rpart[[target]], train_DT_rpart$pred_prob)
auc(roc_DT_train_rpart)
plot(roc_DT_train_rpart)

# test set

# predicted classes first then actual classes for the test set
confusionMatrix(
  test_DT_rpart$pred_class,
  test_DT_rpart[[target]],
  positive = "1",
  mode = "everything"
)

# actual classes first then predicted probabilities for the test set
roc_DT_test_rpart <- roc(test_DT_rpart[[target]], test_DT_rpart$pred_prob)
auc(roc_DT_test_rpart)
plot(roc_DT_test_rpart)


########## --> visualize the DT ----

dev.new(width = 15, height = 20) # This just allows the plot to be shown in a separate window (useful for small screens)

rpart.plot(DT_rpart)
rpart.plot(DT_rpart, yesno = 1, type = 2, fallen.leaves = FALSE) # add additional options to change the appearance.
# see http://www.milbo.org/rpart-plot/prp.pdf for more options to customize the plot



# ----------------------------------------------------------------------------- #
# 10. Fit a logistic regression model --------------------------------------------
# ----------------------------------------------------------------------------- #

# A logistic regression is in the class of a generalized linear model (GLM), various GLMs can be fitted in h2o for different types of responses (continuous, binary, count, multiple categories - multi-class classification)

# An LR model has no hyperparameters to tune.

########## --> Fit the LR model in H2O ----

# Fit the logistic regression model
LR <- h2o.glm(
  x = predictors,
  y = target,
  training_frame = train_h2o,
  family = "binomial", # logistic regression
  lambda = 0, # no regularization (like classical GLM)
  compute_p_values = TRUE, # optional: get p-values
  remove_collinear_columns = TRUE # required when compute_p_values = TRUE and collinear columns exist
)

########## --> Perform inference using the LR model ----

# extract p-values for inference and save into a df called LR_results

LR_results <- LR@model[["coefficients_table"]]

# create odds ratios from the regression coefficient estimates
LR_results$OR <- exp(LR_results[,2])

# round the p-values off to 4 decimal places
LR_results$p_value <- round(LR_results$p_value,4)

View(df)

########## --> Extract predicted probabilities ----

# Save predicted probabilities
preds_LR_train <- h2o.predict(LR, train_h2o)
preds_LR_test <- h2o.predict(LR, test_h2o)

# Convert predictions to R data.frames to extract from H2O environment:
preds_LR_train <- as.data.frame(preds_LR_train)
preds_LR_test <- as.data.frame(preds_LR_test)

# Append column 3 (predicted probabilities for class label = 1) to original training and test sets:

train_LR_pred <- cbind(training_set,
                       setNames(preds_LR_train[, 3, drop = FALSE], "pred_prob"))

test_LR_pred <- cbind(test_set,
                       setNames(preds_LR_test[, 3, drop = FALSE], "pred_prob"))



########## --> Specify threshold ----

# We will look at how to find the optimal threshold in practical 5

threshold <- 0.5

########## --> Determine the predicted class labels ----

# training
train_LR_pred$pred_class <- factor(ifelse(train_LR_pred$pred_prob > threshold,"1","0"))

# test
test_LR_pred$pred_class <- factor(ifelse(test_LR_pred$pred_prob > threshold, "1", "0"))

########## --> Obtain confusion matrix and model performance ----

# training set

# predicted classes first then actual classes for the training set
confusionMatrix(
  train_LR_pred$pred_class,
  train_LR_pred[[target]],
  positive = "1",
  mode = "everything"
)

# actual classes first then predicted probabilities
roc_LR_train <- roc(train_LR_pred[[target]], train_LR_pred$pred_prob)
auc(roc_LR_train)
plot(roc_LR_train)

# test

# predicted classes first then actual classes
confusionMatrix(
  test_LR_pred$pred_class,
  test_LR_pred[[target]],
  positive = "1",
  mode = "everything"
)

# actual classes first then predicted probabilities
roc_LR_test <- roc(test_LR_pred[[target]], test_LR_pred$pred_prob)
auc(roc_LR_test)
plot(roc_LR_test)


#################### Combine ROC curves of test set for all models ############

# Plot (see https://r-charts.com/colors/ for more colours)
plot(
  roc_nb_test,
  col = "#458B74",
  lwd = 2,
  main = "ROC Curve Comparison of test set for NB, DT and LR"
)
lines(roc_DT_test_rpart, col = "#CD3333", lwd = 2)
lines(roc_LR_test, col = "#009ACD", lwd = 2)

# Add legend
legend(
  "bottomright",
  legend = c("Naive Bayes", "Decision tree", "Logistic regression"),
  col = c("#458B74", "#CD3333", "#009ACD"),
  lwd = 2
)

# 6. Balancing of the training set ---------------------------------------------
# ----------------------------------------------------------------------------- #

# We will consider 4 different sampling techniques: under, over, combination of over and under, and SMOTE

##################### Under-sampling ###############################

# The process of undersampling counts the number of minority samples in the dataset (given by for formula for total_under below), then randomly selects the same number from the majority sample. In our case we would end up with 70 randomly chosen non-depression cases ("0") and the original 70 depression cases ("1") resulting in a 50:50 split.
# This has a major drawback as we are only using a very small % of the original dataset.

# save the number of MINORITY cases in an object call total_under:
total_under <- nrow(training_set[training_set[[target]] == "1", ])

train_under <- ovun.sample(
  as.formula(paste(target, "~ .")), # this formula specifies what the target is and the predictors
  data = training_set,
  method = "under",
  N = 2 * total_under, # new total = multiply by 2 for the two classes
  seed = seed
)

# Extract and save the resulting under-sampled data:
train_under_data <- train_under$data

summary(train_under_data[[target]])

############################## Over-sampling ##############################################

# This method repeatedly duplicates randomly selected minority classes until there are an equal number of majority and minority samples. It does have its drawback as the duplicates may lead to generalizing of the minority class.

# save the number of MAJORITY cases in an object call total_over:
total_over <- nrow(training_set[training_set[[target]] == "0", ])

train_over <- ovun.sample(
  as.formula(paste(target, "~ .")),
  data = training_set,
  method = "over",
  N = 2 * total_over, # multiply by 2 for the twp classes
  seed = seed
)

# Extract and save the resulting under-sampled data:
train_over_data <- train_over$data
summary(train_over_data[[target]])

###################### Combination of over and under #################################

# We can apply a combination of both over- and under-sampling, where the number of minority
# cases increases and the number of majority cases decreases.

total_both <- nrow(training_set) # specify the total sample size after the procedure, this can be changed to any value
fraction_new <- 0.50 # specify the approx proportion of minority cases to be produced

train_both <- ovun.sample(
  as.formula(paste(target, "~ .")),
  data = training_set,
  method = "both",
  N = total_both,
  p = fraction_new,
  seed = seed
)

# Extract and save the resulting data (list):
train_both_data <- train_both$data
summary(train_both_data[[target]])

####################################### SMOTE ##################################


# We use the SmoteClassif function which allows us to specify the method of determining the synthetic observations based on the nearest neighbours. We use the dist option to specify the method to use based on the type of data (see https://rdrr.io/cran/UBL/man/smoteClassif.html)

# The depression data has mixed attributes (numerical and categorical), so we use HEOM or HVDM

set.seed(seed)
train_smote_data <- SmoteClassif(
  as.formula(paste(target, "~ .")),
  training_set,
  C.perc = "balance", # minority and majority classes 
  k = 5, # number of nearest neighbours,
  dist = "HVDM"
)

summary(train_smote_data[[target]])

###############################################################################


########################### Which technique to use?? ##########################

# We can use multiple models on each technique to find the best performing model/technique combination but this is very time-consuming. So we can fit a base model instead (usually a simple model) to each balanced data set to find the technique that produces the best performing model (based on the selected metric). We then take the balanced set based on that technique and fit all of the competing models. A logistic regression model is generally the easiest to fit. We will once again use the H2O package for this:

# ----------------------------------------------------------------------------- #
# 7. Initialize H2O -----------------------------------------------------------
# ----------------------------------------------------------------------------- #

h2o.init()

# ----------------------------------------------------------------------------- #
# 8. Specify the attribute names ----------------------------------------------
# ----------------------------------------------------------------------------- #

predictors <- setdiff(names(training_set), target)

# ----------------------------------------------------------------------------- #
# 9.Repeatedly fit the LR model ----------------------------------------------
# ----------------------------------------------------------------------------- #

# Recall that we now have 4 balanced training sets and the original unbalanced set:
# "train_under_data" (based on under-sampling),
# "train_over_data" (based on over-sampling),
# "train_both_data" (based on a combination of over- and under-sampling)
# "train_smote_data" (based on SMOTE)
# "training_set" (the unbalanced original training set)

# We will iterate through each of them, fitting an LR and determine their performances.

# Prepare the results data frame to append the results of each data set:
Performance_comparison <- data.frame()

#################  THE CODE BELOW IS RE-RUN FOR EACH TRAINING SET  (see lecture recording) ############

# Let's create a variable to use for the name of the balanced training set which we can update rather than repeating the code for each training set:

### Loop through this code to get appended results, just change the name of the data set:  


train_set_name <- "train_over_data"

# train_set_name just stores the name of the data set, not the data set itself. To use the actual data frame referenced by its name, you can use get():

training_set_final <- get(train_set_name)

# convert to h2o data sets:

balanced_train_set_h2o <- as.h2o(training_set_final)
test_h2o <- as.h2o(test_set) # convert the test set created on line 43 to an H2O data set 


# Fit the logistic regression model
LR <- h2o.glm(
  x = predictors,
  y = target,
  training_frame = balanced_train_set_h2o,
  family = "binomial", # logistic regression
  lambda = 0, # no regularization (like classical GLM)
  compute_p_values = TRUE # optional: get p-values
)

########## --> Extract predicted probabilities ----

# Save predicted probabilities
preds_LR_train <- h2o.predict(LR, balanced_train_set_h2o)
preds_LR_test <- h2o.predict(LR, test_h2o)

# Convert predictions to R data.frames to extract from H2O environment:
preds_LR_train <- as.data.frame(preds_LR_train)
preds_LR_test <- as.data.frame(preds_LR_test)

# Append column 3 (predicted probabilities for class label = 1) to original training and test sets:

train_LR_pred <- cbind(training_set_final,
                       setNames(preds_LR_train[, 3, drop = FALSE], "pred_prob"))  

test_LR_pred <- cbind(test_set,
                      setNames(preds_LR_test[, 3, drop = FALSE], "pred_prob")) 


# ----------------------------------------------------------------------------- #
# 10. Save model performance results repeatedly for each data set --------------
# ----------------------------------------------------------------------------- #

# When training a classification model on a balanced dataset (achieved through oversampling techniques such as random oversampling or SMOTE), the model learns from an artificial class distribution that does not reflect reality. As a result, the predicted probabilities will be miscalibrated, typically inflated for the minority class. 

# If the goal is simply to predict a class label and the decision threshold has been appropriately tuned, correction may not be necessary. Similarly, if evaluation relies solely on AUC-ROC, calibration does not affect the ranking of predictions and correction can be omitted. 

# Therefore, we will use the AUC to compare the performance of the LR models fitted with the different datasets:

######### Extract and append results to a data frame ############

######### REPEAT THESE STEPS FOR ALL BALANCED DATA SETS ######################### 

# Step 1: Extract metrics from confusion matrices

# Training AUC
roc_LR_train <- roc(train_LR_pred[[target]], train_LR_pred$pred_prob)
auc_train <- auc(roc_LR_train)

# Test AUC
roc_LR_test <- roc(test_LR_pred[[target]], test_LR_pred$pred_prob)
auc_test <- auc(roc_LR_test)

# Step 2: Create individual rows for train and test
performance_train <- data.frame(
  technique = train_set_name,
  dataset = "train",
  auc = auc_train
)

performance_test <- data.frame(
  technique = train_set_name,
  dataset = "test",
  auc = auc_test
)

# Step 3: Append into one data frame (joins results from previous models each time)
Performance_comparison <- rbind(Performance_comparison,performance_train, performance_test, make.row.names = FALSE)


# Step 4: View results
View(Performance_comparison)
print(Performance_comparison)

#### Use the balanced data set that produces the highest AUC.

# ----------------------------------------------------------------------------- #
# 11. Exporting the balanced data set for later use ---------------------------
# ----------------------------------------------------------------------------- #

# You can save the balanced set (and test set) for later use (saves to your working directory):

save(
  train_over_data, # change this based on the optimal balanced data set
  test_set, # save test set too
  file = "Unbalanced and balanced data.RDATA"
)

# Then to use the saves data objects when you start a new R session, load all the saved objects into your RStudio environment:

load("Unbalanced and balanced data.RDATA")

# ----------------------------------------------------------------------------- #
# 12. Correcting probabilities for over- or under- sampling -------------------
# ----------------------------------------------------------------------------- #

# If predicted probabilities are meaningful in the context of the problem, for example in risk scoring or any decision that depends on the magnitude of the probability (such as predicted class labels), then correction is essential. 

# We have two options to obtain threshold-dependent metrics for a model (from a confusion matrix):
# 1. Tune the threshold used to make the class predictions, then probability correction is not required - this will be covered in the last practical (prac 5)
# 2. For random over- or under-sampling, a Bayes-based correction (see slide 48 in chapter 2) can be applied using the known true and artificial class proportions to rescale the predicted probabilities back to reflect the real-world distribution. 

#For SMOTE, this correction is less theoretically clean because synthetic interpolated samples distort not just the class prior but the feature space itself. In this case, probability calibration techniques such as Platt scaling or isotonic regression, applied on a held-out validation set, are more appropriate for recovering well-calibrated probability estimates.

# We will manually correct the probabilities using R code for any model that has been fitted using the balanced training set (over-, under-, combination)


# -------------------------------------------------->
# 1. Specify the name of the dataframe containing the combined training/test sets with predicted probabilities:
# -------------------------------------------------->

combined_train <- train_LR_pred
combined_test <- test_LR_pred

# -------------------------------------------------->
# 2. True class proportions from the ORIGINAL training set (before balancing)
# -------------------------------------------------->
p_1 <- prop.table(table(training_set[[target]]))["1"]   # true minority class proportion

# -------------------------------------------------->
# 3. Balanced class proportions AFTER over/under sampling
# -------------------------------------------------->
p_asterisk_1 <- prop.table(table(combined_train[[target]]))["1"]

# -------------------------------------------------->
# 4. Correction function 
# -------------------------------------------------->
probs_correction <- function(predicted_probs, p_1, p_asterisk_1) {
  
  numerator   <- predicted_probs * (p_1 / p_asterisk_1)
  denominator <- numerator + (1 - predicted_probs) * ((1 - p_1)/(1 - p_asterisk_1))
  
  corrected   <- numerator / denominator
  return(corrected)
  
}

# -------------------------------------------------->
# 5. Apply to training set
# -------------------------------------------------->

# combined_train$pred_prob are the raw predicted probabilities from your model

combined_train$corrected_probs <- probs_correction(combined_train$pred_prob,
                                                   p_1, 
                                                   p_asterisk_1)


# -------------------------------------------------->
# 6. Apply to test set
# -------------------------------------------------->

# combined_test$pred_prob are the raw predicted probabilities from your model

combined_test$corrected_probs <- probs_correction(combined_test$pred_prob,
                                                  p_1, 
                                                  p_asterisk_1)

# -------------------------------------------------->
# 7. Look at model performance on the training and test sets:
# -------------------------------------------------->

########## --> Specify threshold ----

# We will look at how to find the optimal threshold in practical 5

threshold <- 0.003 # for now, let's use a threshold close to the original prevalence

########## --> Determine the predicted class labels ----

# training
combined_train$pred_class <- factor(ifelse(combined_train$corrected_probs > threshold,"1","0"))

# test
combined_test$pred_class <- factor(ifelse(combined_test$corrected_probs > threshold, "1", "0"))

########## --> Obtain confusion matrix and model performance ----

# training set 

# predicted classes first then actual classes for the training set
metrics_train <- confusionMatrix(
  combined_train$pred_class,
  combined_train[[target]],
  positive = "1",
  mode = "everything"
)

# calculate Matthews Correlation Coefficient (MCC) for the training set
mcc_train <- mcc(combined_train$pred_class,  combined_train[[target]])

# Recall: MCC produces a value between −1 and +1, where 
# +1 indicates perfect prediction, 
# 0 indicates performance no better than random guessing, and 
# −1 indicates total disagreement between predictions and actual outcomes. 


# test

# predicted classes first then actual classes
metrics_test <- confusionMatrix(
  combined_test$pred_class,
  combined_test[[target]],
  positive = "1",
  mode = "everything"
)

# calculate Matthews Correlation Coefficient (MCC) for the test set
mcc_test <- mcc(combined_test$pred_class,  combined_test[[target]])

# -------------------------------------------------->
# 8. Save model performance
# -------------------------------------------------->

metrics_combined <- rbind(train = c(as.list(metrics_train$byClass),MCC = mcc_train),  
                          test = c(as.list(metrics_test$byClass),MCC = mcc_test))

# or save as a CSV file

write.csv(metrics_combined, "model_performance_metrics.csv", row.names = TRUE)

############## Shut down H2O cluster so it doesn't use up any more resources ############

h2o.shutdown(prompt = FALSE)


############## Shut down H2O cluster so it doesn't use up any more resources ############

h2o.shutdown(prompt = FALSE)
