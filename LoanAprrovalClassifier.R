
# Part 1: Data Pre-Processing

# Clear local R environment and console

rm(list = ls())
cat("\014")

# Import libraries used

library(formattable)
library(dplyr)
library(ggplot2)
library(psych)
library(caret)
library(class)
library(caTools)
library(glmnet)
library(e1071)
library(randomForest)
library(corrplot)
library(kernlab)
library(reshape2)
library(pROC)
library(MASS)
library(rpart)
library(stats)


# Import CSVs into dataframes

applications <- read.csv("application_record.csv")
credit <- read.csv("credit_record.csv")


# Look at structure of both dataframes
glimpse(applications)
glimpse(credit)


# As seen, the applications dataframe has about 400,000 rows with 18 columns, while the credit dataframe has about a million rows with 3 columns
# 
# These dataframes were found online and blanks/nulls represent empty values instead of NAs
# Find all null values in the dataframe and set them to NA, which can be found easier in R

applications[applications == ""] <- NA
credit[credit == ""] <- NA
print(colSums(is.na(applications)))
print(colSums(is.na(credit)))


# As seen above, the only column in both dataframes with NAs now is OCCUPATION_TYPE in the applications dataframe
# 
# Merge both dataframes into a single dataframe (df) by the common column, ID
# Check the tail of the new dataframe

df <- merge(applications, credit, by = "ID")
print(tail(df, 10))

glimpse(df)

# We can see the new dataframe has 777715 rows and 20 columns <br>
#   
#   Observe the distribution of our target variable, STATUS
# As seen below, most of the data is in 0, C and X

table(df$STATUS)

# We have decided to remove X, so we filter it from the dataframe
# The distribution of STATUS is seen again, to show X has been removed

df <- df %>%
  filter(STATUS != 'X')
table(df$STATUS)

# Randomly filter the dataframe by taking a random entry from each user, to prevent the same user's
# data from appearing twice

df <- df %>%
  group_by(ID)
df <- df %>%
  sample_n(1)

# Remove undesired columns from the dataframe
df <- df %>% ungroup()
df <- dplyr::select(df, -c(ID, CNT_CHILDREN, FLAG_MOBIL, FLAG_WORK_PHONE, FLAG_PHONE, OCCUPATION_TYPE, MONTHS_BALANCE))

# The target variable is now transformed to a binary variable, to enable binary classification

df <- df %>% mutate(STATUS = ifelse(STATUS == "C", 1, 0))

# Observe structure of target variable after these changes
table(df$STATUS)

# The target variable is unbalanced, which may lead to problems in modelling
# 
# Check which columns are non-numerical and which are numerical
# Non-numeric columns

print(names(df)[sapply(df, function(x) is.character(x) | is.factor(x))])


# Numeric columns
print(names(df)[sapply(df, function(x) is.numeric(x))])

# Check if numeric columns are categorical or continuous through number of distinct values.
# (The target variable STATUS has already been converted to a binary variable, so it is not checked)

print(n_distinct(df$AMT_INCOME_TOTAL))
print(n_distinct(df$DAYS_BIRTH))
print(n_distinct(df$DAYS_EMPLOYED))
print(n_distinct(df$FLAG_EMAIL))
print(n_distinct(df$CNT_FAM_MEMBERS))

# As seen above, the first three columns checked (AMT_INCOME_TOTAL, DAYS_BIRTH, and DAYS_EMPLOYED) are 
# continuous as they have far more than 10 distinct values.
# 
# The others are categorical, since they have 10 or less distinct values
# 
# 
# 
# Convert the categorical columns to factor.
# 
# Note that some of the conversions give the new factor variables levels if they are appropriate
# The CNT_FAM_MEMBERS column was also modified while being changed to factor level, with all 
# families with more than 4 members stored as the same value

df$CODE_GENDER <- factor(df$CODE_GENDER)
df$FLAG_OWN_CAR <- factor(df$FLAG_OWN_CAR)
df$FLAG_OWN_REALTY <- factor(df$FLAG_OWN_REALTY)
df$NAME_EDUCATION_TYPE <- factor(df$NAME_EDUCATION_TYPE, levels = c("Lower secondary",
                                                                    "Secondary / secondary special",
                                                                    "Incomplete higher",
                                                                    "Higher education",
                                                                    "Academic degree"))
df$NAME_FAMILY_STATUS <- factor(df$NAME_FAMILY_STATUS)
df$NAME_INCOME_TYPE <- factor(df$NAME_INCOME_TYPE)
df$NAME_HOUSING_TYPE <- factor(df$NAME_HOUSING_TYPE)
df$STATUS <- factor(df$STATUS)

df$CNT_FAM_MEMBERS <- cut(df$CNT_FAM_MEMBERS, breaks = c(0, 1, 2, 3, 20), labels = c("1", "2", "3", "4+"))
df$FLAG_EMAIL <- factor(df$FLAG_EMAIL)

# The two other columns are numerical and continuous, but are stored in negative days, so are converted to years. 
df <- df %>% mutate(DAYS_BIRTH = round((DAYS_BIRTH * -1) / 365.25))
df <- df %>% mutate(DAYS_EMPLOYED =  ifelse(DAYS_EMPLOYED > -1, 0, round((DAYS_EMPLOYED * -1) / 365.25)))

# These columns will now be normalised, along with the income column
df$normalised_income <- scale(df$AMT_INCOME_TOTAL, center = min(df$AMT_INCOME_TOTAL), scale = max(df$AMT_INCOME_TOTAL))
df$normalised_age <- scale(df$DAYS_BIRTH, center = min(df$DAYS_BIRTH), scale = max(df$DAYS_BIRTH))
df$normalised_empl_years <- scale(df$DAYS_EMPLOYED, center = min(df$DAYS_EMPLOYED), scale = max(df$DAYS_EMPLOYED))

# The dataframe is rearranged so the target variable column (STATUS) is the last column
df <- dplyr::select(df, -c(AMT_INCOME_TOTAL, DAYS_BIRTH, DAYS_EMPLOYED))
df <- df[, c(names(df)[!(names(df) %in% "STATUS")], "STATUS")]

# Change the names of the dataframe columns for better visibility
names(df) <- c("GENDER", "OWN_CAR", "OWN_REALTY", "INCOME_TYPE", "EDUCATION_TYPE",
               "FAMILY_STATUS", "HOUSING_TYPE", "EMAIL", "FAMILY_SIZE", "INCOME",
               "AGE", "YEARS_EMPLOYED", "STATUS")

# The pre-processing is now concluded, so the dataframe is shown
str(df)

# Part 2: Exploratory Data Analysis (and handling outliers)

# Boxplot of numerical columns
ggplot(data = df, aes(x = 1, y = INCOME)) + geom_boxplot()
ggplot(data = df, aes(x = 1, y = AGE)) + geom_boxplot()
ggplot(data = df, aes(x = 1, y = YEARS_EMPLOYED)) + geom_boxplot()

# We will remove the outliers seen above in the numeric columns
numeric_cols <- c("INCOME", "AGE", "YEARS_EMPLOYED")

for(col in numeric_cols){
  
  q1 <- quantile(df[[col]], 0.25)
  q3 <- quantile(df[[col]], 0.75) 
  iqr <- q3 - q1
  
  lower_bound <- q1 - 1.5*iqr
  upper_bound <- q3 + 1.5*iqr
  
  data <- subset(df, df[[col]] > lower_bound & df[[col]] < upper_bound)
  
}
par(mfrow=c(1,2))
boxplot(df$INCOME, 
        main="Before Removal",
        xlab="INCOME", 
        ylab="Frequency")

boxplot(data$INCOME,
        main="After Removal", 
        xlab="INCOME",
        ylab="Frequency")
par(mfrow=c(1,2))
boxplot(df$AGE, 
        main="Before Removal",
        xlab="AGE", 
        ylab="Frequency")

boxplot(data$AGE,
        main="After Removal", 
        xlab="AGE",
        ylab="Frequency")
par(mfrow=c(1,2))
boxplot(df$YEARS_EMPLOYED, 
        main="Before Removal",
        xlab="YEARS_EMPLOYED", 
        ylab="Frequency")

boxplot(df$YEARS_EMPLOYED,
        main="After Removal", 
        xlab="YEARS_EMPLOYED",
        ylab="Frequency")

# As seen above, we can observe the effect removing the outliers has on the boxplots
# 
# Pie charts of categorical columns (with more than 2 factors)

ggplot(data = df, aes(x = "", fill = INCOME_TYPE)) +
  geom_bar(width = 1, color = "white") +
  coord_polar("y", start = 0) +
  labs(title = "Income Type Distribution", fill = "Income Type") +
  theme_void() +
  theme(legend.position = "right")

ggplot(data = df, aes(x = "", fill = EDUCATION_TYPE)) +
  geom_bar(width = 1, color = "white") +
  coord_polar("y", start = 0) +
  labs(title = "Education Type Distribution", fill = "Type") +
  theme_void() +
  theme(legend.position = "right")

ggplot(data = df, aes(x = "", fill = FAMILY_STATUS)) +
  geom_bar(width = 1, color = "white") +
  coord_polar("y", start = 0) +
  labs(title = "Family Status Distribution", fill = "Status") +
  theme_void() +
  theme(legend.position = "right")

ggplot(data = df, aes(x = "", fill = HOUSING_TYPE)) +
  geom_bar(width = 1, color = "white") +
  coord_polar("y", start = 0) +
  labs(title = "Housing Type Distribution", fill = "Type") +
  theme_void() +
  theme(legend.position = "right")

ggplot(data = df, aes(x = "", fill = FAMILY_SIZE)) +
  geom_bar(width = 1, color = "white") +
  coord_polar("y", start = 0) +
  labs(title = "Family Sizes Distribution", fill = "Size") +
  theme_void() +
  theme(legend.position = "right")

# Check correlation of numerical columns

numeric_columns <- df[, c("INCOME", "AGE", "YEARS_EMPLOYED")]
corrplot(cor(numeric_columns), method = "color", type = "upper", tl.cex = 0.7, tl.col = "black", tl.srt = 50)


# As seen above, the columns are not very correlated (between 0 and 0.3)
# 
# We can also summarise the columns with numeric values

summary(numeric_columns)


# 3: Modelling and Evaluation

# Each model is evaluated with accuracy, confusion matrix, and ROC Curve and AUC (Area under Curve)
# Split dataframe into train and test

split <- sample.split(df$STATUS, SplitRatio = 0.7)
train_df <- df[split, ]
test_df <- df[!split, ]


# Holdout Method

# Train the logistic model with the train dataset
logistic_model <- glm(STATUS ~ ., data = train_df, family = "binomial")

# Use the model to predict values on the test dataset
predicted <- predict(logistic_model, newdata = test_df, type = "response")


# Look at the probability outputs to choose a decision threshold
summary(predicted)

# Using my chosen decision threshold of 0.4, generate the predicted classes

predicted_class <- ifelse(predicted > 0.4, 1, 0)


# The first evaluation metric is accuracy, which is found to be around 64%

comparison <- data.frame(Actual = test_df$STATUS, Predicted = predicted_class)
print(mean(comparison$Actual == comparison$Predicted)*100)

# The second metric is the confusion matrix, which shows how accurate the classifier is at predicting 0s and 1s

confusion_matrix <- table(comparison$Actual, comparison$Predicted)
rownames(confusion_matrix) <- c("Actual_0", "Actual_1")
colnames(confusion_matrix) <- c("Predicted_0", "Predicted_1")

print("Confusion Matrix:")
print(confusion_matrix)

# As seen above, the model is accurate at predicting 0s but inaccurate at 1s
# 
# The ROC curve will now be output

roc_obj <- roc(test_df$STATUS, predicted)
plot(roc_obj, main = "ROC Curve for Logistic Regression Holdout (Luke)", col = "blue")
text(0.5, 0.2, auc(roc_obj), adj = c(0, 1), col = "black", cex = 1)

# As seen above, the AUC value is 0.51, which indicates a poor classifier that 
# performs only slightly better than average

# K-fold Cross Validation
# The dataset is now evaluated with 10-fold cross validation

control <- trainControl(method = "cv", number = 10)

model <- train(STATUS ~ ., data = df, method = "glm", family = "binomial", trControl = control)

# The logistic model is used to predict, this time on the original dataframe

print(model)

predictions <- predict(model, newdata = df)

# The accuracy of the model is computed


accuracy <- confusionMatrix(predictions, df$STATUS)$overall["Accuracy"]
print(paste("Accuracy:", round(accuracy * 100, 2), "%"))

# This time, the accuracy is 64.6%, which is slightly higher than the holdout method

conf_matrix <- confusionMatrix(predictions, df$STATUS)
print(conf_matrix)

# The confusion matrix reveals that this k-fold model performs worse overall than the holdout method. The 1 class is very rarely correctly predicted, with only 24 correct predictions compared to 82 from holdout.
# 
# We can also see the ROC Curve and AUC for the k-fold method

y_true <- df$STATUS

lr_probs = predict(model, newdata = df, type = "prob")[,2]
plot(roc(y_true, lr_probs), print.auc=TRUE)

# The confusion matrix reveals that this k-fold model performs worse overall than 
# the holdout method. The 1 class is very rarely correctly predicted, with only 24 correct predictions 
# compared to 82 from holdout.

# We can also see the ROC Curve and AUC for the k-fold method

y_true <- df$STATUS

lr_probs = predict(model, newdata = df, type = "prob")[,2]
plot(roc(y_true, lr_probs), print.auc=TRUE)

# As seen above, the AUC is slightly better than the holdout method. 
# However, I would still recommend using the holdout method for logistic regression
# as it predicts the positive class better.

-----------------------------------------------------------------------

##### Support Vector Machine
# For this model, only the holdout method will be used, since it is too complex to be used with k-fold.
# 
# This model uses the same train/test split as logistic regression for consistency.
# The linear kernel hyper-parameter is chosen here as it is the simplest SVM kernel, built for linear data.
svm_model <- svm(STATUS ~ ., data = train_df, kernel = "linear", probability = TRUE)

predicted_probabilities <- predict(svm_model, test_df, probability = TRUE)

mean(predicted_probabilities == test_df$STATUS)

# The SVM model has an accuracy of 64%, around the same as the previous two.


conf_matrix <- table(predicted_probabilities, test_df$STATUS)
print("Confusion Matrix:")
print(conf_matrix)

# The confusion matrix for the SVM model reveals a similar problem to the holdout logistic model, 
# with 0 being predicted with very high accuracy and 1 being predicted with very low accuracy.

roc_obj <- roc(test_df$STATUS, attr(predicted_probabilities, "probabilities")[, 2])

plot(roc_obj, col = "blue", main = "ROC Curve for SVM")
text(0.5, 0.2, auc(roc_obj), adj = c(0, 1), col = "black", cex = 1)

# The AUC in this case is just above 0.5, indicating the worst performance of all the models so far.

-------------------------------------------------------------------


# RandomForest Using Kfold


# Train control for 10-fold CV
ctrl <- trainControl(method="cv", number=2, verboseIter=TRUE) 

# Train Random Forest model
rf_model <- train(STATUS ~ ., data=df, method="rf", trControl=ctrl)
# Predict on original full data 
predictions <- predict(rf_model, new_df)
# Evaluate predictions
conf_matrix_rf <- confusionMatrix(predictions, new_df$STATUS)
print(conf_matrix_rf)
accuracy_rf <- conf_matrix_rf$overall['Accuracy']

print(paste0("Accuracy: ", round(accuracy_rf*100, 2), "%"))

print("Confusion Matrix:")
print(as.matrix(conf_matrix_rf$table))

# ROC Curve
rf_probs <- predict(rf_model, new_df, type="prob")[,2] 
rf_roc <- roc(new_df$STATUS, rf_probs)
plot(rf_roc, print.auc=TRUE)

# The AUC (area under the curve) is around 0.85, which is quite good. 
# Higher AUC indicates better model discrimination ability.
# The curve bends sharply towards the top left corner, suggesting the model 
# is effective at distinguishing between the positive and negative classes.

#visualization of confusion Matrix
cm <- conf_matrix_rf$table

# Create a data frame from the matrix
cm_df <- as.data.frame(as.table(cm))

# Plot confusion matrix
ggplot(cm_df, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), vjust = 1) +
  scale_fill_gradient(low = "white", high = "steelblue") +
  labs(x = "Actual", y = "Predicted", title = "Random Forest Confusion Matrix")

# From this confusion Matrix it is Evident that it is efficiently predicts both 0's and 1's
# with an accuracy of 78.3% .Here TP=6790,TN=19152,FP=2230 and FN=6790.

-----------------------------------------------------------------------------

 # QDA using Holdout method
 # Train the Quadratic Discriminant Analysis model
qda_model <- qda(STATUS ~ ., data = train_df)

# Make predictions on the test set
predicted_class <- predict(qda_model, newdata = test_df)$class

# Create a data frame for comparison
comparison <- data.frame(Actual = test_df$STATUS, Predicted = predicted_class)

# Print accuracy
accuracy <- mean(comparison$Actual == comparison$Predicted)
print(paste("Accuracy:", accuracy))

##QDA on Holdout method is providing an accuracy of 64%

# Create a confusion matrix
conf_matrix <- confusionMatrix(data = comparison$Predicted, reference = comparison$Actual)

# Extract confusion matrix as a table
conf_matrix_table <- as.table(conf_matrix$table)

print(conf_matrix)

# As per the confusion matrix, even though it has an accuracy of 64%, it's not predicting 1's properly.

# Plot the confusion matrix using ggplot2
ggplot(data = as.data.frame(conf_matrix_table), aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = sprintf("%d", Freq)), vjust = 1) +
  scale_fill_gradient(low = "lightblue", high = "lightcoral") +
  theme_minimal() +
  labs(title = "Confusion Matrix",
       x = "Actual",
       y = "Predicted")

# Access performance metrics
accuracy <- conf_matrix$overall["Accuracy"]
precision <- conf_matrix$byClass["Precision"]
recall <- conf_matrix$byClass["Recall"]
f1_score <- conf_matrix$byClass["F1"]
print(precision)
print(accuracy)
print(recall)
print(f1_score)

# Make predictions on the test set
predicted_probs <- predict(qda_model, newdata = test_df)$posterior[, 2]  # Probability of the positive class

# Create a ROC curve
roc_curve <- roc(test_df$STATUS, predicted_probs)

# Plot the ROC curve
plot(roc_curve, main = "Receiver Operating Characteristic (ROC) Curve for QDA", col = "blue")

# Add a legend
legend("bottomright", legend = paste("AUC =", round(auc(roc_curve), 2)), col = "black", lty = 1:2)

# AUC is only slightly above the random line
# 
# QDA using K-fold method

# Define the number of folds
num_folds <- 5

# Create a data partitioning object for k-fold cross-validation
folds <- createFolds(df$STATUS, k = num_folds, list = TRUE, returnTrain = FALSE)

# Initialize a vector to store cross-validation results
cv_results <- numeric(num_folds)

# Perform k-fold cross-validation with QDA
for (i in 1:num_folds) {
  # Split the data into training and test sets for this fold
  fold_train_data <- df[-folds[[i]], ]
  fold_test_data <- df[folds[[i]], ]
  
  # Fit QDA model
  qda_model <- qda(STATUS ~ ., data = fold_train_data)
  
  # Make predictions on the test set
  predicted_class <- predict(qda_model, newdata = fold_test_data)$class
  
  # Create a data frame for comparison
  comparison <- data.frame(Actual = fold_test_data$STATUS, Predicted = predicted_class)
  
  # Calculate accuracy for this fold
  accuracy <- mean(comparison$Actual == comparison$Predicted)
  cv_results[i] <- accuracy
}

# Display cross-validation results
print("Cross-Validation Results:")

print(mean(cv_results))

# Using K fold method the accuracy is comparable with holdout method. In both only an average performance is visible.
 

# -----------------------------------------------------------------------------

# Naive Bayes Model

# Define the Naive Bayes model
nb_model <- naiveBayes(STATUS ~ ., data = train_df, laplace = 1)

# Make predictions on the test set with probabilities
predictions_probs <- predict(nb_model, newdata = test_df, type = "raw")

# Chosen threshold
threshold <- 0.5

# Convert probabilities to binary predictions based on the threshold
binary_predictions <- ifelse(predictions_probs[, "1"] > threshold, "1", "0")

# Evaluate the model
accuracy <- sum(binary_predictions == test_df$STATUS) / length(test_df$STATUS) * 100
conf_matrix <- table(test_df$STATUS, binary_predictions)
classification_report <- confusionMatrix(as.factor(binary_predictions), as.factor(test_df$STATUS))

# Print the results
print(paste("Naive Bayes Model Accuracy: ", round(accuracy, 2), "%"))
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(classification_report)

# -----------------------------------------------------------------------------
 
# Decision Tree

# Create a training control with k-fold cross-validation
ctrl <- trainControl(method = "cv", number = 10)

# Specify the decision tree model
tree_model <- train(STATUS ~ ., data = df, method = "rpart", trControl = ctrl)

# Print the model
print(tree_model)

# Plot the decision tree
plot(tree_model$finalModel)
text(tree_model$finalModel, pretty = 0)

predictions <- predict(tree_model, newdata = df)
conf_matrix <- confusionMatrix(predictions, df$STATUS)
print(conf_matrix)

# Plot the confusion matrix
plot(conf_matrix$table, col = c("darkgreen", "darkred"),
     main = "Confusion Matrix",
     sub = paste("Accuracy =", conf_matrix$overall["Accuracy"]))

# Add numeric values to the confusion matrix plot
text(x = 1:2, y = 1:2, labels = conf_matrix$table,
     col = c("black", "blue"), cex = 2, pos = 4)