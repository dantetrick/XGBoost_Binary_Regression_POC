# Set Project Parameters
gc()
options(scipen = 999)

# Load Packages
pkgs <- sapply(c("dplyr", "data.table", "plotROC","car","caret", "glmnet", "xgboost"), function(x) {require(x, character.only = T)})

# Load data
filesToLoad <- lapply(list.files("~/BCG Test ML", pattern = ".csv", full.names = T), function(x) {fread(x)})

# Bind All data and mix up
set.seed(8675309)
dt <- rbindlist(filesToLoad) %>% 
  sample_frac(1) %>% 
  select(-year)

# Separate Defaults and create a test and train data sets
dt_default <- dt[default == 1, ]

# Scatterplot
defaultScatters <- scatterplotMatrix(dt_default %>% select(creditScore, houseAge, yearsEmploy, ccDebt))

# Keep all but 50 random default for training data sets
dt_default_train <- dt_default[1:(nrow(dt_default)-50), ] %>% select(creditScore, houseAge, yearsEmploy, ccDebt, default)

# Keep th 50 random default for test data sets
dt_default_test <- dt_default[((nrow(dt_default)-49):nrow(dt_default)), ] %>% select(creditScore, houseAge, yearsEmploy, ccDebt, default)

############################################################################################################################################

# Separate the nondefault 
dt_nondefault <- dt[default == 0, ]

# Keep 4x as many non default points for testing later
dt_nondefault_test <- dt_nondefault[1:200,]

############################################################################################################################################

# Create full test data for later
dt_test <- bind_rows(dt_default_test, dt_nondefault_test) %>% 
  select(creditScore, houseAge, yearsEmploy, ccDebt, default) %>% 
  sample_frac(1) %>% 
  as.matrix()

xgbMatrix_test <- xgb.DMatrix(data = dt_test[,1:4]
                              , label = dt_test[,5]
                              )

############################################################################################################################################

# Remove all testing points from nondefault to create overall training set 
dt_nondefault_train <- dt_nondefault[201:nrow(dt_nondefault),]

# Create 1st sample data set
set.seed(8675309)
dt_nondefault_samp <- dt_nondefault_train %>% 
  sample_n(nrow(dt_default)*4) %>% 
  select(creditScore, houseAge, yearsEmploy, ccDebt, default) 

############################################################################################################################################

# Create Scatters
nondefaultScatters_samp <- scatterplotMatrix(dt_nondefault_samp %>% select(-default))

############################################################################################################################################

# Create Train Set
set.seed(8675309)
dt_train <- bind_rows(dt_nondefault_samp, dt_default_train) %>% 
  sample_frac(1) %>% 
  as.matrix()


xgbMatrix_train <- xgb.DMatrix(dt_train[, 1:4], label = dt_train[,5])

############################################################################################################################################

cv <- xgb.cv(data = xgbMatrix_train
             , nrounds = 1000
             , nfold = 4
             , metrics = list("rmse","auc")
             , max_depth = 4
             , eta = .01
             , objective = "binary:logistic"
             )

# Print CV
print(cv)

# Pick the best CV
NROUNDS <- cv$evaluation_log[, which(cv$evaluation_log$test_auc_mean == max(cv$evaluation_log$test_auc_mean))[1]]

# Create Model using NRounds from CV
XGBoost_Model <-   xgboost(data = xgbMatrix_train
                           , nfold = 4
                           , max.depth = 4
                           , nrounds = NROUNDS
                           , nthread = 4
                           , metrics = list("rmse","auc")
                           , eta = .1
                           , objective = "binary:logistic")
                          
############################################################################################################################################

# Predict Model
dt_test <- dt_test %>% tbl_df() %>% 
  mutate(prediction = predict(XGBoost_Model
                            , newdata = xgbMatrix_test
                            ))


ggplot(dt_test, aes(d = default, m = prediction)) + geom_roc() + theme_bw()

############################################################################################################################################

# Set cutcut
Sequences = seq(0,1, by = .01)
# x <- Sequences[[1]]
GridConfuser <- lapply(Sequences, function(x) {
  
  
  dt <- dt_test %>% 
    mutate(test = ifelse(prediction >= x, 1, 0)
           , confuser = ifelse(test == 1 & default == 1, "Correct_Default",
                             ifelse(test == 0 & default == 0, "Correct_Non_Default",
                                    ifelse(test == 1 & default == 0, "False_Positive",
                                           ifelse(test == 0 & default == 1, "False_Negative",NA))))
    )
  
  ConfusionMatrix <- dt %>% 
    group_by(confuser) %>% 
    summarize(Count = n()) %>%
    t() %>% 
    data.frame(., row.names = NULL) 
  
    names(ConfusionMatrix) <- as.character(unlist(ConfusionMatrix[1,]))
  
    ConfusionMatrix <- ConfusionMatrix[-1, ] %>% 
    mutate(Sequence = x) 
    

plotme <- ggplot(dt, aes(x = confuser)) +
  geom_bar()
  

output <- list(data = dt
               , plot = plotme
, table = ConfusionMatrix)

})


############################################################################

dt_ConfusionGrid <- rbindlist(lapply(GridConfuser, function(x){x[[3]]}), use.names = T, fill = T) %>% 
  tbl_df() %>% 
  mutate_all(funs(ifelse(is.na(.), 0, as.numeric(as.character(.))))) %>%
  mutate(Errors = (as.numeric(False_Positive) + as.numeric(False_Negative)) ) %>%
  select(Correct_Default, Correct_Non_Default, False_Positive, False_Negative, Errors, Sequence) %>% 
  arrange(Errors)

############################################################################################################################################

cutLine <- as.numeric(unlist(dt_ConfusionGrid[1,"Sequence"]))

dt_test <- dt_test %>%
  mutate(test = ifelse(prediction >= cutLine, "default", "non_default")
         , default = ifelse(default == 1, "default", "non_default"))

xtab <- confusionMatrix(table(dt_test$default, dt_test$test))



