# Xgboost
install.packages("drat", repos="https://cran.rstudio.com")
drat:::addRepo("dmlc")
install.packages("xgboost", repos="http://dmlc.ml/drat/", type = "source")
require(xgboost)

#Set Working Directory & Read data
getwd()
setwd("C:/Users/Chandra/Desktop/AllState")
Train <- read.csv("C:/Users/Chandra/Desktop/AllState/train_woe.csv", stringsAsFactors = TRUE)
Test <- read.csv("C:/Users/Chandra/Desktop/AllState/test_woe_v2.csv", stringsAsFactors = TRUE)

names(Train)
str(Train)
sum_train <- data.frame(summary(Train))
sum_train$Freq[1]
write.csv(sum_train, file = "sum_train.csv")

#change target name to "target_bin"
# colnames(Train[c(16)]) <- "target_bin"
# str(Train)


#Dividing the dataset on basis of CART

names(Train)

Train_cat80_h1 <- Train[Train$cat80_w < 0.265,]
Train_cat80_h2 <- Train[Train$cat80_w >= 0.265,]

Test_cat80_h1 <- Test[Test$cat80_w < 0.265,]
Test_cat80_h2 <- Test[Test$cat80_w >= 0.265,]

Train_cat80_h1$loss = log1p(Train_cat80_h1$loss)
Train_cat80_h2$loss = log1p(Train_cat80_h2$loss)


# Parameters Start


# MAE Calculation

  mae = function(target, p_target) {
  differ = abs(target- p_target)
  sum_differ <- sum(differ)
  n = length(target)
  x = sum_differ/n
  }
  

xgb_train_fn = function(df, param, nround, nf){
  
  train_matrix = as.matrix(df[,c(-1,-16)])
  
  clf <- xgb.cv(params            = param, 
              data                = train_matrix,
              label               = df[,c(16)],
              nrounds             = nround, #300, #280, #125, #250, # changed from 300
              verbose             = 1,
              #early.stop.round   = 40,
              #watchlist          = watchlist,
              #maximize            = FALSE,
              eval_metric         ="rmse",
              nfold               =nf)
}


first_model <-   xgb_train_fn(Train_cat80_h1, param, 1000, 4)
second_model <-  xgb_train_fn(Train_cat80_h2, param, 1000, 4)

min_err = function(df)
{
  k <- -999
  v <- as.data.frame(df)[,c(3)]
  for(i in 1:length(v))
  {
    s = v[i+1] - v[i]
    if (s>0)
    {
      k <- i
      
    }
    break;
  }
 return(k)
}

first_model2 <- as.data.frame(first_model)
write.csv(first_model2, "first_model2.csv")
second_model2 <- as.data.frame(second_model)
write.csv(second_model2, "second_model2.csv")

opt_n1 <- min_err(first_model)
opt_n2 <- min_err(second_model)

opt_n1
opt_n2

param <- list(objective           = "reg:linear", 
              booster             = "gbtree",
              eta                 = 0.01, # 0.06, #0.01,0.005
              max_depth           = 10, #changed from default of 4,6,8,10,15,20
              subsample           = 1, #(.5,0.7,1)
              colsample_bytree    = 0.4, #(.5,0.7,1)
              min_child_weight    = 10.29  ## 3/ Event rate - Rule of Thumb )
  )

xgb_params = list(
  seed = 0,
  colsample_bytree = 0.5,
  subsample = 0.8,
  eta = 0.05, # replace this with 0.01 for local run to achieve 1113.93
  objective = 'reg:linear',
  max_depth = 12,
  alpha = 1,
  gamma = 2,
  min_child_weight = 1,
  base_score = 7.76
)


if (opt_n1 <- - 999 )
  opt_n1 <- 500
if (opt_n2 <- - 999 )
  opt_n2 <- 550
opt_n1
opt_n2


#Best Model for fitting

xgb_best_fn = function(df, opt_n) {
clf_best <- xgboost(params              = xgb_params, 
                    data                = as.matrix(df[,c(-1,-16)]), 
                    label               = df$loss,
                    nrounds             = opt_n, #2000, #300, #280, #125, #250, # changed from 300
                    verbose             = 1
                    #early.stop.round    = 200,
                    #watchlist           = watchlist,
                    #maximize            = FALSE,
                    #eval_metric="auc"
                    #nfold=3
)
return(clf_best)
}

# Final Prediction on Train
col1 = names(Train_cat80_h1[,c(-1,-16)])
best_model_v1 <-xgb_best_fn(Train_cat80_h1,opt_n1)
p_loss <- predict(best_model_v1, as.matrix(Train_cat80_h1[,col1]))
pred_ids_v1 <- cbind(Train_cat80_h1[,c(1,16)],p_loss)
pred_ids_v1$p_loss <- expm1(pred_ids_v1$p_loss) 
pred_ids_v1$loss <- expm1(pred_ids_v1$loss) 



#mean(pred_ids_v1$loss)

col2 = names(Train_cat80_h2[,c(-1,-16)])
best_model_v2 <-xgb_best_fn(Train_cat80_h2,opt_n2)
p_loss <- predict(best_model_v2, as.matrix(Train_cat80_h2[,col2]))
pred_ids_v2 <- cbind(Train_cat80_h2[,c(1,16)],p_loss)  
pred_ids_v2$p_loss <- expm1(pred_ids_v2$p_loss) 
pred_ids_v2$loss <- expm1(pred_ids_v2$loss)
  
# Appending final predicted results
  
final_train <- rbind(pred_ids_v1,pred_ids_v2)
mae_train <- mae(final_train$loss,final_train$p_loss)
mae_train


# Final Prediction on Test
tcol1 = names(Test_cat80_h1[,c(-1)])
tp_loss <- predict(best_model_v1, as.matrix(Test_cat80_h1[tcol1]),missing =NA)
tp_loss <- expm1(tp_loss) 
tpred_ids_v1 <- cbind(Test_cat80_h1[,c(1)],tp_loss)



tcol2 = names(Test_cat80_h2[,c(-1)])
tp_loss <- predict(best_model_v2, as.matrix(Test_cat80_h2[,tcol2]), missing=NA)
tp_loss <- expm1(tp_loss) 
tpred_ids_v2 <- cbind(Test_cat80_h2[,c(1)],tp_loss)  

# Appending final predicted results
  
final_test <- rbind(tpred_ids_v1,tpred_ids_v2)
write.csv(final_test,"fin_test10.csv")

  
  













