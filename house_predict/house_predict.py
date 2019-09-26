# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 15:36:55 2018

@author: sburns2
"""


from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
import pandas as pd
import numpy as np


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from numpy import array

from numpy import argmax

#from keras.layers import Dense, Activation
#from keras.models import Sequential
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
def one_hot_ecode(data):
    
    values = array(data)
  #  print(values)
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    #print(integer_encoded)
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    print("sd",integer_encoded[2])
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    feature_name = onehot_encoder.feature_indices_
    par_names = onehot_encoder._get_param_names
    print("act",onehot_encoder.active_features_)

    #  print(onehot_encoded)
    # invert first example
    #inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])
    inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])
 #   print(inverted)
    return onehot_encoded,inverted,feature_name,par_names

#A function to calculate Root Mean Squared Logarithmic Error (RMSLE)
def rmsle(y_pred, y_test) : 
    assert len(y_test) == len(y_pred)
    return np.sqrt(np.mean((np.log(1+y_pred) - np.log(1+y_test))**2))
def NN(X_train,X_test,y_train,n,X2):
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    X2 = sc.transform(X2)
    # Initialising the ANN
    model = Sequential()
    # Adding the input layer and the first hidden layer
    print(len(X_train))
    model.add(Dense(32, activation = 'relu', input_dim = n))
    # Adding the second hidden layer 
    model.add(Dense(units = 32, activation = 'relu'))
    # Adding the third hidden layer
    model.add(Dense(units = 32, activation = 'relu'))
   # Adding the output layer
    model.add(Dense(units = 1))
   #model.add(Dense(1))
   # Compiling the ANN
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')

   # Fitting the ANN to the Training set
    model.fit(X_train, y_train, batch_size = 10, epochs = 38)
    
    return model,X_test,X2

def main():
   
    df = pd.read_csv("./Data/train.csv.csv")
    df_test = pd.read_csv("./Data/test.csv.csv")
    print(df_test)
   
    
    X = df[['Foundation','RoofMatl','GarageQual','GarageYrBlt','ExterQual','YearBuilt','OverallQual','OverallCond','KitchenQual','GarageType','SaleType','YrSold','MiscVal','PavedDrive','MoSold','Fireplaces','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','Id','1stFlrSF','2ndFlrSF','GrLivArea','TotRmsAbvGrd','BedroomAbvGr','BldgType','Neighborhood','Street','LandSlope','MSZoning','LotConfig','MSSubClass','LotFrontage','LotArea','PoolArea','SalePrice','SaleCondition','HouseStyle']]
    #print("maxporch",X['OpenPorchSF'].min())
    X = X.dropna()
    X['Age'] = 2010 - (X['YearBuilt']) 
    X['GarageAge'] = 2010 - (X['GarageYrBlt']) 
  
    X = pd.concat([X, pd.get_dummies(X['HouseStyle'], prefix='HouseStyle')], axis=1)
    X = pd.concat([X, pd.get_dummies(X['LotConfig'], prefix='Lot')], axis=1)
    X = pd.concat([X, pd.get_dummies(X['LandSlope'], prefix='Slope')], axis=1)
    X = pd.concat([X, pd.get_dummies(X['MSZoning'], prefix='Zone')], axis=1)
    X = pd.concat([X, pd.get_dummies(X['Street'], prefix='St')], axis=1)
    X = pd.concat([X, pd.get_dummies(X['Neighborhood'], prefix='Ne')], axis=1)
    X = pd.concat([X, pd.get_dummies(X['BldgType'], prefix='Bld')], axis=1)
    X = pd.concat([X, pd.get_dummies(X['MoSold'], prefix='Mo')], axis=1)
    X = pd.concat([X, pd.get_dummies(X['SaleCondition'], prefix='SC')], axis=1)
    X = pd.concat([X, pd.get_dummies(X['PavedDrive'], prefix='PD')], axis=1)
    X = pd.concat([X, pd.get_dummies(X['YrSold'], prefix='YS')], axis=1)
    X = pd.concat([X, pd.get_dummies(X['SaleType'], prefix='ST')], axis=1)
    X = pd.concat([X, pd.get_dummies(X['GarageType'], prefix='GT')], axis=1)
    X = pd.concat([X, pd.get_dummies(X['KitchenQual'], prefix='KQ')], axis=1)
    X = pd.concat([X, pd.get_dummies(X['ExterQual'], prefix='EQ')], axis=1)
    X = pd.concat([X, pd.get_dummies(X['GarageQual'], prefix='GQ')], axis=1)
    X = pd.concat([X, pd.get_dummies(X['RoofMatl'], prefix='RM')], axis=1)
    X = pd.concat([X, pd.get_dummies(X['Foundation'], prefix='F')], axis=1)
  
    
    X2 = df_test[['Foundation','RoofMatl','GarageQual','GarageYrBlt','ExterQual','YearBuilt','OverallQual','OverallCond','KitchenQual','GarageType','SaleType','YrSold','MiscVal','PavedDrive','SaleCondition','MoSold','Fireplaces','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','Id','1stFlrSF','2ndFlrSF','GrLivArea','TotRmsAbvGrd','BedroomAbvGr','BldgType','Neighborhood','Street','LandSlope','MSZoning','LotConfig','MSSubClass','LotFrontage','LotArea','PoolArea','HouseStyle']]
    print("X2",X2)
    X2['Age'] = 2010 - (X2['YearBuilt'])
    X2['GarageAge'] = 2010 - (X2['GarageYrBlt']) 
    print("____________")
    print("X3",X2)
    X2_Id = X2['Id']
    #X2 = X2.dropna()
    X2 = X2.fillna(X2.mean())
    print("X2",X2)
    X2 = pd.concat([X2, pd.get_dummies(X2['HouseStyle'], prefix='HouseStyle')], axis=1)
    X2 = pd.concat([X2, pd.get_dummies(X2['LotConfig'], prefix='Lot')], axis=1)
    X2 = pd.concat([X2, pd.get_dummies(X2['LandSlope'], prefix='Slope')], axis=1)
    X2 = pd.concat([X2, pd.get_dummies(X2['MSZoning'], prefix='Zone')], axis=1)
    X2 = pd.concat([X2, pd.get_dummies(X2['Street'], prefix='St')], axis=1)
    X2 = pd.concat([X2, pd.get_dummies(X2['Neighborhood'], prefix='Ne')], axis=1)
    X2 = pd.concat([X2, pd.get_dummies(X2['BldgType'], prefix='Bld')], axis=1)
    X2 = pd.concat([X2, pd.get_dummies(X2['MoSold'], prefix='Mo')], axis=1)
    X2 = pd.concat([X2, pd.get_dummies(X2['PavedDrive'], prefix='PD')], axis=1)
    X2 = pd.concat([X2, pd.get_dummies(X2['SaleCondition'], prefix='SC')], axis=1)
    X2 = pd.concat([X2, pd.get_dummies(X2['YrSold'], prefix='YS')], axis=1)
    X2 = pd.concat([X2, pd.get_dummies(X2['SaleType'], prefix='ST')], axis=1)
    X2 = pd.concat([X2, pd.get_dummies(X2['GarageType'], prefix='GT')], axis=1)
    X2 = pd.concat([X2, pd.get_dummies(X2['KitchenQual'], prefix='KQ')], axis=1)
    X2 = pd.concat([X2, pd.get_dummies(X2['ExterQual'], prefix='EQ')], axis=1)
    X2 = pd.concat([X2, pd.get_dummies(X2['GarageQual'], prefix='GQ')], axis=1)
    X2 = pd.concat([X2, pd.get_dummies(X2['RoofMatl'], prefix='RM')], axis=1)
    X2 = pd.concat([X2, pd.get_dummies(X2['Foundation'], prefix='F')], axis=1)
    
    print("\n")
   # print("X2 Mo",X2['SaleCondition'].unique())
    #print(X['SaleCondition' ])
    X_A = X['Foundation' ].unique()
    X_B = X2['Foundation'].unique()
    
    common = []
    for i in X_A:
        if i in X_B:
            common.append(i)
    print("\n --",common)        
    
    #encoded_hst,inverted_hst = one_hot_ecode(X2['HouseStyle'])
    #print(X2['HouseStyle'].unique())
    Y = X[['SalePrice']]
    labels = np.array(Y)
    labels = labels.ravel()
    
    X_forcorr = X[['F_PConc', 'F_CBlock', 'F_BrkTil', 'F_Wood', 'F_Slab', 'F_Stone','RM_CompShg', 'RM_WdShngl', 'RM_WdShake', 'RM_Tar&Grv','GQ_TA', 'GQ_Fa', 'GQ_Gd', 'GQ_Po','GarageAge','EQ_Gd', 'EQ_TA', 'EQ_Ex', 'EQ_Fa','Age','OverallQual','OverallCond','KQ_Gd', 'KQ_TA', 'KQ_Ex', 'KQ_Fa','GT_Attchd','GT_Detchd','GT_BuiltIn','GT_CarPort','ST_WD','ST_New','ST_COD','ST_ConLD','ST_ConLI','ST_CWD','ST_ConLw','ST_Con','ST_Oth','YS_2008','YS_2007','YS_2009','YS_2006','YS_2010','MiscVal','PD_Y','PD_N','PD_P','SC_Normal','SC_Abnorml','SC_Partial','SC_AdjLand','SC_Alloca','SC_Family','Mo_1','Mo_2','Mo_3','Mo_4','Mo_5','Mo_6','Mo_7','Mo_8','Mo_9','Mo_10','Mo_11','Mo_12','MoSold','Fireplaces','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','TotRmsAbvGrd','2ndFlrSF','1stFlrSF','GrLivArea','BedroomAbvGr','Bld_1Fam', 'Bld_2fmCon', 'Bld_Duplex', 'Bld_TwnhsE', 'Bld_Twnhs','Ne_CollgCr', 'Ne_Veenker', 'Ne_Crawfor', 'Ne_NoRidge', 'Ne_Mitchel', 'Ne_Somerst', 'Ne_OldTown', 'Ne_BrkSide', 'Ne_Sawyer', 'Ne_NridgHt', 'Ne_SawyerW', 'Ne_NAmes', 'Ne_IDOTRR', 'Ne_MeadowV', 'Ne_Edwards', 'Ne_Timber', 'Ne_StoneBr', 'Ne_ClearCr', 'Ne_Gilbert', 'Ne_NWAmes', 'Ne_NPkVill', 'Ne_Blmngtn', 'Ne_BrDale', 'Ne_SWISU', 'Ne_Blueste','St_Pave','St_Grvl','Zone_RL','Zone_RM','Zone_FV','Zone_RH','Zone_C (all)','MSSubClass','Slope_Gtl','Slope_Mod','Slope_Sev','Lot_Inside','Lot_Corner','Lot_CulDSac','Lot_FR3','LotFrontage','LotArea','PoolArea','HouseStyle_SLvl','HouseStyle_SFoyer',  'HouseStyle_2.5Unf', 'HouseStyle_2Story','HouseStyle_1Story','SalePrice']]

    X = X[['F_PConc', 'F_CBlock', 'F_BrkTil', 'F_Wood', 'F_Slab', 'F_Stone','RM_CompShg', 'RM_WdShngl', 'RM_WdShake','GQ_TA', 'GQ_Fa', 'GQ_Gd','GarageAge','EQ_Gd', 'EQ_TA', 'EQ_Ex', 'EQ_Fa','Age','OverallQual','OverallCond','KQ_Gd', 'KQ_TA', 'KQ_Ex', 'KQ_Fa','GT_Attchd','GT_Detchd','GT_BuiltIn','GT_CarPort','ST_COD','PD_P','SC_Normal','SC_Abnorml','SC_Partial','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','Bld_1Fam','GrLivArea','2ndFlrSF','1stFlrSF','TotRmsAbvGrd','BedroomAbvGr', 'Bld_2fmCon', 'Bld_Duplex', 'Bld_TwnhsE', 'Bld_Twnhs','Ne_CollgCr', 'Ne_Veenker', 'Ne_Crawfor', 'Ne_NoRidge', 'Ne_Mitchel', 'Ne_Somerst', 'Ne_OldTown', 'Ne_BrkSide', 'Ne_Sawyer', 'Ne_NridgHt', 'Ne_SawyerW', 'Ne_NAmes', 'Ne_IDOTRR', 'Ne_MeadowV', 'Ne_Edwards', 'Ne_Timber', 'Ne_StoneBr', 'Ne_ClearCr', 'Ne_Gilbert', 'Ne_NWAmes', 'Ne_NPkVill', 'Ne_Blmngtn', 'Ne_BrDale', 'Ne_SWISU', 'Ne_Blueste','St_Pave','St_Grvl','Zone_RL','Zone_RM','Zone_FV','Zone_RH','Zone_C (all)','MSSubClass','Slope_Gtl','Slope_Mod','Slope_Sev','Lot_Inside','Lot_Corner','Lot_CulDSac','Lot_FR3','LotFrontage','LotArea','PoolArea','HouseStyle_SLvl','HouseStyle_SFoyer',  'HouseStyle_2.5Unf', 'HouseStyle_2Story','HouseStyle_1Story']]
    col_num = len(X.columns)
    print("features",col_num)
    X2 = X2[['F_PConc', 'F_CBlock', 'F_BrkTil', 'F_Wood', 'F_Slab', 'F_Stone','RM_CompShg', 'RM_WdShngl', 'RM_WdShake','GQ_TA', 'GQ_Fa', 'GQ_Gd', 'GarageAge','EQ_Gd', 'EQ_TA', 'EQ_Ex', 'EQ_Fa',  'Age','OverallQual','OverallCond','KQ_Gd', 'KQ_TA', 'KQ_Ex', 'KQ_Fa','GT_Attchd','GT_Detchd','GT_BuiltIn','GT_CarPort','ST_COD','PD_P','SC_Normal','SC_Abnorml','SC_Partial','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','Bld_1Fam','GrLivArea','2ndFlrSF','1stFlrSF','TotRmsAbvGrd','BedroomAbvGr', 'Bld_2fmCon', 'Bld_Duplex', 'Bld_TwnhsE', 'Bld_Twnhs','Ne_CollgCr', 'Ne_Veenker', 'Ne_Crawfor', 'Ne_NoRidge', 'Ne_Mitchel', 'Ne_Somerst', 'Ne_OldTown', 'Ne_BrkSide', 'Ne_Sawyer', 'Ne_NridgHt', 'Ne_SawyerW', 'Ne_NAmes', 'Ne_IDOTRR', 'Ne_MeadowV', 'Ne_Edwards', 'Ne_Timber', 'Ne_StoneBr', 'Ne_ClearCr', 'Ne_Gilbert', 'Ne_NWAmes', 'Ne_NPkVill', 'Ne_Blmngtn', 'Ne_BrDale', 'Ne_SWISU', 'Ne_Blueste','St_Pave','St_Grvl','Zone_RL','Zone_RM','Zone_FV','Zone_RH','Zone_C (all)','MSSubClass','Slope_Gtl','Slope_Mod','Slope_Sev','Lot_Inside','Lot_Corner','Lot_CulDSac','Lot_FR3','LotFrontage','LotArea','PoolArea','HouseStyle_SLvl','HouseStyle_SFoyer',  'HouseStyle_2.5Unf', 'HouseStyle_2Story','HouseStyle_1Story']]
    X_corr = X_forcorr.corr()
    #sns.heatmap(X_corr)
    corr_matrix = X_forcorr.corr().abs()
    sol = ( corr_matrix.where( np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool)).stack().sort_values(ascending=False)  ) 
    print(sol)
    print(X_corr['SalePrice'])
  
    
    X_train, X_test , y_train , y_test = train_test_split(X,labels,test_size=0.15,random_state=32)
   
    model = RandomForestRegressor(n_estimators = 400, random_state = 42,max_features = 45,max_depth =13) 
    model.fit(X_train, y_train.ravel())
   
    predictions = model.predict(X_test)
    predictions2 = model.predict(X2)

    print("types",type(y_test),type(predictions))
    r2 = (r2_score(y_test, predictions))
    err = rmsle(y_test, predictions)
    np.savetxt("submission.csv",np.c_[X2_Id,predictions2],fmt = "%s",delimiter = ',')
    print("r2",r2)
    print("rmsle",err)
    print(len(labels),len(predictions))
    plt.scatter(y_test,predictions)
    plt.xlabel('True values')
    plt.ylabel('Predictions')
    plt.show
    
   
         
    
    
main()    