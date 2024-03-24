import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import sklearn
import warnings

warnings.filterwarnings('ignore')

data = pd.read_csv('./train-2.csv')
print(data.head(5))
print(data.columns)
print(data.info())

#some data analytics:-
print(data['current price'].describe())
# plt1 = sns.heatmap(data.corr(),cmap='RdYlGn',linewidths=0.2,annot=True,annot_kws={'size':10})
# plt.show()  #current price has strong negative co relation with kms travelled by the car.
#positive co relation:- on road now, on road old prices.

# plt2 = px.scatter(data,x='on road old', y='on road now')
# plt2.show()
#the price of the car has been increased.

# plt3 = px.scatter(data,x='on road now',y='current price')  #current price has been decreased from now
# plt3.show()

# plt4 = px.scatter(data,x='km',y='current price')
# plt4.show()  #km vary linearly with the current price having negative slope.(beneficial for prediction)

# plt5 = px.scatter(data,x='condition',y='current price')
# plt5.show()
# plt7 = sns.pairplot(data[['years', 'km', 'rating', 'condition',
#        'economy', 'top speed', 'hp', 'torque','current price']],diag_kind='kde')
# plt.show()


import random
random.seed(42)
from sklearn.model_selection import train_test_split
train_df,val_df = train_test_split(data,test_size=0.2,random_state=42)
print(train_df)
input_cols = ['on road old', 'on road now', 'years', 'km', 'rating',
       'condition', 'economy', 'top speed', 'hp', 'torque']
target_cols = ['current price']
train_inputs = train_df[input_cols]
train_targets = train_df[target_cols]
val_inputs = val_df[input_cols]
val_targets = val_df[target_cols]
print(train_inputs.columns)
print(train_targets)


#building a linear regression model:-
class mean_regressor:
    def fit(self,inputs,targets):
        self.mean = targets.mean()
    def predict(self,inputs):
        return np.full(inputs.shape[0],self.mean)
from sklearn.metrics import mean_squared_error
def rmse(targets,preds):
    return mean_squared_error(targets,preds,squared=False)
model = mean_regressor()
model.fit(train_inputs,train_targets)
pred = model.predict(train_inputs)
mse = rmse(train_targets,pred)  #model is pretty bad because of high mse value
print(mse)

from sklearn.linear_model import LinearRegression
model1 = LinearRegression()
model1.fit(train_inputs,train_targets)
pred1 = model1.predict(train_inputs)
pred2 = model1.predict(val_inputs)
mse1 = rmse(pred1,train_targets)
mse2 = rmse(pred2,val_targets)
print(mse1,mse2)

from sklearn.ensemble import RandomForestRegressor
model2 = RandomForestRegressor(n_estimators=50,n_jobs=-1,max_depth=200)
model2.fit(train_inputs,train_targets)
pred3 = model2.predict(train_inputs)
pred4 = model2.predict(val_inputs)
mse3 = rmse(pred3,train_targets)
mse4 = rmse(pred4,val_targets)
print(mse3,mse4)

from sklearn.ensemble import HistGradientBoostingRegressor
model3 = HistGradientBoostingRegressor()
model3.fit(train_inputs,train_targets)
pred5 = model3.predict(train_inputs)
pred6 = model3.predict(val_inputs)
mse5 = rmse(pred5,train_targets)
mse6 = rmse(pred6,val_targets)
print(mse5,mse6)

import xgboost as xgb
model4 = xgb.XGBRegressor()
model4.fit(train_inputs,train_targets)
pred7 = model4.predict(train_inputs)
pred8 = model4.predict(val_inputs)
mse7 = rmse(pred7,train_targets)
mse8 = rmse(pred8,val_targets)
print(mse7,mse8)

from sklearn.svm import SVR
model5 = SVR()
model5.fit(train_inputs,train_targets)
pred9 = model5.predict(train_inputs)
pred10 = model5.predict(val_inputs)
mse9 = rmse(pred9,train_targets)
mse10 = rmse(pred10,val_targets)
print(mse9,mse10)  #worse than our mean regressor model

from sklearn.neighbors import KNeighborsRegressor
model6 = KNeighborsRegressor()
model6.fit(train_inputs,train_targets)
pred11 = model6.predict(train_inputs)
pred12 = model6.predict(val_inputs)
mse11 = rmse(pred11,train_targets)
mse12 = rmse(pred12,val_targets)
print(mse11,mse12)

from sklearn.ensemble import VotingRegressor
model7 = VotingRegressor([('model1',model1),('model3',model3),('model4',model4)])
model7.fit(train_inputs,train_targets)
pred13 = model7.predict(train_inputs)
pred14 = model7.predict(val_inputs)
mse13 = rmse(pred13,train_targets)
mse14 = rmse(pred14,val_targets)
print(mse13,mse14)
#lowest train mse :- 3031.55189922806, lowest val mse :- 8888.890404244054

#single entry output :-
# user_input = input('input the entries separated by commas : ')
# single_entry = list(map(float, user_input.split(',')))
# single_entry = np.array(single_entry).reshape(1, -1)
# pred_exp = model7.predict(single_entry)
# print(pred_exp)


#using tensorflow:-
import tensorflow as tf
# from tensorflow.keras.layers import Normalization, Dense, InputLayer
# from tensorflow.keras.losses import MeanSquaredError, Huber, MeanAbsoluteError
# from tensorflow.keras.metrics import RootMeanSquaredError
# from tensorflow.keras.optimizers import Adam
from keras.src.layers import InputLayer, Normalization, Dense
from keras.src.losses import MeanAbsoluteError
from keras.src.metrics import RootMeanSquaredError
from keras.src.optimizers import Adam


tensor_data = tf.constant(data)
tensor_data = tf.cast(tensor_data,dtype= tf.float32)
tensor_data = tf.random.shuffle(tensor_data)
print(tensor_data[0:5])
X = tensor_data[:,3:11]
print(X[:5])
Y = tensor_data[:,11]
print(Y[:5])
Y = tf.expand_dims(Y,axis=1)
print(Y.shape)
#Normalization Process {(value-mean)/standard deviation, std**2 = variance}
normalizer = Normalization(axis= 1)
# x_normalized = tf.constant([[1,2,3,4],  #sample example
#                            [4,5,6,7]])
# normalizer.adapt(x_normalized)
# x_normalized = normalizer(x_normalized)
# print(x_normalized)
normalizer.adapt(X)
X = normalizer(X)
print(X[:5])
# from tensorflow.keras.utils import plot_model
model_1 = tf.keras.Sequential([InputLayer(input_shape= (8,)),normalizer,Dense(1)])
print(model_1.summary())
# plot_1 = plot_model(model_1,to_file='model_1.png',show_shapes=True)
# plt.show()
model_1.compile(optimizer = Adam(learning_rate = 1.),loss = MeanAbsoluteError(),metrics = [RootMeanSquaredError()])
history = model_1.fit(X,Y,epochs = 100,verbose = 1) #epochs means the number of times it is going through gradient descent
print(history.history)

plt1 = sns.scatterplot(history.history['loss'])
plt2 = px.scatter(history.history,y='root_mean_squared_error')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.title('loss function graph')

plt.show()
plt2.show()

#performance measurement

print(model_1.evaluate(X,Y))





