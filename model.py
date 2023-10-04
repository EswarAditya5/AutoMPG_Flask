import pandas as pd
import pickle
mp=pd.read_csv('Auto MPG Reg.csv')
mp.horsepower=mp.horsepower.fillna(mp.horsepower.median())

y=mp.mpg
X=mp[['cylinders','displacement','horsepower','weight','acceleration','modelyear','origin']]
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X,y)
reg.score(X,y)

pickle.dump(reg,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))