import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


veriler = pd.read_csv("satisverileri.csv")
satislar = veriler[["Satislar"]]
aylar=veriler[["Aylar"]]

x_train,x_test,y_train,y_test = train_test_split(aylar,satislar,test_size=0.33,random_state=0)
standartScaler = StandardScaler()

#Yapmaya Gerek Yok Ama Yapılabilir
X_train = standartScaler.fit_transform(x_train)
X_test = standartScaler.fit_transform(x_test)
Y_train= standartScaler.fit_transform(y_train)
Y_test = standartScaler.fit_transform(y_test)

lr= LinearRegression()
lr.fit(x_train,y_train)

predictions=lr.predict(x_test)
x_train=x_train.sort_index()
y_train=y_train.sort_index()

#Şema Çizilmeden Önce Verilerin Sort Edilmesi Gerekli
plt.plot(x_train,y_train)
plt.plot(x_test,lr.predict(x_test))












