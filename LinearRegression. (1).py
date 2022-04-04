#!/usr/bin/env python
# coding: utf-8

# In[53]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
data = pd.read_csv("Ecommerce_Customers.csv")
columns = ["Email","Address","Avatar","Avg Session Length","Time on App","Time on Website","Length of Membership","Yearly Amount Spent"]
data = pd.read_table("Ecommerce_Customers.csv", sep=',', header=None, names=columns)
X = data['Avg Session Length', 'Time on App','Time on Website', 'Length of Membership']
y = data['Yearly Amount Spent']


# In[49]:


df0


# In[50]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=.50 ,random_state=35)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,y_train)
Predict = reg.predict( X_test)


# In[51]:


from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test,Predict))
print('MSE:', metrics.mean_squared_error(y_test,Predict))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test,Predict)))


# In[52]:


plt.scatter(y_test,Predict)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')


# In[43]:


x= data['Avg Session Length']
y = data['Yearly Amount Spent']
plt.xlabel("Avg Session Length")
plt.ylabel("Yearly Amount Spent")
plt.scatter(x,y)
plt.show()


# In[42]:


x= data['Time on App']
y = data['Yearly Amount Spent']
plt.xlabel("Time on App")
plt.ylabel("Yearly Amount Spent")
plt.scatter(x,y)
plt.show()


# In[41]:


x= data['Time on Website']
y = data['Yearly Amount Spent']
plt.xlabel("Time on Website")
plt.ylabel("Yearly Amount Spent")

plt.scatter(x,y)
plt.show()


# In[40]:


x= data["Length of Membership"]
y = data['Yearly Amount Spent']
plt.xlabel("Length of Membership")
plt.ylabel("Yearly Amount Spent")

plt.scatter(x,y)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




