import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('epo2.csv')


reg = linear_model.LinearRegression()
reg.fit(df[['ANX']],df[['EEG_epochs']])
print(reg.predict([[10.2]]))
plt.title('Linear Regression')
plt.xlabel('Range of ANX')
plt.ylabel('Epochs')
plt.scatter(df['ANX'],df['EEG_epochs'])
plt.plot(df[['ANX']],reg.predict(df[['ANX']]),color = 'blue')
plt.grid()
plt.show()