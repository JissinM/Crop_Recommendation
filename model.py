import pandas as pd
import numpy as np
import pandas as pd
import pickle

data = pd.read_csv("data.csv")

print(data.head())

import warnings
warnings.filterwarnings('ignore')
x = data.iloc[:, :7].values
y = data.iloc[:, -1].values


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

prediction = model.predict((np.array([[90,
                                       40,
                                       40,
                                       20,
                                       80,
                                       7,
                                       200]])))
print("The Suggested Crop for Given Climatic Condition is :", prediction)
pickle.dump(model, open("model.pkl","wb"))


