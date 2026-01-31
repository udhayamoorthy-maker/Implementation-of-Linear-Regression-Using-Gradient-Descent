# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the required library and read the dataframe.
2.Write a function computeCost to generate the cost function.
3.Perform iterations og gradient steps with learning rate.
4.Plot the Cost function using Gradient Descent and generate the required graph. 

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Udhayamoorthy A
RegisterNumber:  212225040477
*/
```
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("C:/Users/acer/Downloads/50_Startups.csv")

x = data["R&D Spend"].values
y = data["Profit"].values

x = (x - np.mean(x)) / np.std(x)


w = 0.0          # weight
b = 0.0          # bias
alpha = 0.01     # learning rate
epochs = 100
n = len(x)

losses = []


for i in range(epochs):
    # Prediction
    y_hat = w * x + b

    loss = np.mean((y_hat - y) ** 2)
    losses.append(loss)
    dw = (2/n) * np.sum((y_hat - y) * x)
    db = (2/n) * np.sum(y_hat - y)

    w = w - alpha * dw
    b = b - alpha * db

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(losses)
plt.xlabel("Iterations")
plt.ylabel("Loss (MSE)")
plt.title("Loss vs Iterations")

plt.subplot(1, 2, 2)
plt.scatter(x, y, label="Data")
plt.plot(x, w * x + b, label="Regression Line")
plt.xlabel("R&D Spend (scaled)")
plt.ylabel("Profit")
plt.title("Linear Regression using Gradient Descent")
plt.legend()

plt.tight_layout()
plt.show()

print("Final Weight (w):", w)
print("Final Bias (b):", b)
```
## Output:

<img width="593" height="548" alt="image" src="https://github.com/user-attachments/assets/23ebad5e-8e31-437f-9965-165a4590f792" />

<img width="443" height="521" alt="image" src="https://github.com/user-attachments/assets/bcf0b76d-1375-4d8c-9a02-cfcaf854cf15" />

Final Weight (w): 33671.51979690389
Final Bias (b): 97157.57273469678


​
## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
