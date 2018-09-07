import numpy as np
import matplotlib.pyplot as plt


# generate matrix with random values in the range (-10,10)
# 10 classes (rows) by 100 instances (columns)
array = 10*np.random.uniform(-1,1, (10,100))
print("ARRAY:")
print(array)

# choose the right value for each column randomly randint(0,9)
true_index = np.random.randint(0,9)
print("TRUE INDEX:")
print(true_index)

# compute SVM Loss for each and average
for i in range(0,array.size):
    for j in range(0,array[i].size):
        if i == 0 and j % 10 == 0:
            print(i,j)



# compute Softmax cross-entropy for each and average



# repeat the process 50 times



# plot the averages to the graph