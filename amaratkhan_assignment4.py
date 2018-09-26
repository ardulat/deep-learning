import numpy as np


def sign(value, threshold=0):
    if (value > threshold):
        return 1
    else:
        return 0


if __name__ == "__main__":
    x1 = 1
    x2 = 0
    x3 = 1
    y = 0
    b = np.random.uniform(0,0.2)
    print(b)

    # randomly initialize weights
    w1 = np.random.randint(0,10)
    w2 = np.random.randint(0,10)
    w3 = np.random.randint(0,10)
    wb = 1 # except weight for bias

    print(w1, w2, w3)

    epsilon = 0.5
    hasConverged = False

    while (hasConverged == False):
        y_pred = sign(w1*x2 + w2*x2 + w3*x3 + wb*b)
        if (y_pred != y):
            w1 = w1 - epsilon
            w2 = w2 - epsilon
            w3 = w3 - epsilon
        else:
            hasConverged = True
        
    print(w1, w2, w3)