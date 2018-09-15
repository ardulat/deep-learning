import numpy as np


def activate(value, threshold=0):
    if (value > threshold):
        return 1
    else:
        return 0

if __name__ == "__main__":
    
    # STEP 1: Initialize weights
    M = np.zeros((8,8))
    # precomputed weights
    w1 = w2 = w3 = w4 = 1
    w5 = -0.5
    w6 = -1.5
    w7 = 1
    w8 = -1
    w9 = -0.5
    b1 = b2 = b3 = 1
    # insert weights (and their indications = 999) to matrix
    M[4,0] = w1
    M[0,4] = 999

    M[5,0] = w2
    M[0,5] = 999

    M[4,1] = w3
    M[1,4] = 999

    M[5,1] = w4
    M[1,5] = 999

    M[4,2] = w5
    M[2,4] = 999

    M[5,3] = w6
    M[3,5] = 999

    M[7,4] = w7
    M[4,7] = 999

    M[7,5] = w8
    M[5,7] = 999

    M[7,6] = w9
    M[6,7] = 999

    # STEP 2: read dataset (XOR)
    X = [[0,0],
         [0,1],
         [1,0],
         [1,1]]
    Y = [0,
         1,
         1,
         0]

    # STEP 3: forward-propagate and verify
    trues = 0
    for x, y in zip(X, Y):
        nodes = [x[0], x[1], b1, b2, None, None, b3, None] # order of nodes in adjacency matrix
        
        for i in range(len(nodes)):
            if (nodes[i] == None):
                sigma = 0
                for j in range(len(M[0])):
                    if (M[j,i] == 999):
                        sigma += nodes[j] * M[i,j]
                nodes[i] = activate(sigma)

        y_predicted = nodes[len(nodes)-1]

        # verify
        if (y == y_predicted):
            trues += 1
            print('SUCCESS!')
        else:
            print('FAIL!')
                
    accuracy = (trues / len(Y)) * 100
    print("Accuracy is %.2f percent." % accuracy)
