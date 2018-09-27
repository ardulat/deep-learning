import numpy as np
import matplotlib.pyplot as plt

"""
    The assignment was implemented by Anuar Maratkhan.
    Important note: the adjacency matrix has not been used
    for clearly understanding of backpropagation on code.
    The code can be extended to using either adjacency matrix
    or vector of weights, however, it was decided to simplify
    the process of backpropagation on the code you see below,
    along with the process of evaluation of the assignment.
    Thanks for your understanding, and sorry for any inconvenience.
"""


# activation function
def sigmoid(x, derivative=False):
  if (derivative == True):
    return x*(1-x)
  else:
    return 1/(1+np.exp(-x))

if __name__ == "__main__":
    
    for i in range(10):
        
        print("EXPERIMENT #%d:" % (i+1))

        # STEP 1: Initialize weights
        M = np.zeros((10,10))

        # randomly initialize weights
        w1 = np.random.uniform(0.001,0.5)
        w2 = np.random.uniform(0.001,0.5)
        w3 = np.random.uniform(0.001,0.5)
        w4 = np.random.uniform(0.001,0.5)
        w5 = np.random.uniform(0.001,0.5)
        w6 = np.random.uniform(0.001,0.5)
        w7 = np.random.uniform(0.001,0.5)
        w8 = np.random.uniform(0.001,0.5)
        w9 = np.random.uniform(0.001,0.5)
        w10 = np.random.uniform(0.001,0.5)
        w11 = np.random.uniform(0.001,0.5)
        w12 = np.random.uniform(0.001,0.5)
        w13 = 0.25 # b1 = 0.25
        w14 = 0.25 # b1 = 0.25
        w15 = 0.45 # b2 = 0.45
        w16 = 0.45 # b2 = 0.45

        W = [w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12]

        # Adjacency matrix (not used in the implementation)
        M[3,0] = w1
        M[0,3] = 999
        M[3,1] = w2
        M[1,3] = 999
        M[4,0] = w3
        M[0,4] = 999
        M[4,1] = w4
        M[1,4] = 999
        M[3,2] = 1 # weights for bias
        M[2,3] = 999
        M[4,2] = 1 # weights for bias
        M[2,4] = 999

        M[6,3] = w5
        M[3,6] = 999
        M[6,4] = w6
        M[4,6] = 999
        M[7,3] = w7
        M[3,7] = 999
        M[7,4] = w8
        M[4,7] = 999
        M[6,5] = 1 # weights for bias
        M[5,6] = 999
        M[7,5] = 1 # weights for bias
        M[5,7] = 999

        M[8,6] = w9
        M[6,8] = 999
        M[8,7] = w10
        M[7,8] = 999
        M[9,6] = w11
        M[6,9] = 999
        M[9,7] = w12
        M[7,9] = 999

        # STEP 2: read dataset (XOR)
        X = [0.2,0.6]
        Y = [0.1,0.9]

        # STEP 3: forward-propagate and verify
        epochs = 10001 # 1 added for convenient print of convergence
        errors = []

        for step in range(epochs):
            
            # Forward propagate
            # LAYER 1
            net_h1 = w1 * X[0] + w2 * X[1] + w13
            out_h1 = sigmoid(net_h1)

            net_h2 = w3 * X[0] + w4 * X[1] + w14
            out_h2 = sigmoid(net_h1)

            # LAYER 2
            net_h3 = w5 * out_h1 + w6 * out_h2 + w15
            out_h3 = sigmoid(net_h3)

            net_h4 = w7 * out_h1 + w8 * out_h2 + w16
            out_h4 = sigmoid(net_h4)

            # OUTPUT LAYER
            net_o1 = w9 * out_h3 + w10 * out_h4
            out_o1 = sigmoid(net_o1)

            net_o2 = w11 * out_h3 + w12 * out_h4
            out_o2 = sigmoid(net_o2)


            # Calculating the Total Error
            target_o1 = Y[0]
            target_o2 = Y[1]
            E_o1 = 1/2 * (target_o1 - out_o1)**2
            E_o2 = 1/2 * (target_o2 - out_o2)**2
            E_total = E_o1 + E_o2
            errors.append(E_total)


            if (step) % 1000 == 0:
                print("EPOCH %d\t\t\tERROR: %.30f" % (step, E_total))


            # Backward propagate
            alpha = 0.3 # learning rate
            # TO-DO: add decay rate

            # OUTPUT LAYER
            delta_out_o1 = out_o1 - target_o1
            delta_net_o1 = sigmoid(out_o1, derivative=True)
            # w9
            delta_w9 = out_h3
            delta_w9 = delta_out_o1 * delta_net_o1 * delta_w9
            # w10
            delta_w10 = out_h4
            delta_w10 = delta_out_o1 * delta_net_o1 * delta_w10

            delta_out_o2 = out_o2 - target_o2
            delta_net_o2 = sigmoid(out_o2, derivative=True)
            # w11
            delta_w11 = out_h3
            delta_w11 = delta_out_o2 * delta_net_o2 * delta_w11
            # w12
            delta_w12 = out_h4
            delta_w12 = delta_out_o2 * delta_net_o2 * delta_w12

            # HIDDEN LAYER 2
            delta_out_h3 = delta_out_o1 * delta_net_o1 * w9 + delta_out_o2 * delta_net_o2 * w11
            delta_net_h3 = sigmoid(out_h3, derivative=True)
            # w5
            delta_w5 = out_h1
            delta_w5 = delta_net_h3 * delta_w5 * delta_out_h3
            # w6
            delta_w6 = out_h2
            delta_w6 = delta_net_h3 * delta_w6 * delta_out_h3
            # w15
            delta_w15 = delta_net_h3 * delta_out_h3

            delta_out_h4 = delta_out_o1 * delta_net_o1 * w10 + delta_out_o2 * delta_net_o2 * w12
            delta_net_h4 = sigmoid(out_h4, derivative=True)
            # w7
            delta_w7 = out_h1
            delta_w7 = delta_net_h4 * delta_w7 * delta_out_h4
            # w8
            delta_w8 = out_h2
            delta_w8 = delta_net_h4 * delta_w8 * delta_out_h4
            # w16
            delta_w16 = delta_net_h4 * delta_out_h4

            # HIDDEN LAYER 1
            delta_out_h1 = delta_out_h3 * delta_net_h3 * w5
            delta_net_h1 = sigmoid(out_h1, derivative=True)
            # w1
            delta_w1 = X[0]
            delta_w1 = delta_out_h1 * delta_net_h1 * delta_w1
            # w2
            delta_w2 = X[1]
            delta_w2 = delta_out_h1 * delta_net_h1 * delta_w2
            # w13
            delta_w13 = delta_out_h1 * delta_net_h1

            delta_out_h2 = delta_out_h4 * delta_net_h4 * w8
            delta_net_h2 = sigmoid(out_h2, derivative=True)
            # w3
            delta_w3 = X[0]
            delta_w3 = delta_out_h2 * delta_net_h2 * delta_w3
            # w4
            delta_w4 = X[1]
            delta_w4 = delta_out_h2 * delta_net_h2 * delta_w4
            # w14
            delta_w14 = delta_out_h2 * delta_net_h2

            # UPDATE THE WEIGHTS
            w9 = w9 - alpha * delta_w9
            w10 = w10 - alpha * delta_w10
            w11 = w11 - alpha * delta_w11
            w12 = w12 - alpha * delta_w12
            w5 = w5 - alpha * delta_w5
            w6 = w6 - alpha * delta_w6
            w7 = w7 - alpha * delta_w7
            w8 = w8 - alpha * delta_w8
            w15 = w15 - alpha * delta_w15
            w16 = w16 - alpha * delta_w16
            w1 = w1 - alpha * delta_w1
            w2 = w2 - alpha * delta_w2
            w3 = w3 - alpha * delta_w3
            w4 = w4 - alpha * delta_w4
            w13 = w13 - alpha * delta_w13
            w14 = w14 - alpha * delta_w14
            
        print()

        plt.plot(list(range(epochs)), errors, label='run #{}'.format(i+1))
        plt.title('Backpropagation')
        plt.xlabel('epochs')
        plt.ylabel('errors (log scaled)')
        plt.xscale('log') # for proper representation of error decrease
        plt.legend()
    
    plt.show()