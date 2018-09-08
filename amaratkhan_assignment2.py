import numpy as np
import matplotlib.pyplot as plt


def svm_loss(column, true_index, offset=1):
    loss_sum = 0
    for i in range(0,column.size):
        if i != true_index:
            x = max(0, column[i]-column[true_index]+offset)
            loss_sum += x
    return loss_sum

def softmax_loss(column, true_index):
    loss_sum = 0
    for i in range(0,column.size):
        x = np.exp(column[i])
        loss_sum += x
    normalized = np.exp(column[true_index])/loss_sum
    return -np.log(normalized)


if __name__ == "__main__":

    svm_losses = []
    softmax_losses = []

    for iteration in range(50):
        # generate matrix with random values in the range (-10,10)
        # 10 classes (rows) by 100 instances (columns)
        array = 10*np.random.uniform(-1,1, (10,100))

        transposed = array.T # transpose array for accessing each column

        svm_loss_sum = 0
        softmax_loss_sum = 0
        for column in transposed:
            # generate true index for each column
            true_index = np.random.randint(0,9)
            # compute svm loss
            svm_loss_column = svm_loss(column, true_index)
            svm_loss_sum += svm_loss_column

            # compute softmax cross-entropy
            softmax_loss_column = softmax_loss(column, true_index)
            softmax_loss_sum += softmax_loss_column

        svm_loss_average = svm_loss_sum / transposed[0].size
        softmax_loss_average = softmax_loss_sum / transposed[0].size

        svm_losses.append(svm_loss_average)
        softmax_losses.append(softmax_loss_average)


    # plot the averages to the graph
    x = list(range(50))
    plt.scatter(x, svm_losses)
    plt.scatter(x, softmax_losses, marker='+')
    plt.legend(['svm_loss', 'softmax_cross_entropy'])
    plt.show()