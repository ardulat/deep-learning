import numpy as np
import matplotlib.pyplot as plt

from random import randint

def partition(A, start, end):
    pivot = A[end]
    partitionIndex = start

    for i in range(start, end):
        if (A[i] < pivot):
            A[i], A[partitionIndex] = A[partitionIndex], A[i]
            partitionIndex += 1
    
    A[end], A[partitionIndex] = A[partitionIndex], A[end]

    return partitionIndex


def quickSortHelper(A, start, end):
    if (start < end):
        pivotIndex = partition(A, start, end)
        quickSortHelper(A, start, pivotIndex-1)
        quickSortHelper(A, pivotIndex+1, end)

def quickSort(A):
    start = 0
    end = len(A)
    quickSortHelper(A, start, end-1)

def binarySearch(A, x):
    """A - array, x - element to search"""
    low = 0
    high = len(A)-1
    mid = 0
    comparisons = 0

    while (low <= high):
        mid = int((low+high)/2)
        comparisons += 1
        if(A[mid] < x): # look on the right half
            low = mid+1
        elif(A[mid] > x): # look on the left half
            high = mid-1
        else: # we found it
            # print("WE FOUND IT!")
            return comparisons
    
    # print("THERE IS NO SUCH ELEMENT!")
    return comparisons

def exhaustiveSearch(A, x):
    comparisons = 0
    for i in range(0, len(A)):
        comparisons += 1
        if (A[i] == x):
            # print("WE FOUND IT!")
            return comparisons
        
    # print("THERE IS NO SUCH ELEMENT!")
    return comparisons

if __name__ == "__main__":

    # uncomment if you want to make the output of random function the same each time
    # np.random.seed(42)

    # generating random array to check if sort and search works
    A = np.random.randint(0, 100, 10)
    quickSort(A)
    print(A)

    x = binarySearch(A, 51)
    x = exhaustiveSearch(A, 72)

    # compute execution times for BINARY SEARCH
    binarySearchResults = []
    binarySearchX = []

    for i in range(0, 1001, 10):
        binarySearchX.append(i)
        t = 10 # number of times to calculate
        total_comparisons = 0
        for j in range(0, t):
            A = np.random.randint(0, 100, i) # generate random array with size = i
            quickSort(A)
            x = randint(0, 100) # generate random integer to search
            comparisons = binarySearch(A, x)
            total_comparisons += comparisons
        avg_comparisons = total_comparisons / t
        binarySearchResults.append(avg_comparisons)

    # compute execution times for EXHAUSTIVE SEARCH
    exhaustiveSearchResults = []
    exhaustiveSearchX = []

    for i in range(0, 1001, 10):
        exhaustiveSearchX.append(i)
        t = 10 # number of times to calculate
        total_comparisons = 0
        for j in range(0, t):
            A = np.random.randint(0, 100, i) # generate random array with size = i
            quickSort(A)
            x = randint(0, 100) # generate random integer to search
            comparisons = exhaustiveSearch(A, x)
            total_comparisons += comparisons
        avg_comparisons = total_comparisons / t
        exhaustiveSearchResults.append(avg_comparisons)

    print(binarySearchX)
    print(binarySearchResults)
    print()
    print(exhaustiveSearchX)
    print(exhaustiveSearchResults)

    plt.plot(binarySearchX, binarySearchResults, exhaustiveSearchX, exhaustiveSearchResults, 'r')

    plt.show()




    
