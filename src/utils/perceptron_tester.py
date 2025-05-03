def perceptron(x1, x2, w1, w2, b):
    return int((x1*w1 + x2*w2 + b) > 0)

"""
Function to test perceptron weights from bnomial site 
"""
if __name__ == "__main__":
    print(perceptron(1, 1, 1, 1, 1))