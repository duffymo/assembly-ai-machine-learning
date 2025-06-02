"""
Function to evaluate a perceptron model
"""
def perceptron(x1, x2, w1, w2, b):
    return int((x1*w1 + x2*w2 + b) > 0)


"""
Function to test NOR perceptron implementations
"""
def nor_perceptron(w1, w2, b):
    return [perceptron(0.0, 0.0, w1, w2, b),
            perceptron(0.0, 1.0, w1, w2, b),
            perceptron(1.0, 0.0, w1, w2, b),
            perceptron(1.0, 1.0, w1, w2, b)]
"""
Function to test perceptron weights from bnomial site 
"""
if __name__ == "__main__":
    w1 = [-1.0, 0.5, -2.0]
    w2 = [-1.0, 0.5, -1.0]
    b = [1.0, -0.1, 0.8]
    for j in range(0, 3):
        print(j, nor_perceptron(w1[j], w2[j], b[j]))

