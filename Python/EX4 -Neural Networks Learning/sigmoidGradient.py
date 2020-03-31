from sigmoid import sigmoid

#.....Sigmoid Gradient.....
def sigmoidGradient(z):
    return sigmoid(z)*(1-sigmoid(z))