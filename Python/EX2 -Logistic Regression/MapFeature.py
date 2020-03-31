# Importing PolynomialFeatures from SKLEARN
from sklearn.preprocessing import PolynomialFeatures

def MapFeature(X):
    poly = PolynomialFeatures(6)       # Creating the model
    X2 = poly.fit_transform(X)         # Transforming the data into the sixth power polynomial
    return X2