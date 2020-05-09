import jsonlines
import numpy as np
import pickle
import sys  

def evaluate(predicted, actual):
    correct = 1 - np.count_nonzero(np.subtract(predicted, actual)) / len(actual)
    mae = np.sum(np.abs(np.subtract(predicted, actual))) / len(predicted)
    return correct, mae

if len(sys.argv) == 3:
    predicted_file = sys.argv[1]
    actual_file = sys.argv[2]
    
    predicted = []
    stars = []
    with jsonlines.open(predicted_file) as f:
        for line in f.iter():
            predicted.append(float(line['predicted_stars']))
    
    with open(actual_file,"rb") as f:
        stars = pickle.load(f)
    
    assert len(predicted) == len(stars), "Files are not of the same size"
    correct, mae = evaluate(predicted, stars)

    print("Percent Correct:", correct)
    print("MAE:", mae)
else:
    print("Wrong Number of Args")
