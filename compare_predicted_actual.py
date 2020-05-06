import jsonlines, sys
import numpy as np

def evaluate(predicted, actual):
    correct = 1 - np.count_nonzero(np.subtract(predicted, actual)) / len(actual)
    mae = np.sum(np.abs(np.subtract(predicted, actual))) / len(predicted)
    return correct, mae

if len(sys.argv) == 3:
    predicted_file = sys.argv[1]
    actual_file = sys.argv[2]
    
    predicted = []
    actual = []
    with jsonlines.open(predicted_file) as f:
        for line in f.iter():
            predicted.append(line['predicted_stars'])
    
    with jsonlines.open(actual_file) as f:
        for line in f.iter():
            actual.append(line['stars'])
    
    assert len(predicted) == len(actual), "Files are not of the same size"
    correct, mae = evaluate(predicted, actual)
    print("Percent Correct:", correct)
    print("MAE:", mae)
else:
    print("Wrong Number of Args")
