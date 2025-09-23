import numpy as np

def compute_weights(decisionMatrix, Y):
    # Identify correct predictions
    correct_mask = (decisionMatrix == Y.reshape(-1, 1))
    # Identify mistakes per instance
    mistakes_per_instance = (decisionMatrix != Y.reshape(-1, 1)).sum(axis=1)
    # Calculate weights matrix and sum across instance
    weights = (correct_mask.astype(float) * (mistakes_per_instance / decisionMatrix.shape[1]).reshape(-1, 1)).sum(axis=0)
    return weights

def weighted_voting(decisionMatrix, weights, Y):
    # get unique values from Y
    classes = np.unique(Y)
    # get the weight matrix
    weight_matrix = weights.reshape(1, -1, 1) * (decisionMatrix[..., None] == classes)
    class_sums = weight_matrix.sum(axis=1)
    final_predictions = classes[np.argmax(class_sums, axis=1)]
    return np.array(final_predictions)