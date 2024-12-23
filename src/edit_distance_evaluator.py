'''
LlMs sometimes do not fit to the response expected from them and return more characters
as output class, so we use as predicted class the one with the smallest edit distance
(in few words, the most similar among the possible classes).
'''
from sklearn.metrics import classification_report

def edit_distance(s1, s2):    
    # Initialize the distance matrix
    dp = [[0] * (len(s2) + 1) for _ in range(len(s1) + 1)]

    # Initialize the first row
    for i in range(len(s1) + 1):
        dp[i][0] = i

    # Initialize the first column
    for j in range(len(s2) + 1):
        dp[0][j] = j

    # Calculate de distance matrix
    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            dp[i][j] = min(
                dp[i - 1][j - 1] + (s1[i - 1] != s2[j - 1]),
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1
            )

    # Return the edit distance
    return dp[-1][-1]

def evaluate(y_true, y_pred):
    # Get the unique values of y_true
    unique_values = y_true.unique()

    # For those values of y_pred that are not in unique_values,
    # we look for the closest value by edit distance
    y_pred = y_pred.apply(lambda x: min(unique_values, key=lambda y: edit_distance(x, y)))
    return classification_report(y_true, y_pred, output_dict=True)