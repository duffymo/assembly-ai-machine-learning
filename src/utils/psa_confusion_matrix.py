"""
Make sure you understand exactly what specificity is.
https://priorprobability.com/2021/03/08/bayes-7-sensitivity-and-specificity/
"""
def psa_confusion_matrix(sensitivity, specificity, prob_disease):
    tp = sensitivity * prob_disease
    fp = (1.0 - specificity) * (1.0 - prob_disease)
    fn = (1.0 - sensitivity) * prob_disease
    tn = specificity * (1.0 - prob_disease)
    return [[tp, fp], [fn, tn]]


"""
Function to calculate confusion matrix
for PSA dataset.

See https://www.jclinepi.com/article/S0895-4356(20)31225-7/fulltext

and https://pmc.ncbi.nlm.nih.gov/articles/PMC137591/
"""
if __name__ == '__main__':
    print('psa confusion matrix')
    # values for 4 ng/ml PSA cutoff
    cm = psa_confusion_matrix(0.86, 0.33, 0.35)
    print(cm)
    print("probability of positive test result: ", cm[0][0] + cm[0][1])
    print("probability of negative test result: ", cm[1][0] + cm[1][1])
    print("probability of presence of disease : ", cm[0][0] + cm[1][0])
    print("probability of absence  of disease : ", cm[0][1] + cm[1][1])
    print("confusion matrix sum               : ", cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])
