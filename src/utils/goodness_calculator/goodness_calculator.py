
def precision(tp, fp, fn, tn):
    return tp / (tp + fp)

def recall(tp, fp, fn, tn):
    return tp / (tp + fn)

def f_beta(tp, fp, fn, tn, beta=1.0):
    bsq = beta*beta
    p = precision(tp, fp, fn, tn)
    r = recall(tp, fp, fn, tn)
    return (1.0 + bsq)*p*r/(bsq*(p + r))

if __name__ == '__main__':
    tp = int(input(f'# of true  positives: '))
    fp = int(input(f'# of false positives: '))
    fn = int(input(f'# of false negatives: '))
    tn = int(input(f'# of true  negatives: '))
    print('tp       : ', tp)
    print('fp       : ', fp)
    print('fn       : ', fn)
    print('tn       : ', tn)
    print('precision: ', precision(tp, fp, fn, tn))
    print('recall   : ', recall(tp, fp, fn, tn))
    print('f1       : ', f_beta(tp, fp, fn, tn))
    print('f2       : ', f_beta(tp, fp, fn, tn, 2.0))