"""
The classic Bayes problem
https://en.wikipedia.org/wiki/Bayes%27_theorem
Law of Total Probability
How likely is it that you are ill when a test comes up positive?

p(a|b) * p(b) = p(b|a) * p(a)

If a == sick and b == positive test
Then

p(sick|pos) = p(pos|sick) * p(sick) / p(pos)
p(pos) = p(pos|sick) * p(sick) + p(pos|not-sick) * p(not-sick)

Another example is from manufacturing QA: if a test comes up positive,
how likely is it that the product is defective?

p(defective|positive) = p(positive|defective) * p(defective) / (p(positive|defective) * p(defective) + p(positive|not-defective) * p(not-defective))

To find out the probability that the product is still good, given a positive test, just subtract.

p(not-defective|positive) = 1 - p(defective|positive)

From Binomial example:

p(defective) = 0.001
p(not-defective) = 1 - p(defective) = 0.999
p(positive|defective) = 0.98 (test accuracy)
p(positive|not-defective) = 0.02 (test accuracy)

Therefore:

p(defective|positive) = 0.98 * 0.001 / (0.98 * 0.001 + 0.02 * 0.999) = 0.046755
p(not-defective|positive) = 1 - p(defective|positive) = 0.953245
"""
def bayes(p_pos_sick, p_sick, p_pos_not_sick, p_not_sick):
    p_pos = p_pos_sick * p_sick + p_pos_not_sick * p_not_sick
    return p_pos_sick * p_sick / p_pos

if __name__ == '__main__':
    p_b_a        = float(input(f'p(b|a)      : '))
    p_a          = float(input(f'p(a)        : '))
    p_b_not_a    = float(input(f'p(b|not_a)  : '))
    p_not_a      = float(input(f'p(not_a))   : '))
    print('p(a|b): ', bayes(p_b_a, p_a, p_b_not_a, p_not_a))
