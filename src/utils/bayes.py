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

"""
Charlotte took a test from the pharmacy to understand whether she had the flu. She lives in a town with 10,000 people, and 5% are sick.

The test is 90% accurate, and Charlotte got a positive result. Fortunately, she knows enough about probabilities to understand she is likely healthy.

What should be the accuracy of the test for Charlotte to be likely sick?

Solution:

First, let's understand why Charlotte will likely be healthy even when she gets a positive result.

5% of the people that live in Charlotte's town are sick, which means that 500 individuals are sick and 9,500 are healthy.

The test is 90% accurate. If all 500 sick individuals take the test, 500 * 0.9 = 450 will be correctly diagnosed as sick, but 50 will be incorrectly diagnosed as healthy. If all 9,500 healthy individuals take the test, 9,500 * 0.9 = 8,550 will be correctly diagnosed as healthy, but 950 will be incorrectly diagnosed as sick.

Charlotte could be one of those 450 sick people who had a positive test, but she could also be one of the 950 healthy with an incorrect positive result. To determine the probability of Charlotte being sick, we can compute how likely she is to be in each group.

Out of 450 + 950 = 1,400 positive results, Charlotte is 450 / 1,400 = 32% likely to be sick, and 950 / 1,400 = 68% likely to be healthy.

Despite what Charlotte's test suggests, she is likely to be healthy. Knowing this, what should be the accuracy of the test to change this result?

We will need at least a 51% probability of Charlotte being healthy. That means that:

x = test accuracy
a = sick individuals diagnosed as sick 
b = healthy individuals diagnosed as sick

a / (a + b) = 0.51

a = 500 * x = 500x
b = 9500 * (1 - x) = 9500 - 9500x
We can now put it all together:

500x / (500x + 9500 - 9500x) = 0.51
500x / (9500 - 9000x) = 0.51
500x = 0.51 * (9500 - 9000x)
500x = 4845 - 4590x
500x + 4590x = 4845
5090x = 4845
x = 4845 / 5090
x = 0.951
The test should be at least 95.1% accurate for Charlotte to be likely sick.

"""
def required_test_accuracy(sick_diagnosed_sick, healthy_diagnosed_sick):
    return sick_diagnosed_sick/(sick_diagnosed_sick + healthy_diagnosed_sick)

if __name__ == '__main__':
    p_b_a        = float(input(f'p(b|a)      : '))
    p_a          = float(input(f'p(a)        : '))
    p_b_not_a    = float(input(f'p(b|not_a)  : '))
    p_not_a      = float(input(f'p(not_a))   : '))
    print('p(a|b): ', bayes(p_b_a, p_a, p_b_not_a, p_not_a))
