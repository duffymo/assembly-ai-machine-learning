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
