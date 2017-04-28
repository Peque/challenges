from itertools import permutations

import numpy


def payment(coins):
    """
    Calculate the total payment given an array of coins.
    """
    return coins[0] + abs(numpy.diff(coins)).sum()


def payment_permutations(n):
    """
    Return all the possible coing permutations for a bag of coins of size `n`.
    """
    coins = coins = numpy.arange(n) + 1
    return numpy.array([payment(p) for p in permutations(coins)])


def payment_max(n):
    """
    Calculate the maximum payment for an array of coins of size `n`.
    """
    return (n + 1) * n // 2


def payment_mean(n):
    """
    Calculate the mean of the payment for a bag of coins of dimension `n`.
    """
    if n != int(n) or n < 1:
        raise ValueError('Parameter must be a positive integer, not "%s"!' % n)
    return (2 * n ** 2 + 3 * n + 1) / 6.


def payment_std(n):
    """
    Calculate the standard deviation of the payment for a bag of coins of
    dimension `n`.
    """
    if n != int(n) or n < 1:
        raise ValueError('Parameter must be a positive integer, not "%s"!' % n)
    if n == 1:
        return 0
    return ((40 * n ** 3 - 35 * n ** 2 - 10 * n + 65) / 900.) ** 0.5


def payment_probability_bruteforce(n, ge):
    """
    Calculate the probability of the total payment being greater or equal
    than `ge` for a bag of `n` coins.

    This calculation is performed by brute-force, so it is pretty slow (not
    recommended for N greater than 10).
    """
    if n != int(n) or n < 1:
        raise ValueError('Parameter must be a positive integer, not "%s"!' % n)
    return (payment_permutations(n) >= ge).mean()


def payment_probability_montecarlo(n, ge, repeat):
    """
    Calculate the probability of the total payment being greater or equal
    than `ge` for a bag of `n` coins.

    This calculation is performed using a Monte Carlo simulation, so the
    result is only approximate.
    """
    if n != int(n) or n < 1:
        raise ValueError('Parameter must be a positive integer, not "%s"!' % n)
    coins = numpy.arange(n) + 1
    total = 0
    for i in range(int(repeat)):
        numpy.random.shuffle(coins)
        if payment(coins) >= ge:
            total += 1
    return total / repeat
