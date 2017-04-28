from math import factorial

import pytest
import numpy
from pytest import approx

from coin import payment
from coin import payment_max
from coin import payment_mean
from coin import payment_permutations
from coin import payment_probability_bruteforce
from coin import payment_probability_montecarlo
from coin import payment_std


@pytest.mark.parametrize('coins,result', [
    ([1], 1),
    ([3, 1, 2], 6),
    ([5, 3, 2, 4, 1], 13),
    ([5, 8, 2, 4, 1, 7, 3, 6], 32),
    ([1, 8, 4, 7, 6, 3, 2, 5], 23),
])
def test_payment(coins, result):
    """
    Test correct `payment` function behavior against well-known results.
    """
    assert payment(coins) == result


@pytest.mark.parametrize('n,result', [
    (1, 1),
    (2, 3),
    (3, 6),
    (4, 10),
    (7, 28),
    (8, 36),
    (10, 55),
])
def test_payment_max(n, result):
    """
    Test `payment_max` function given known result values.
    """
    assert payment_max(n) == approx(result)


@pytest.mark.parametrize('n,result', [
    (1, 1.),
    (2, 2.5),
    (3, 4.666666),
    (4, 7.5),
    (7, 20.),
    (8, 25.5),
    (10, 38.5),
])
def test_payment_mean_value(n, result):
    """
    Test `payment_mean` function given known result values.
    """
    assert payment_mean(n) == approx(result)


@pytest.mark.parametrize('n', [1, 2, 3, 4, 5, 6, 7, 8])
def test_payment_mean_permutations(n):
    """
    Test `payment_mean` function against a "brute force" calculation.
    """
    mean = numpy.mean(payment_permutations(n))
    assert payment_mean(n) == approx(mean)


@pytest.mark.parametrize('n', [10, 20, 100])
def test_payment_mean_montecarlo(n):
    """
    Test `payment_mean` function with Monte Carlo methods (higher N values).
    """
    iterations = 100000

    coins = numpy.arange(n) + 1
    total = 0
    for i in range(iterations):
        numpy.random.shuffle(coins)
        total += payment(coins)
    result = total / iterations
    assert payment_mean(n) == approx(result, rel=1e-2)


@pytest.mark.parametrize('n', [0, -1, 1.1])
def test_payment_mean_error(n):
    """
    Test `payment_mean` function exceptions.
    """
    with pytest.raises(ValueError):
        payment_mean(n)


@pytest.mark.parametrize('n,result', [
    (1, 0.),
    (2, 0.5),
    (3, 0.942809),
    (4, 1.5),
    (7, 3.651484),
    (8, 4.5),
    (10, 6.365270),
])
def test_payment_std_value(n, result):
    """
    Test `payment_std` function given known result values.
    """
    assert payment_std(n) == approx(result)


@pytest.mark.parametrize('n', [1, 2, 3, 4, 5, 6, 7, 8])
def test_payment_std_permutations(n):
    """
    Test `payment_std` function against a "brute force" calculation.
    """
    std = numpy.std(payment_permutations(n))
    assert payment_std(n) == approx(std)


@pytest.mark.parametrize('n', [10, 20, 100])
def test_payment_std_montecarlo(n):
    """
    Test `payment_std` function with Monte Carlo methods (higher N values).
    """
    iterations = 100000

    coins = numpy.arange(n) + 1
    payments = []
    for i in range(iterations):
        numpy.random.shuffle(coins)
        payments.append(payment(coins))
    result = numpy.std(payments)
    assert payment_std(n) == approx(result, rel=1e-2)


@pytest.mark.parametrize('n', [0, -1, 1.1])
def test_payment_std_error(n):
    """
    Test `payment_std` function exceptions.
    """
    with pytest.raises(ValueError):
        payment_std(n)


@pytest.mark.parametrize('n,prob,result', [
    (1, 1, 1.),
    (3, 6., 0.1666666),
    (3, 6.01, 0.),
    (7, 22, 0.3515873),
])
def test_payment_probability_bruteforce(n, prob, result):
    """
    Test `payment_probability_bruteforce` function given known result values.
    """
    assert payment_probability_bruteforce(n, prob) == approx(result)


@pytest.mark.parametrize('n,prob,result', [
    (1, 1, 1.),
    (3, 6., 0.1666666),
    (3, 6.01, 0.),
    (7, 22, 0.3515873),
    (10, 45, 0.1817956349),
])
def test_payment_probability_montecarlo(n, prob, result):
    """
    Test `payment_probability_montecarlo` function given known result values.
    """
    assert payment_probability_montecarlo(n, prob, repeat=1e5) == \
        approx(result, rel=1e-2)


@pytest.mark.parametrize('n', [0, -1, 1.1])
def test_payment_probability_bruteforce_error(n):
    """
    Test `payment_std` function exceptions.
    """
    with pytest.raises(ValueError):
        payment_probability_bruteforce(n, 0)
