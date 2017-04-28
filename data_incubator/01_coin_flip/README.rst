Mean function
=============

First we calculate some data points by "brute force" (computing all possible
permutations) for 1 <= N <= 10.

.. code:: python

    import numpy


    def calculate_payment(coins):
        return coins[0] + abs(numpy.diff(coins)).sum()


    def payment_mean(n):
        coins = coins = numpy.arange(n) + 1
        results = [calculate_payment(p) for p in permutations(coins)]
        return numpy.mean(results)


    means = numpy.array([payment_mean(i) for i in range(1, 11)])

Differenciating the array of results a few times resulted in an array of
zeros, which suggests a polynomial equation. Solving the differential
equation taking into account the contour conditions results in second-degree
polynomial equation.

Another approach, rather than solving the differential equation "by hand"
(with a computer, but having to write the contour conditions), is to create
a linear regression model (with non-linear features for the polynomial terms).

.. code:: python

    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split


    sizes = numpy.arange(1, 11)
    poly = numpy.matrix([sizes, sizes ** 2]).transpose()

    X_train, X_test, y_train, y_test = train_test_split(
        poly, means, test_size=0.3, random_state=1)

    model = LinearRegression()
    model.fit(X_train, y_train)

We knew the order of the polynomial, but even adding higher-order polynomial
features to the model results in zero-weighted "extra" features (the model
will not be using them even without adding L1 regularization).

As the training data is exact, there is no need to regularize and there is no
overfit (zero error in the training data results in zero error in test data as
well).


Standard deviation function
===========================

The procedure is the same as for the mean function.

First we calculate some data points by "brute force" (computing all possible
permutations) for 1 <= N <= 10.

.. code:: python

    def payment_std(n):
        coins = coins = numpy.arange(n) + 1
        results = [calculate_payment(p) for p in permutations(coins)]
        return numpy.std(results)


    deviations = numpy.array([payment_std(i) for i in range(1, 11)])

With this configuration, we can not reach an all-zero vector when
differenciating it, so we need to use the squared standard deviation and
2 <= N <= 10 instead.

.. code:: python

    deviations = numpy.array([payment_std(i) ** 2 for i in range(2, 11)])

The linear regression model is created with third-degree polynomial features
this time:

.. code:: python

    sizes = numpy.arange(2, 11)
    poly = numpy.matrix([sizes, sizes ** 2, sizes ** 3]).transpose()

    X_train, X_test, y_train, y_test = train_test_split(
        poly, deviations, test_size=0.3, random_state=1)

    model = LinearRegression()
    model.fit(X_train, y_train)


Probability
===========

Tried to fit the data to a typical probability distribution without much
success, so opted for the brute-force alternative when reasonable and for the
Monte Carlo approximation when brute-force is not computationally viable.

Calculating the probability of the total payment being greater than or equal
to 45 for N = 10 can be done by "brute force":
