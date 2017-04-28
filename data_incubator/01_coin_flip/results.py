from coin import payment_mean
from coin import payment_std
from coin import payment_probability_bruteforce as p_bruteforce
from coin import payment_probability_montecarlo as p_montecarlo


if __name__ == '__main__':

    print('Mean for N=10: ', payment_mean(10))
    print('Mean for N=20: ', payment_mean(20))
    print('Standard deviation for N=10: ', payment_std(10))
    print('Standard deviation for N=20: ', payment_std(20))
    print('P(x>=45|N=10): ', p_bruteforce(10, ge=45))
    print('P(x>=160|N=20) (approx): ', p_montecarlo(20, ge=160, repeat=1e6))
