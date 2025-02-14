import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

if __name__ == '__main__':

    # Example decay data (replace with your actual data)
    x_decay = np.linspace(0, 10, 20)
    y_decay = 5 * np.exp(-0.5 * x_decay) + np.random.normal(0, 0.2, len(x_decay))

    # Example growth data (replace with your actual data)
    x_growth = np.linspace(0, 10, 20)
    y_growth = 2 * np.exp(0.3 * x_growth) + np.random.normal(0, 0.5, len(x_growth))

    # Define exponential functions
    def exp_decay(x, a, b, c):
        return a * np.exp(-b * x) + c

    def exp_growth(x, a, b, c):
        return a * np.exp(b * x) + c

    # Fit decay
    params_decay, _ = curve_fit(exp_decay, x_decay, y_decay, p0=(5, 0.5, 0))

    # Fit growth
    params_growth, _ = curve_fit(exp_growth, x_growth, y_growth, p0=(2, 0.3, 0))

    # Generate fitted data
    x_fit = np.linspace(0, 10, 100)
    y_fit_decay = exp_decay(x_fit, *params_decay)
    y_fit_growth = exp_growth(x_fit, *params_growth)

    # Plot results
    plt.figure(figsize=(10, 5))

    # Decay plot
    plt.subplot(1, 2, 1)
    plt.scatter(x_decay, y_decay, label='Data', color='blue')
    plt.plot(x_fit, y_fit_decay, label='Fit', color='red')
    plt.title('Exponential Decay Fit')
    plt.legend()

    # Growth plot
    plt.subplot(1, 2, 2)
    plt.scatter(x_growth, y_growth, label='Data', color='green')
    plt.plot(x_fit, y_fit_growth, label='Fit', color='red')
    plt.title('Exponential Growth Fit')
    plt.legend()

    plt.show()

    # Print fitted parameters
    print(f'Decay Parameters: a={params_decay[0]:.2f}, b={params_decay[1]:.2f}, c={params_decay[2]:.2f}')
    print(f'Growth Parameters: a={params_growth[0]:.2f}, b={params_growth[1]:.2f}, c={params_growth[2]:.2f}')
