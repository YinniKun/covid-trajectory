import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import math

# I vs alpha (up to 1) steady state

# Parameters
alpha = 0.007
omega = 0.17
epsilon = 0.62
delta = 3

## For Linear
# tau = 4.224
# zeta = 0.602

## For exponential
# tau = 6.032
# zeta = 0.0421

## For heaviside step function
tau = 4.224
zeta = 0.602*12


# System of ODEs
def model_eqns(y, t, alpha=alpha, omega=omega, epsilon=epsilon, delta=delta, tau=tau, zeta=zeta):
    S, I, R, V = y

    # for Linear
    # dSdt = -(zeta*t+tau) * S * I + omega*R + omega*V - alpha*S
    # dIdt = ((zeta*t+tau) * S + (zeta*t+tau) * (1 - epsilon) * V - delta) * I
    # dRdt = delta * I - omega*R
    # dVdt = -(zeta*t+tau) * (1 - epsilon) * V * I - omega*V + alpha*S

    # for exponential
    # dSdt = -(zeta*np.exp(t*tau)) * S * I + omega*R + omega*V - alpha*S
    # dIdt = ((zeta*np.exp(t*tau)) * S + (zeta*np.exp(t*tau)) * (1 - epsilon) * V - delta) * I
    # dRdt = delta * I - omega*R
    # dVdt = -(zeta*np.exp(t*tau)) * (1 - epsilon) * V * I - omega*V + alpha*S

    # for step function
    dSdt = -(tau+math.floor(t/12)*zeta) * S * I + omega*R + omega*V - alpha*S
    dIdt = ((tau+math.floor(t/12)*zeta) * S + (tau+math.floor(t/12)*zeta) * (1 - epsilon) * V - delta) * I
    dRdt = delta * I - omega*R
    dVdt = -(tau+math.floor(t/12)*zeta) * (1 - epsilon) * V * I - omega*V + alpha*S

    return [dSdt, dIdt, dRdt, dVdt]


def I_vs_time_plot(param_str, param, dependent_var_idx):
    '''
        For now, we investigate the steady state of I vs alpha, could also be used for other parameters
        dependent variable: 0 for S, 1 for I, 2 for R, 3 for V
        '''
    # Initial conditions
    y_0 = [0.9, 0.1, 0, 0]

    # Time points
    t = np.linspace(0, 80, 216)

    # Range of values of the parameter
    parameter_vals = np.linspace(param, 1, 6)

    # Solve
    for param in parameter_vals:
        sol = odeint(model_eqns, y_0, t, args=(param,))

        # Plot for steady state vs parameter
        dependent_variable = sol[:, dependent_var_idx]

        plt.plot(t, dependent_variable, label=f'$\{param_str} = {round(param, 2)}$')

    dependent_var = {0: 'S', 1: 'I', 2: 'R', 3: 'V'}
    title = rf'Time Series of {dependent_var[dependent_var_idx]} for Various Parameters $\{param_str}$'
    plt.title(title)
    plt.xlabel('Time (in month)')
    plt.ylabel('Population Proportion')
    plt.legend()
    plt.savefig(f'{title}.jpg', dpi=1000)
    plt.show()


if __name__ == '__main__':
    # I_vs_time_plot('alpha', alpha, 1)  # varying alpha
    I_vs_time_plot('epsilon', epsilon, 1)  # varying epsilon