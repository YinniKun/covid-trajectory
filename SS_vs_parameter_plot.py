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


def steady_state_vs_parameter(which_parameter: float, dependent_var_idx: int):
    '''
    For now, we investigate the steady state of I vs alpha, could also be used for other parameters
    dependent variable: 0 for S, 1 for I, 2 for R, 3 for V
    '''
    # Initial conditions
    y_0 = [0.9, 0.1, 0, 0]

    # Time points
    t = np.linspace(0, 80, 216)

    # Range of values of the parameter
    parameter_vals = np.arange(which_parameter, 1.1, 0.1)
    print(parameter_vals)
    steady_state = []

    # Solve
    for param in parameter_vals:
        sol = odeint(model_eqns, y_0, t, args=(param,))

        # Plot for steady state vs parameter
        dependent_variable = sol[:, dependent_var_idx]

        steady_state.append(dependent_variable[-1])

    return parameter_vals, steady_state


if __name__ == '__main__':
    idx = 1  # for I variable
    param = 'alpha'  # or 'omega', 'epsilon', 'delta'

    dependent_var = {0: 'S', 1: 'I', 2: 'R', 3: 'V'}

    param_vals, steady_state = steady_state_vs_parameter(alpha, idx)
    plt.title(f'Steady state of ${dependent_var[idx]}$ vs $\{param}$')
    plt.plot(param_vals, steady_state, 'ro')
    plt.xlabel(f'${param}$')
    plt.ylabel(f'Steady state of ${dependent_var[idx]}$')
    plt.ylim(0, 1)
    # plt.savefig(f'{dependent_var[idx]}_SS_vs_{param}_plot.jpg', dpi=1000)
    plt.show()

