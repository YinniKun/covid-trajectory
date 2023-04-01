import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft
import math

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

## For step function
tau = 4.224
zeta = 0.602*12

# System of ODEs
def model_eqns(y, t):
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

# Initial conditions
# y_0 = [0.05,0.25,0.4,0.3] #Majority has recovered or vaccinated
# y_0 = [0.3,0.2,0,0.5] #Large vaccination before the infection starts 
y_0 = [0.9,0.1,0,0] #No vaccination and everyone is susceptible
# y_0 = [0,0.2,0.8,0] #majority has been infected

# Time points
t = np.linspace(0, 80, 8000)

# Solve
sol = odeint(model_eqns, y_0, t)

# Plot
title = fr'SIRVS Model ($\alpha$={alpha}, $\omega$={omega}, $\epsilon$={epsilon}, $\delta$={delta}, $\tau$={tau}, $\zeta$={zeta})'
title = f"SIRVS Model - Initial Condition = {y_0}"
plt.title(title)
plt.plot(t, sol[:, 0], label='S')
plt.plot(t, sol[:, 1], label='I')
plt.plot(t, sol[:, 2], label='R')
plt.plot(t, sol[:, 3], label='V')
plt.legend()
plt.xlabel('Time (in month)')
plt.ylabel('Population Proportion')
# plt.savefig(f'SIRVS Model.jpg', dpi=1000)
plt.show()

# FFT
# sampling rate
sr = 80

# cases
x = sol[36:, 1]
X = fft(x)
N = len(X)
n = np.arange(N)
T = N/sr
freq = n/T

plt.stem(freq, np.abs(X), 'b', markerfmt=" ", basefmt="-b")
plt.xlabel('Freq (1/month)')
plt.ylabel('FFT Amplitude |I(freq)|')
plt.title("Power Spectrum of the SIRVS Model from Week 36")
plt.xlim(0, 0.3)
plt.ylim(0,80)
plt.savefig(f'FTT_36.jpg', dpi=1000)
plt.show()
