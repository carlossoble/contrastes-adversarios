# Importamos las librerías necesarias
from scipy.stats import norm, truncnorm
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from math import sqrt



def valoracion_activos(num_obs = 1): # num_obs indica el número de observaciones a recibir

    # Definimos funciones a utilizar según las premisas especificadas en el problema.
    def P_i_a_i_theta(theta):
        prob = norm(x_P_i_a_i_theta, y_P_i_a_i_theta)
        return prob.pdf(theta)

    def P_i_a_i_a(a):
        if a == 1:
            output = x_P_i_a_i_a[0]
        elif a == 1.05:
            output = x_P_i_a_i_a[1]
        elif a == 1.1:
            output = x_P_i_a_i_a[2]
        elif a == 1.15:
            output = x_P_i_a_i_a[3]
        return output

    def h_i_a_i(y, theta, m):
        x_va = m * theta
        y_va = m * theta / chi_h_i_a_i
        prob = norm(x_va, y_va)
        return prob.pdf(y)

    def f_optim(a):
        output = np.dot((0.9 * theta_P_a_i - theta_0 + (a ** 2 - 1) * theta_0 / x_L_a_i),
                        cubic_spline[n](np.clip(a * x_P_a_i, min(Y), max(Y)))) + np.sum(
            (a ** 2 - 1) * theta_0 / x_L_a_i * (1 - cubic_spline[n](np.clip(a * x_P_a_i, min(Y), max(Y)))))
        return output

    # Generación de P_A^i(d|y)
    N1 = 200  # simulaciones
    N2 = 300  # muestras
    Y = np.linspace(8, 15, 701)
    denominador = np.zeros(len(Y))
    numerador = np.zeros(len(Y))
    theta_0 = 10
    X = {}
    X_amp = {}
    cubic_spline = {}

    for n1 in range(N1):
        mu_P_i_a_i_theta = np.random.normal(loc=10, scale=0.2)
        sigma_P_i_a_i_theta = np.random.normal(loc=1, scale=0.1)
        lambda_P_i_a_i_a = np.random.chisquare(df=10)
        chi_h_i_a_i = np.random.chisquare(df=30)
        a = (5 - mu_P_i_a_i_theta) / sigma_P_i_a_i_theta
        b = (15 - mu_P_i_a_i_theta) / sigma_P_i_a_i_theta
        theta_P_i_a_i = truncnorm.rvs(a, b, loc=mu_P_i_a_i_theta, scale=sigma_P_i_a_i_theta, size=N2)
        a_P_i_a_i = np.random.exponential(scale=1 / lambda_P_i_a_i_a, size=N2) + 1
        valores_h = np.array([h_i_a_i(Y, theta, a) for theta, a in zip(theta_P_i_a_i, a_P_i_a_i)])
        denominador = valores_h.sum(axis=0)
        numerador = np.sum(valores_h * (theta_P_i_a_i > theta_0)[:, np.newaxis], axis=0)
        if np.isnan(denominador).any():
            raise ValueError("Denominador tiene valores NaN. Reiniciar cálculo.")
        X[n1] = np.where(denominador > 0, numerador / denominador, 0)
        cubic_spline[n1] = CubicSpline(Y, X[n1], extrapolate=False)

    # Generación de P_i(a)
    N3 = 1000
    N4 = 300  # muestras en la integral
    A = []
    for n3 in range(N3):
        mu_P_a_i_theta = np.random.normal(loc=10, scale=0.2)
        sigma_P_a_i_theta = truncnorm.rvs(a=(0 - 0.5) / 0.1, b=np.inf, loc=0.5, scale=0.1)
        theta_P_a_i = np.random.normal(mu_P_a_i_theta, sigma_P_a_i_theta, N4)
        mu_P_a_i_x = np.random.normal(theta_P_a_i, theta_P_a_i / 50)
        sigma_P_a_i_x = np.random.chisquare(theta_P_a_i) / (30 * sqrt(num_obs))
        x_P_a_i = np.random.normal(mu_P_a_i_x, sigma_P_a_i_x)
        n = np.random.randint(0, N1)
        x_L_a_i = np.random.chisquare(df=10)
        a = minimize(f_optim, 1.1, bounds=[(1, 5)]).x[0]
        A.append(a)

    # Cálculo de d(y)
    N5 = 20000  # muestras en la integral
    aux_acciones = np.random.randint(0, N3, N5)
    acciones = np.array([A[n] for n in aux_acciones])
    theta_i = np.random.normal(10, 1, N5)
    p_i_x_theta = norm(theta_i, theta_i / (30 * sqrt(num_obs)))

    # Visualización de la decisión óptima
    y_values = np.linspace(0.1, 20, 6001)
    resultados = np.array([(y * np.dot((theta_0 - theta_i), p_i_x_theta.pdf(y / acciones)) / N5) < 0 for y in y_values])
    plt.plot(y_values[resultados], resultados[resultados], color='green', label='Comprar')
    plt.plot(y_values[~resultados], resultados[~resultados], color='red', label='No comprar')
    plt.xlabel('y')
    plt.yticks([0, 1], ['No comprar', 'Comprar'])
    plt.show()

    print("La decisión óptima con ", num_obs, "observaciones es comprar cuando y > ", min(y_values[resultados]))