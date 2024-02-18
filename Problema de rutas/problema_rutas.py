# Importamos las librerías necesarias
import random
from scipy.stats import norm
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt


def problema_rutas():


    # Definimos funciones a utilizar según las premisas especificadas en el problema
    def x_mod(accion, x):
        if accion == 0:
            y = 1.12 * x + 15
        elif accion == 1:
            y = 1.05 * x + 18
        elif accion == 2:
            y = 1.1 * x
        elif accion == 3:
            y = x + 6
        elif accion == 4:
            n = random.randint(1, 2)
            if n == 1:
                y = 1.2 * x
            else:
                y = x
        elif accion == 5:
            y = x
        return y

    def x_mod_inv(accion, y, lluvia):
        if accion == 0:
            x = (y - 15) / 1.12
        elif accion == 1:
            x = (y - 18) / 1.1
        elif accion == 2:
            x = y / 1.1
        elif accion == 3:
            x = y - 6
        elif accion == 4:
            if lluvia == True:
                x = y / 1.2
            elif lluvia == False:
                x = y
        elif accion == 5:
            x = y
        return x

    def coste(accion):
        if accion == 0:
            coste = 30
        elif accion == 1:
            coste = 30
        elif accion == 2:
            coste = 6
        elif accion == 3:
            coste = 6
        elif accion == 4:
            coste = 6
        elif accion == 5:
            coste = 0
        return coste

    def h_i_a_i(y, theta, accion):
        mu = x_mod(accion, theta - 120)
        sigma = mu / chi_h_i_a_i
        prob = norm(mu, sigma)
        return prob.pdf(y)

    def f_optim(a):
        y = [x_mod(a, x) for x in x_P_a_i]
        output = np.mean((coste(a) - 300) * cubic_spline[n](np.clip(y, min(Y), max(Y)))) + np.mean(
            coste(a) * (1 - cubic_spline[n](np.clip(y, min(Y), max(Y)))))
        return output


    # Generación P_A^i(d|y)
    N1 = 200  # simulaciones
    N2 = 1000  # muestras
    Y = np.linspace(0.1, 150, 1001)
    denominador = np.zeros(len(Y))
    numerador = np.zeros(len(Y))
    theta_0 = 180
    X = {}
    cubic_spline = {}
    for n1 in range(N1):
        unif_P_i_a_i_theta = np.random.uniform(25, 35)
        beta_P_i_a_i_theta = 30 / (unif_P_i_a_i_theta ** 2)
        alfa_P_i_a_i_theta = 30 * beta_P_i_a_i_theta
        dir_P_i_a_i_a = np.random.dirichlet([1, 1, 1, 1, 1, 1])
        chi_h_i_a_i = np.random.chisquare(df=10)

        theta_P_i_a_i = 120 + np.random.gamma(shape=alfa_P_i_a_i_theta, scale=1 / beta_P_i_a_i_theta, size=N2)
        a_P_i_a_i = np.random.choice(len(dir_P_i_a_i_a), size=N2, p=dir_P_i_a_i_a)

        valores_h = np.array([h_i_a_i(Y, theta, accion) for theta, accion in zip(theta_P_i_a_i, a_P_i_a_i)])

        denominador = valores_h.sum(axis=0)
        numerador = np.sum(valores_h * (theta_P_i_a_i > theta_0)[:, np.newaxis], axis=0)
        if np.isnan(denominador).any():
            raise ValueError("Denominador tiene valores NaN. Reiniciar cálculo.")
        X[n1] = np.where(denominador > 0, numerador / denominador, 0)
        cubic_spline[n1] = CubicSpline(Y, X[n1], extrapolate=False)


    # Generación de P_i(a)
    N3 = 5000
    N4 = 1000  # muestras en la integral
    A = []
    for n3 in range(N3):
        unif_P_i_a_i_theta = np.random.uniform(25, 35)
        beta_P_i_a_i_theta = 30 / (unif_P_i_a_i_theta ** 2)
        alfa_P_i_a_i_theta = 30 * beta_P_i_a_i_theta
        theta_P_a_i = 120 + np.random.gamma(shape=alfa_P_i_a_i_theta, scale=1 / beta_P_i_a_i_theta, size=N4)
        mu_P_a_i_x = np.random.normal(theta_P_a_i - 120, (theta_P_a_i - 120) / 50)
        chi_sigma_P_a_i_x = np.random.chisquare(df=10)
        sigma_P_a_i_x = (theta_P_a_i - 120) / chi_sigma_P_a_i_x
        x_P_a_i = np.random.normal(mu_P_a_i_x, sigma_P_a_i_x)
        n = np.random.randint(0, N1)
        valores_f = [f_optim(a) for a in range(6)]
        a = np.argmin(valores_f)
        A.append(a)


    # Cálculo de d(y)
    theta_0 = 180
    N5 = 50000  # muestras en la integral
    aux_acciones = np.random.randint(0, N3, N5)
    acciones = np.array([A[n] for n in aux_acciones])
    theta_i = 120 + np.random.exponential(scale=30, size=N5)
    p_i_x_theta = norm(theta_i - 120, (theta_i - 120) / 10)
    costes = 400 * ((theta_i > theta_0) & (theta_i < (theta_0 + 60))) + 700 * (
                (theta_i > theta_0 + 60) & (theta_i < (theta_0 + 120))) + 1000 * (theta_i > theta_0 + 120)


    # Visualización de la decisión óptima
    y_values = np.linspace(0.25, 200, 800)

    results_no_peaje = [(1 / 2 * np.dot(costes, p_i_x_theta.pdf(
        np.array([x_mod_inv(a, y, True) for a in acciones]))) + 1 / 2 * np.dot(costes, p_i_x_theta.pdf(
        np.array([x_mod_inv(a, y, False) for a in acciones])))) / N5 for y in y_values]

    results_peaje = [(1 / 2 * np.sum(
        300 * p_i_x_theta.pdf(np.array([x_mod_inv(a, y, True) for a in acciones]))) + 1 / 2 * np.sum(
        300 * p_i_x_theta.pdf(np.array([x_mod_inv(a, y, False) for a in acciones])))) / N5 for y in y_values]

    peaje = np.array(results_peaje) < np.array(results_no_peaje)
    plt.figure(figsize=(8, 4))
    plt.plot(y_values[peaje], peaje[peaje], color='green', label='Peaje')
    plt.plot(y_values[~peaje], peaje[~peaje], color='red', label='No peaje')
    plt.xlabel('y')
    plt.yticks([0, 1], ['No peaje', 'Peaje'], fontsize=12)
    plt.show()

    print("La decisión óptima es pagar el peaje cuando y > ", min(y_values[peaje]), " minutos.")





def problema_rutas_sin_perdida():


    # Definimos funciones a utilizar según las premisas especificadas en el problema
    def x_mod(accion, x):
        if accion == 0:
            y = 1.12 * x + 15
        elif accion == 1:
            y = 1.05 * x + 18
        elif accion == 2:
            y = 1.1 * x
        elif accion == 3:
            y = x + 6
        elif accion == 4:
            n = random.randint(1, 2)
            if n == 1:
                y = 1.2 * x
            else:
                y = x
        elif accion == 5:
            y = x
        return y

    def x_mod_inv(accion, y, lluvia):
        if accion == 0:
            x = (y - 15) / 1.12
        elif accion == 1:
            x = (y - 18) / 1.1
        elif accion == 2:
            x = y / 1.1
        elif accion == 3:
            x = y - 6
        elif accion == 4:
            if lluvia == True:
                x = y / 1.2
            elif lluvia == False:
                x = y
        elif accion == 5:
            x = y
        return x

    def coste(accion):
        if accion == 0:
            coste = 30
        elif accion == 1:
            coste = 30
        elif accion == 2:
            coste = 6
        elif accion == 3:
            coste = 6
        elif accion == 4:
            coste = 6
        elif accion == 5:
            coste = 0
        return coste

    def h_i_a_i(y, theta, accion):
        mu = x_mod(accion, theta - 120)
        sigma = mu / chi_h_i_a_i
        prob = norm(mu, sigma)
        return prob.pdf(y)

    def f_optim(a):
        y = [x_mod(a, x) for x in x_P_a_i]
        output = np.mean((coste(a) - 300) * cubic_spline[n](np.clip(y, min(Y), max(Y)))) + np.mean(
            coste(a) * (1 - cubic_spline[n](np.clip(y, min(Y), max(Y)))))
        return output


    # Generación P_A^i(d|y)
    N1 = 200  # simulaciones
    N2 = 1000  # muestras
    Y = np.linspace(0.1, 150, 1001)
    denominador = np.zeros(len(Y))
    numerador = np.zeros(len(Y))
    theta_0 = 180
    X = {}
    cubic_spline = {}
    for n1 in range(N1):
        unif_P_i_a_i_theta = np.random.uniform(25, 35)
        beta_P_i_a_i_theta = 30 / (unif_P_i_a_i_theta ** 2)
        alfa_P_i_a_i_theta = 30 * beta_P_i_a_i_theta
        dir_P_i_a_i_a = np.random.dirichlet([1, 1, 1, 1, 1, 1])
        chi_h_i_a_i = np.random.chisquare(df=10)

        theta_P_i_a_i = 120 + np.random.gamma(shape=alfa_P_i_a_i_theta, scale=1 / beta_P_i_a_i_theta, size=N2)
        a_P_i_a_i = np.random.choice(len(dir_P_i_a_i_a), size=N2, p=dir_P_i_a_i_a)

        valores_h = np.array([h_i_a_i(Y, theta, accion) for theta, accion in zip(theta_P_i_a_i, a_P_i_a_i)])

        denominador = valores_h.sum(axis=0)
        numerador = np.sum(valores_h * (theta_P_i_a_i > theta_0)[:, np.newaxis], axis=0)
        if np.isnan(denominador).any():
            raise ValueError("Denominador tiene valores NaN. Reiniciar cálculo.")
        X[n1] = np.where(denominador > 0, numerador / denominador, 0)
        cubic_spline[n1] = CubicSpline(Y, X[n1], extrapolate=False)


    # Generación de P_i(a)
    N3 = 5000
    N4 = 1000  # muestras en la integral
    A = []
    for n3 in range(N3):
        unif_P_i_a_i_theta = np.random.uniform(25, 35)
        beta_P_i_a_i_theta = 30 / (unif_P_i_a_i_theta ** 2)
        alfa_P_i_a_i_theta = 30 * beta_P_i_a_i_theta
        theta_P_a_i = 120 + np.random.gamma(shape=alfa_P_i_a_i_theta, scale=1 / beta_P_i_a_i_theta, size=N4)
        mu_P_a_i_x = np.random.normal(theta_P_a_i - 120, (theta_P_a_i - 120) / 50)
        chi_sigma_P_a_i_x = np.random.chisquare(df=10)
        sigma_P_a_i_x = (theta_P_a_i - 120) / chi_sigma_P_a_i_x
        x_P_a_i = np.random.normal(mu_P_a_i_x, sigma_P_a_i_x)
        n = np.random.randint(0, N1)
        valores_f = [f_optim(a) for a in range(6)]
        a = np.argmin(valores_f)
        A.append(a)


    # Cálculo de d(y)
    theta_0 = 180
    N5 = 50000  # muestras en la integral
    aux_acciones = np.random.randint(0, N3, N5)
    acciones = np.array([A[n] for n in aux_acciones])
    theta_i = 120 + np.random.exponential(scale=30, size=N5)
    p_i_x_theta = norm(theta_i - 120, (theta_i - 120) / 10)

    # Visualización de la decisión óptima
    x_values = np.linspace(0.25, 150, 800)
    total = [(1 / 2 * np.sum(p_i_x_theta.pdf(
        np.array([x_mod_inv(a, y, True) for a in acciones]))) + 1 / 2 * np.sum(p_i_x_theta.pdf(
        np.array([x_mod_inv(a, y, False) for a in acciones])))) for y in x_values]
    hipotesis_nula = [(1 / 2 * np.dot(theta_i > theta_0, p_i_x_theta.pdf(
        np.array([x_mod_inv(a, y, True) for a in acciones]))) + 1 / 2 * np.dot(theta_i > theta_0, p_i_x_theta.pdf(
        np.array([x_mod_inv(a, y, False) for a in acciones])))) for y in x_values]
    probabilidad = np.array(hipotesis_nula) / np.array(total)

    plt.figure(figsize=(8, 4))
    plt.plot(x_values, probabilidad, color='green', label='Peaje')
    plt.xlabel('y')
    plt.ylabel(r'$p_D(\theta > \theta_0 \mid y)$')
    plt.show()

    print("Rechazo la idea de pagar el peaje con", r'\alpha = 0,05', "cuando y < ", x_values[np.sum(probabilidad < 0.05)], " minutos.")
    print("Rechazo la idea de pagar el peaje con", r'\alpha = 0,1', "cuando y < ", x_values[np.sum(probabilidad < 0.1)], " minutos.")