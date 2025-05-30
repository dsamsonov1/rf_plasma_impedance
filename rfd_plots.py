from rfd_conf import cf
import numpy as np
import matplotlib.pyplot as plt
from rfd_utils import *

def plot_Te():
    en_range = np.arange(1, 7.1, 0.1)

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1, 1, 1)
    minor_ticks = np.arange(0, 101, 4)
    ax.set_xticks(minor_ticks, minor=True)
    plt.plot(en_range, cf["Vp"] * cf["ng"] * Kiz(en_range), label='Kiz side')
    plt.plot(en_range, (cf["Ae"] + cf["Ag"]) * u_Bohm(en_range), label='Bohm side')
    plt.axvline(cf["Te"], color='cyan', linestyle=':')
    plt.grid(which='minor', linestyle='--')
    plt.grid(which='major', linestyle='--')
    plt.minorticks_on()
    plt.yscale('log')
    plt.xlabel('Temperature [eV]')
    plt.ylabel('Particle balance [m^3/s]')
    plt.legend()
    plt.show()

def plot_K():
    en_range = np.arange(1, 7.1, 0.1)

    plt.figure(figsize=(10, 10))
    plt.plot(en_range, (Kel(en_range) * cf["eps_el"]) / Kiz(en_range), linestyle=':', label='elastic')
    plt.plot(en_range, cf["eps_iz"] * np.ones(en_range.size), linestyle=':', label='ionization')
    plt.plot(en_range, (Kex(en_range) * cf["eps_ex"]) / Kiz(en_range), linestyle=':', label='excitation')
    plt.plot(en_range, eps_c(en_range), label='total')
    plt.yscale('log')
    plt.xlabel('Temperature [eV]')
    plt.ylabel('Ionization cost [eV]')
    plt.legend()
    plt.grid()
