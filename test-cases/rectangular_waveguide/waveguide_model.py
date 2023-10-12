# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 21:58:45 2023

@author: D. Loukrezis
"""

import numpy as np


def debye2_single_freq(freq=20.,
                       width=30., height=3., fill_l=7., offset=5.,
                       epss1=2., epss2=2.2, eps8=1.,
                       tau_eps_const1=1., tau_eps_const2=1.1,
                       mues1=2., mues2=3., mue8=1.,
                       tau_mue_const1=1.,tau_mue_const2=2.,
                       res='abs'):
    """Input parameters:
    1. frequency,
    2. width,
    3. height,
    4. filling length,
    5. vacuum offset,
    6. static permittivity 1,
    7. static permittivity 2,
    8. high-frequency permittivity,
    9. permittivity relax. time constant 1
    10. permittivity relax. time constant 2,
    11. static permeability 1,
    12. static permeability 2,
    13. high-frequency permeability,
    14. permeability relax. time constant 1,
    15. permeability relax. time constant 2.

    Output: S11 parameter.
    res = 'dB' (default) or 'abs' or 'complex' or 'im' or 're' """

    # scalings
    freq = freq * 1e9      # scale to GHz
    width = width * 1e-3   # scale to mm
    height = height * 1e-3 # scale to mm
    fill_l = fill_l * 1e-3 # scale to mm
    offset = offset * 1e-3 # scale to mm

    # constants
    c0 = 299792458 # speed of light in vacuum
    mue0 = 4 * np.pi * 1e-7 # vacuum permeability
    eps0 = 1.0 / (mue0 * c0**2.0) # vacuum permitivity

    # frequency
    w  = 2*np.pi*freq # omega
    w0 = 2*np.pi*20e9
    tau = 1.0 / w0 # medium's relaxation time


    # material constants (Debye2 relaxation model)
    #eps8 = 1.0 # HF permittivity
    #epss = 2.0 # static permittivity
    tau_eps1 = tau_eps_const1 * tau # relaxation time of medium
    tau_eps2 = tau_eps_const2 * tau # relaxation time of medium
    # relative permitivity of the medium for every frequency sample
    epsr = eps8 + (epss1 - eps8)*(1.0 + 1.0j*w*tau_eps1)**(-1) \
                + (epss2 - eps8)*(1.0 + 1.0j*w*tau_eps2)**(-1)

    #mue8 = 1.0 # HF permeability
    #mues = 2.4 # static permeability
    tau_mue1 = tau_mue_const1 * tau # relaxation time of medium
    tau_mue2 = tau_mue_const2 * tau
    # relative permeability of the medium for every frequency sample
    muer = mue8 + (mues1 - mue8)*(1 + 1.0j*w*tau_mue1)**(-1) \
                + (mues2 - mue8)*(1 + 1.0j*w*tau_mue2)**(-1)

    # wavenumbers
    k0 = w/c0 # in vacuum
    k = k0*np.sqrt(muer*epsr) # in material

    # propagation constants
    ky = np.pi / width # in y-direction
    kz = np.sqrt(k0**2 - ky**2) # in z-direction for vacuum
    kz1 = np.sqrt(k**2 - ky**2) # in z-direction for the debye-material

    # wave impedance for waveguide (TE-wave)
    Z0 = mue0*w/kz # for vacuum
    Z1 = mue0*muer*w/kz1 # for the filling

    # coefficient used to determine the reflection and transmission
    # coefficients from the equation system
    A = (-Z0/Z1 + 1)*np.exp(-1j*(offset + fill_l)*kz1)
    B = (Z0/Z1 + 1)*np.exp(1j*(offset + fill_l)*kz1)

    # reflection coefficient at the surface between the 2 materials
    r2 = (2*Z1*np.exp(-1j*offset*kz)) / ((Z1-Z0)*(np.exp(1j*kz1*offset) - \
                    np.exp(1j*(2*fill_l + offset)*kz1)*((Z1+Z0)/(Z1-Z0))**2))
    # transmission coefficient at the surface between the 2 materials
    t2 = -r2*B/A
    # reflection coefficient at the input port
    r1 = 0.5*np.exp(-1j*offset*kz)*(t2*np.exp(-1j*offset*kz1)*(-Z0/Z1 + 1) + \
                  r2*np.exp(1j*offset*kz1)*(Z0/Z1+1))

    # calculation of the S-parameters using the definition
    if res == 'complex':
        S11complex = r1
        return S11complex
    elif res == 'abs':
        S11abs = np.abs(r1)
        return S11abs
    elif res == 'dB':
        S11dB = 20*np.log10(np.abs(r1))
        return S11dB
    elif res == 'im':
        S11imag = r1.imag
        return S11imag
    elif res == 're':
        S11real = r1.real
        return S11real
    else:
        print("Not acceptable 'res' input.")
        return


def debye2_broadband(fmin=5, fmax=30., fsamples=1001,
                     width=30., height=3., fill_l=7., offset=5.,
                     epss1=2., epss2=2.2, eps8=1.,
                     tau_eps_const1=1., tau_eps_const2=1.1,
                     mues1=2., mues2=3., mue8=1.,
                     tau_mue_const1=1.,tau_mue_const2=2.,
                     res='abs'):
    """Input parameters:
    1. width,
    2. height,
    3. filling length,
    4. vacuum offset,
    5. static permittivity 1,
    6. static permittivity 2,
    7. high-frequency permittivity,
    8. permittivity relax. time constant 1
    9. permittivity relax. time constant 2,
    10. static permeability 1,
    11. static permeability 2,
    12. high-frequency permeability,
    13. permeability relax. time constant 1,
    14. permeability relax. time constant 2.

    Output: S11 parameter.
    res = 'dB' (default) or 'abs' or 'complex' or 'im' or 're' """

    # get vector of operating frequencies
    freq = np.linspace(fmin, fmax, fsamples)

    # scalings
    freq = freq * 1e9      # scale to GHz
    width = width * 1e-3   # scale to mm
    height = height * 1e-3 # scale to mm
    fill_l = fill_l * 1e-3 # scale to mm
    offset = offset * 1e-3 # scale to mm

    # constants
    c0 = 299792458 # speed of light in vacuum
    mue0 = 4 * np.pi * 1e-7 # vacuum permeability
    eps0 = 1.0 / (mue0 * c0**2.0) # vacuum permitivity

    # frequency
    w  = 2*np.pi*freq # omega
    w0 = 2*np.pi*20e9
    tau = 1.0 / w0 # medium's relaxation time


    # material constants (Debye2 relaxation model)
    #eps8 = 1.0 # HF permittivity
    #epss = 2.0 # static permittivity
    tau_eps1 = tau_eps_const1 * tau # relaxation time of medium
    tau_eps2 = tau_eps_const2 * tau # relaxation time of medium
    # relative permitivity of the medium for every frequency sample
    epsr = eps8 + (epss1 - eps8)*(1.0 + 1.0j*w*tau_eps1)**(-1) \
                + (epss2 - eps8)*(1.0 + 1.0j*w*tau_eps2)**(-1)

    #mue8 = 1.0 # HF permeability
    #mues = 2.4 # static permeability
    tau_mue1 = tau_mue_const1 * tau # relaxation time of medium
    tau_mue2 = tau_mue_const2 * tau
    # relative permeability of the medium for every frequency sample
    muer = mue8 + (mues1 - mue8)*(1 + 1.0j*w*tau_mue1)**(-1) \
                + (mues2 - mue8)*(1 + 1.0j*w*tau_mue2)**(-1)

    # wavenumbers
    k0 = w/c0 # in vacuum
    k = k0*np.sqrt(muer*epsr) # in material

    # propagation constants
    ky = np.pi / width # in y-direction
    kz = np.sqrt(k0**2 - ky**2) # in z-direction for vacuum
    kz1 = np.sqrt(k**2 - ky**2) # in z-direction for the debye-material

    # wave impedance for waveguide (TE-wave)
    Z0 = mue0*w/kz # for vacuum
    Z1 = mue0*muer*w/kz1 # for the filling

    # coefficient used to determine the reflection and transmission
    # coefficients from the equation system
    A = (-Z0/Z1 + 1)*np.exp(-1j*(offset + fill_l)*kz1)
    B = (Z0/Z1 + 1)*np.exp(1j*(offset + fill_l)*kz1)

    # reflection coefficient at the surface between the 2 materials
    r2 = (2*Z1*np.exp(-1j*offset*kz)) / ((Z1-Z0)*(np.exp(1j*kz1*offset) - \
                    np.exp(1j*(2*fill_l + offset)*kz1)*((Z1+Z0)/(Z1-Z0))**2))
    # transmission coefficient at the surface between the 2 materials
    t2 = -r2*B/A
    # reflection coefficient at the input port
    r1 = 0.5*np.exp(-1j*offset*kz)*(t2*np.exp(-1j*offset*kz1)*(-Z0/Z1 + 1) + \
                  r2*np.exp(1j*offset*kz1)*(Z0/Z1+1))

    # calculation of the S-parameters using the definition
    if res == 'complex':
        S11complex = r1
        return S11complex
    elif res == 'abs':
        S11abs = np.abs(r1)
        return S11abs
    elif res == 'dB':
        S11dB = 20*np.log10(np.abs(r1))
        return S11dB
    elif res == 'im':
        S11imag = r1.imag
        return S11imag
    elif res == 're':
        S11real = r1.real
        return S11real
    else:
        print("Not acceptable 'res' input.")
        return