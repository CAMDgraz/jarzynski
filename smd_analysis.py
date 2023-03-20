#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniel Platero-Rochart [daniel.platero-rochart@medunigraz.at]
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import heapq
import argparse
import glob
import os


def parse_arguments():
    """
    Parse argument from cli.

    Returns:
        user_inputs: argparse.Namespace

    """
    desc = '\nAnalizys of sMD results using Jarzynski equality'
    parser = argparse.ArgumentParser(prog='smd_analysis',
                                     description=desc,
                                     allow_abbrev=False,
                                     usage='%(prog)s -jobs_list (or -ext) '
                                           '[options]')
    # -- Arguments: Input -----------------------------------------------------
    inp = parser.add_argument_group(title='Input Options')
    inp.add_argument('-jobs_list', dest='jobs_list', action='store',
                     help='File with the sMD jobs', type=str,
                     required=False, default=None)
    inp.add_argument('-ext', dest='file_extension', action='store',
                     help='If not -jobs_list provided you must supply the\
                     extension of the jobs file (e.g. dat)', type=str,
                     required=False, default=None)
    inp.add_argument('-time', dest='time', action='store',
                     help='Simulation time in ps (default: 2)', type=float,
                     required=False, default=2)
    inp.add_argument('-tstep', dest='time_step', action='store',
                     help='Time step of the simulation in ps (default: 0.02)',
                     type=float, required=False, default=0.02)
    inp.add_argument('-T', dest='temperature', action='store',
                     help='Temperature of the simulation in K (default: 300)',
                     type=float, required=False, default=300)
    inp.add_argument('-check', dest='check_jobs', action='store',
                     help='Identify incomplete/missing jobs (default: False)',
                     type=bool, required=False, default=False,
                     choices=[True, False])
    # -- Arguments: Jarzynski parameters --------------------------------------
    jarz = parser.add_argument_group(title='Jarzynski options')
    jarz.add_argument('-jarz_type', dest='jarz_type', action='store',
                      help='Type of Jarzynski to perform ' +
                      '(default: exponential)',
                      type=str, required=False, default='exponential',
                      choices=['cumulant', 'exponential'])
    # -- Arguments: Output ----------------------------------------------------
    output = parser.add_argument_group(title='Output options')
    output.add_argument('-out', dest='output', action='store',
                        help='Output directory (default: ./)', type=str,
                        required=False, default='./')

    args = parser.parse_args()
    return args, parser


def general_canvas(figsize, dpi):
    """
    Customization of plots

    Returns:
        None
    """
    mpl.rc('figure', figsize=figsize, dpi=dpi)
    mpl.rc('xtick', direction='in', top=False)
    mpl.rc('xtick.major', top=False)
    mpl.rc('xtick.minor', top=False)
    mpl.rc('ytick', direction='in', right=True)
    mpl.rc('ytick.major', right= False)
    mpl.rc('ytick.minor', right=False)
    mpl.rc('axes', labelsize=20)
    plt.rcParams['axes.autolimit_mode'] = 'data'
    mpl.rc('lines', linewidth=2)
    mpl.rc('font', family='monospace', size=20)
    mpl.rc('grid', alpha=0.5, color='gray', linewidth=1, linestyle='--')

    return


def exponential_average(works, time_vect, njobs, T, kb):
    """
    Perform exponential average

    Parameters:
        works : np.array
            Array containing the pulling works of the trajectories
        time_vect : np.array
            Array containing the time values (0:time:time_step)
        njobs : int
            Number of sMD trajectories
        T : float
            Temperature of the simulations
        kb : float
            Boltzmann constant
    Return:
        energy : np.array
            Array containing the energies resulting from JE
        avg_work : np.array
            Array containing the average work for each time step

    """
    avg_work = np.zeros(len(time_vect))
    exp_work = np.zeros(len(time_vect))
    for work_vect in works:
        exp = np.exp(-work_vect/(kb*T))
        exp_work += exp
        avg_work += work_vect

    avg_work = avg_work/njobs
    avg_exp_work = exp_work/njobs
    energy = -kb*T*np.log(avg_exp_work)

    return energy, avg_work


def cumulant_expansion(works, time_vect, njobs, T, kb):
    """
    Unbiased 2nd order cumulant expansion

    Parameters:
        works : np.array
            Array containing the pulling works of the trajectories
        time_vect : np.array
            Array containing the time values (0:time:time_step)
        njobs : int
            Number of sMD trajectories
        T : float
            Temperature of the simulations
        kb : float
            Boltzmann constant

    Return:
        energy : np.array
            Array containing the energies resulting from JE
        avg_work : np.array
            Array containing the average work for each time step

    """
    M = np.shape(works)[0]
    mean = np.mean(works, axis=0)                                   # <W>
    beta = 1/(kb*T)                                                 # ß
    mean_cuad = np.mean(works**2, axis=0)                           # <W^2>
    energy = mean - (beta/2)*(M/(M-1))*(mean_cuad - mean**2)        # F

    avg_work = np.zeros(len(time_vect))
    for work_vect in works:
        avg_work += work_vect

    avg_work = avg_work/njobs

    return energy, avg_work


def find_maximum(energy, time_vect):
    """
    Find energy maximum value using a gradient approach

    Parameters:
        energy : np.array
            Array containing the enrgies resulting from JE
        time_vect : np.array
            Array containing the time values (0:time:time_step)

    Return:
        m_energy : float
            Maximum energy value
        m_step : float
            Time step where the m_energy is found

    """
    maximum = []
    heapq.heapify(maximum)

    gradients = np.gradient(energy, time_vect)
    for grad in range(0, len(gradients) - 1):
        if (gradients[grad] > 0) and (gradients[grad+1] < 0):
            heapq.heappush(maximum, (energy[grad], grad))

    try:
        m_energy, m_step = heapq.nlargest(1, maximum)[0]
    except IndexError:
        m_energy, m_step = None, None
    return m_energy, m_step


def check_jobs(jobs, time, time_step):
    """
    Function for check incomplete or missing jobs

    Parameters:
        jobs : list
            List containing the path to the output files of the sMD
        time : float
            Simulation time in ps
        time_step : float
            Time step used in the simulations

    Return:
        incomplete_jobs : list
            List containing the path to the incomplete or missing output files
            of the sMD

    """
    length_QMMM = len(np.arange(0, time, time_step))

    incomplete_jobs = []
    for job in jobs:
        try:
            job_ = pd.read_csv(job, delim_whitespace=True, skiprows=3,
                               header=None, nrows=length_QMMM)
        except pd.errors.EmptyDataError:
            incomplete_jobs.append(job)
        except FileNotFoundError:
            incomplete_jobs.append(job)
        if len(job_) != length_QMMM:
            incomplete_jobs.append(job)

    return incomplete_jobs


if __name__ == '__main__':
    args, parser = parse_arguments()
    # =========================================================================
    # Checking args
    # =========================================================================
    if (args.jobs_list is not None) and (args.file_extension is not None):
        print('\n\n>>> WARNING !!!')
        print('Both, jobs list and extension provided, jobs list will be used')
    elif (args.jobs_list is None) and (args.file_extension is None):
        print('\n\n>>> Fatal Error !!!: ' +
              'Neither jobs list nor extension provided')
        parser.print_help()
        exit()

    # -- general variables ----------------------------------------------------

    kb = 0.001982923700                     # Boltzmann constant (kcal/mol*T)

    time_vect = np.arange(0, args.time, args.time_step)
    length_QMMM = len(time_vect)

    # -- Reading input --------------------------------------------------------
    if args.jobs_list is not None:
        print('** Reading jobs from {} list\n'.format(args.jobs_list))
        with open(args.jobs_list, 'r') as inp:
            jobs = inp.read().splitlines()
    elif args.file_extension is not None:
        print(
              '** Reading jobs with extension: {}'.format(args.file_extension)
             )
        jobs = glob.glob('*.{}'.format(args.file_extension))

    # -- checking job files ---------------------------------------------------
    if args.check_jobs is True:
        print('** Check jobs set to: True\n** Checking jobs')
        incomplete_smd = check_jobs(jobs, args.time, args.time_step)
        if len(incomplete_smd) != 0:
            for incomplete in incomplete_smd:
                print('>> Incomplete job or missing ' +
                      'file: {}'.format(incomplete))
    else:
        print('** Check jobs set to: False\n')
        incomplete_smd = []

    # -- reading complete job files -------------------------------------------
    actual_jobs = list(set(jobs)-set(incomplete_smd))
    pulling_works = np.zeros((len(actual_jobs), len(time_vect)))
    for idx, job in enumerate(actual_jobs):
        try:
            job_ = pd.read_csv(job, delim_whitespace=True, skiprows=3,
                               header=None, nrows=length_QMMM)

            pulling_works[idx] = np.asarray(job_[job_.columns[-1]])
        except (pd.errors.EmptyDataError, FileNotFoundError):
            raise ValueError('>>> File error !!!: '
                             'set -check True to discard incomplete ' +
                             'or missing files')
    # =========================================================================
    # Performing Jarzynski and finding maximum
    # =========================================================================
    if args.jarz_type == 'cumulant':
        print('** Performing cumulant 2nd order Jarzynski')
        energy, avg_work = cumulant_expansion(pulling_works, time_vect,
                                              len(actual_jobs),
                                              args.temperature, kb)
    elif args.jarz_type == 'exponential':
        print('** Performing exponential average Jarzynski')
        energy, avg_work = exponential_average(pulling_works, time_vect,
                                               len(actual_jobs),
                                               args.temperature, kb)
    print('** Finding maximum value')
    m_energy, m_step = find_maximum(energy, time_vect)
    if (m_energy is None):
        print('>> No maximum found :(')
    else:
        print('Maximum found in: '
              ' {:.2f}ps --> {:.4f}kcal/mol'.format(time_vect[m_step],
                                                    m_energy))
    # =========================================================================
    # Plotting and saving results
    # =========================================================================
    if args.output != './':
        try:
            os.mkdir(args.output)
        except FileExistsError:
            raise Exception('{} directory exists.'.format(args.output) +
                            'Specifx another location or rename it')

    general_canvas([12, 8], 300)
    fig, ax = plt.subplots()

    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.25))
    ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.125))
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(10))
    ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(5))

    ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:.2f}'))

    ax.set_xlabel(r'Time $(ps)$')
    ax.set_ylabel(r'Free Energy $(kcal\ mol^{-1})$')
    ax.set_xlim(0, args.time)

    for work_vect in pulling_works:
        ax.plot(time_vect, work_vect, alpha=0.2)
    ax.plot(time_vect, energy, color='black')

    fig.savefig('{}/FE.png'.format(args.output))
    plt.close(fig)

    with open('{}/time_energy.txt'.format(args.output), 'w') as out:
        out.write('Time(ps)  AvgWork(kcal/mol)  Free_Energy(kcal/mol)\n')
        for t, w, e in zip(time_vect, avg_work, energy):
            out.write('{:>8.2f}  {:>17.4f}  {:>21.4f}\n'.format(t, w, e))

    print('** Done!!!')
