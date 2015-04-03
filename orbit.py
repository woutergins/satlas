"""
.. module:: orbit
    :platform: Windows
    :synopsis: Molecular dynamics simulation of orbits in (ion) traps

.. moduleauthor:: Wouter Gins <wouter.gins@fys.kuleuven.be>
"""

import numpy as np
from scipy import integrate


def paulForce(U, V, freq, r0, z0, t, y):
    fx = -(V*np.cos(freq*t)/(r0*r0)-U/(z0*z0)) * y[0]
    fy = -(-V*np.cos(freq*t)/(r0*r0)-U/(z0*z0)) * y[1]
    fz = -2 * U * y[2] / (z0 * z0)
    return np.hstack((fx, fy, fz))


def paulSolver(U, V, freq, r0, z0, t):
    def rhs(y, t):
        dydt = paulForce(U, V, freq, r0, z0, t, y[:-3])
        dydt = np.hstack((y[-3:], dydt))
        return dydt
    y0 = np.array([0.5, 0.5, 1.5, 0, 0, 0.0])
    sol = integrate.odeint(rhs, y0, t)
    return sol


class PaulSolver(object):

    def __init__(self, U, V, freq, r0, z0):
        super(PaulSolver, self).__init__()
        self.U = U
        self.V = V
        self.freq = freq
        self.r0 = r0
        self.z0 = z0
        self.pos = np.array([0.1, 0.5, 1.5])
        self.vel = np.array([0, 0, -0.5])
        self.prevpos = None
        self.prevvel = None
        self.dt = 0.001

    def force(self, t, pos):
        return paulForce(self.U, self.V, self.freq, self.r0, self.z0, t, pos)

    def simulate(self, t):
        steps = int(t / self.dt)
        print(steps)
        dt = self.dt

        y = np.zeros((steps, 3))
        y[0, :] = self.pos
        self.prevvel = self.vel
        self.prevpos = y[0, :]
        accel = self.force(0, self.prevpos)

        for i in range(1, steps):
            y[i, :] = self.prevpos + self.prevvel * dt + accel * dt * dt * 0.5
            vdum = self.prevvel + accel * dt * 0.5
            accel = self.force((i + 1) * dt, self.prevpos)
            self.prevvel = vdum + accel * dt * 0.5

        return y
