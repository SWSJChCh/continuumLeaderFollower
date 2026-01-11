import sys
import time
import numpy as np
from scipy.signal import fftconvolve

'''
Helper functions
'''

def cflDt(h, dF, safety=0.01):
    # CFL timestep restriction
    lims = []

    if dF > 0:
        lims.append(h * h / (2.0 * dF))

    if len(lims) == 0:
        return 1.0

    return safety * min(lims)


def laplacianNeumann(Z, h):
    lap = np.zeros_like(Z)

    lap[1:-1] = (Z[:-2] + Z[2:] - 2.0 * Z[1:-1]) / (h * h)
    lap[0]    = 2.0 * (Z[1]  - Z[0])  / (h * h)
    lap[-1]   = 2.0 * (Z[-2] - Z[-1]) / (h * h)

    return lap


def expKernel(L, h, interactionLength):
    threeSigma = 3.0 * interactionLength
    boxCap = max(0.0, L / 2.0 - h)
    radius = min(threeSigma, boxCap)

    nPts = max(1, int(np.floor(radius / h)))
    x = np.arange(-nPts, nPts + 1) * h

    ker = np.exp(-0.5 * (np.abs(x) / interactionLength) ** 2)

    # Normalise so that sum * h ≈ interaction length
    ker *= interactionLength / (np.sum(ker) * h)

    return ker


def nonlocalConvolution(Z, kernel, h):
    return h * fftconvolve(Z, kernel, mode='same')


def shiftVacuum(arr, n):
    if n == 0:
        return arr.copy()

    out = np.zeros_like(arr)

    if n > 0:
        if n < arr.size:
            out[n:] = arr[:-n]
    else:
        n = -n
        if n < arr.size:
            out[:-n] = arr[n:]

    return out

'''
Model class
'''

class explicitNeuralCrest1D:
    '''
    Explicit Forward–Euler solver.

    Leaders:
        Pure translation via integer grid-cell shifts.

    Followers:
        Diffusion + nonlocal taxis with volume filling.
    '''

    def __init__(
        self,
        l0, f0,
        vL, dF,
        muFL, muFF,
        K,
        xiFL, xiFF,
        L, meshPoints,
        timeSpan, timeEvaluations,
        dt=None,
        safety=0.01,
        clipNonneg=True,
        boundaryMode='vacuum'
    ):
        self.l0 = l0
        self.f0 = f0

        self.vL = float(vL)
        self.dF = float(dF)

        self.muFL = float(muFL)
        self.muFF = float(muFF)

        self.K = float(K)

        self.xiFL = float(xiFL)
        self.xiFF = float(xiFF)

        self.L = float(L)
        self.N = int(meshPoints)

        self.x = np.linspace(0.0, self.L, self.N)

        if self.N > 1:
            self.h = self.x[1] - self.x[0]
        else:
            self.h = self.L

        self.t0, self.tf = timeSpan
        self.tEval = np.asarray(timeEvaluations, dtype=float)

        self.kernelFL = expKernel(self.L, self.h, self.xiFL)
        self.kernelFF = expKernel(self.L, self.h, self.xiFF)

        self.l, self.f = self.defaultInitialCondition()

        self.clipNonneg = clipNonneg
        self.boundaryMode = boundaryMode

        # Accumulated leader displacement (grid-cell units)
        self.shiftAccum = 0.0

        if dt is None:
            self.dt = cflDt(self.h, self.dF, safety)
        else:
            self.dt = float(dt)


    def defaultInitialCondition(self):
        x = self.x

        l = np.zeros_like(x)
        f = np.zeros_like(x)

        xLower = 65.0
        xUpper = 870.0
        delta  = 5.0

        f[:] = 0.5 * self.f0 * (
            np.tanh((x - xLower) / delta)
            - np.tanh((x - xUpper) / delta)
        )

        x0 = 895.0
        sigma = 12.0

        base = np.exp(-0.5 * ((x - x0) / sigma) ** 2)
        l[:] = (base / base.max()) * f.max()

        return l, f


    def shiftLeaders(self):
        if self.vL == 0.0:
            return

        self.shiftAccum += self.vL * self.dt / self.h
        nShift = int(np.trunc(self.shiftAccum))

        if nShift != 0:
            if self.boundaryMode == 'wrap':
                self.l = np.roll(self.l, nShift)
            else:
                self.l = shiftVacuum(self.l, nShift)

            self.shiftAccum -= nShift


    def step(self):
        # Leader update
        self.shiftLeaders()

        l = self.l
        f = self.f

        C = (
            self.muFF * nonlocalConvolution(f, self.kernelFF, self.h)
            + self.muFL * nonlocalConvolution(l, self.kernelFL, self.h)
        )

        lapF = laplacianNeumann(f, self.h)

        # Finite-volume taxis
        P = self.K - l - f

        dCdxFace = (C[1:] - C[:-1]) / self.h
        PFace = 0.5 * (P[1:] + P[:-1])
        uFace = PFace * dCdxFace

        fUp = np.where(uFace >= 0.0, f[:-1], f[1:])

        FFace = np.zeros(self.N + 1)
        FFace[1:-1] = fUp * uFace

        dFluxDx = (FFace[1:] - FFace[:-1]) / self.h

        dfDt = self.dF * lapF - dFluxDx

        self.f = f + self.dt * dfDt

        if self.clipNonneg:
            np.maximum(self.f, 0.0, out=self.f)


    def simulateWithProgress(self, yFile, tFile):
        tGrid = self.tEval
        outY = np.empty((2 * self.N, tGrid.size), dtype=float)

        tCurr = self.t0

        outY[:self.N, 0] = self.l
        outY[self.N:, 0] = self.f

        for k in range(1, tGrid.size):
            tTarget = tGrid[k]

            while tCurr + self.dt < tTarget - 1e-15:
                self.step()
                tCurr += self.dt

            dtLast = tTarget - tCurr

            if dtLast > 1e-15:
                dtSave = self.dt
                self.dt = dtLast
                self.step()
                self.dt = dtSave
                tCurr = tTarget

            outY[:self.N, k] = self.l
            outY[self.N:, k] = self.f

        np.savetxt(yFile, outY, delimiter=',')
        np.savetxt(tFile, tGrid, delimiter=',')


'''
CLI wrapper
'''

if __name__ == "__main__":

    l0 = 0.95
    f0 = 0.95

    vL = 1.0
    dF = 10.0

    muFL = int(sys.argv[1])
    muFF = int(sys.argv[2])

    xiFL = int(sys.argv[3])
    xiFF = int(sys.argv[4])

    K = 1.0
    L = 2000.0
    meshPoints = 20000

    timeSpan = (0.0, 12 * 60.0)
    timeEvaluations = np.linspace(timeSpan[0], timeSpan[1], 24)

    sim = explicitNeuralCrest1D(
        l0, f0,
        vL, dF,
        muFL, muFF,
        K,
        xiFL, xiFF,
        L, meshPoints,
        timeSpan, timeEvaluations,
        dt=None,
        safety=0.01,
        clipNonneg=True,
        boundaryMode='vacuum'
    )

    yFile = f"Y_mu_fl{muFL}_mu_ff{muFF}_xi_fl{xiFL}_xi_ff{xiFF}.txt"
    tFile = f"T_mu_fl{muFL}_mu_ff{muFF}_xi_fl{xiFL}_xi_ff{xiFF}.txt"

    sim.simulateWithProgress(yFile, tFile)
