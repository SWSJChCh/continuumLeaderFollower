'''
nonLocalAlignment.py
'''

import os
import time
import numpy as np
from numba import jit

#Run simulation and save outputs
lmbdVals = [2, 6, 10]
interactionVals = [10, 50, 100]

#Spatial and polarisation meshes
L = 2000.0                                          #Domain length (µm)
nx = 500                                            #Number of spatial mesh points
pMin, pMax = -1.0, 1.0                              #Minimum / maximum polarisation values
nPol = 100                                          #Number of polarisation mesh points

dx = L / (nx - 1)                                   #Spatial step on mesh
dp = (pMax - pMin) / (nPol - 1)                     #Polarisation step on mesh

#Model parameters
v0 = 1.0                                            #Maximum cell velocity (µm/min)
Df = 1.0                                            #Follower cell spatial diffusion constant
Dp = 1e-3                                           #Follower cell polarisation diffusion constant
pLead = 1.0                                         #Polarisation of leader cells (fixed)

#Simulation parameters
timeStart = 0.0                                     #Simulation start time (min)
timeEnd = 12 * 60                                   #Simulation final time (min)
dt = 1e-4                                           #Timestep (min)
nSteps = int((timeEnd - timeStart) / dt)            #Number of timesteps in simulation
nRecord = 100                                       #Number of data points to record
recordIndices = np.linspace(0, nSteps, nRecord, dtype=np.int64)

#Leader displacement, measured in spatial mesh intervals, per timestep
leaderShiftIncrement = v0 * pLead * dt / dx

#Output parameters
outputFolder = '.'                                  #Directory in which output files are saved
printProgress = True                                #Print progress and total-mass diagnostics


'''
Construct a normalised Gaussian interaction kernel
'''
def gaussianKernel(dx, length, nSigma=3.0):
    #Number of mesh points on either side of the kernel centre
    half = int(np.ceil(nSigma * length / dx))
    #Spatial coordinates on the kernel support
    x = np.arange(-half, half + 1, dtype=np.float64) * dx
    #Gaussian interaction weights with the exact compact support |x| <= 3 xi
    kernel = np.exp(-0.5 * (x / length)**2)
    kernel[np.abs(x) > nSigma * length] = 0.0
    #Normalise the discrete kernel to have unit integral
    return kernel / (np.sum(kernel) * dx)


'''
Construct the follower and leader initial conditions
'''
def initialConditions(xGrid, pGrid, dp):
    #Follower slab parameters
    f0Init = 0.95
    xLower = 65.0
    xUpper = 870.0
    deltaX = 5.0

    #Leader Gaussian parameters
    x0L = 895.0
    sigmaL = 12.0

    #Follower density in physical space
    fX = 0.5 * f0Init * (np.tanh((xGrid - xLower) / deltaX) - \
                         np.tanh((xGrid - xUpper) / deltaX))

    #Leader density in physical space: stated amplitude l0 = 0.95
    base = np.exp(-0.5 * ((xGrid - x0L) / sigmaL)**2)
    l0 = f0Init * base

    #Follower Gaussian parameters in polarisation space
    p0 = 0.0
    sigmaP = 0.05

    #Gaussian polarisation distribution
    gaussP = np.exp(-0.5 * ((pGrid - p0) / sigmaP)**2)
    #Normalise to integrate to one on the discrete polarisation grid
    gaussP /= np.sum(gaussP) * dp

    #Construct separable follower density
    f0 = fX[:, np.newaxis] * gaussP[np.newaxis, :]

    #Return contiguous float64 arrays for Numba
    return np.ascontiguousarray(f0, dtype=np.float64), \
           np.ascontiguousarray(l0, dtype=np.float64)


'''
Record the complete follower and leader state
'''
@jit(nopython=True, cache=True, inline='always')
def recordState(recordArray, recordNumber, f, l):
    #Dimensions of follower density
    nxLocal = f.shape[0]
    nPolLocal = f.shape[1]
    #Index in packed state vector
    stateIndex = 0

    #Record follower density in row-major order
    for i in range(nxLocal):
        for j in range(nPolLocal):
            recordArray[recordNumber, stateIndex] = f[i, j]
            stateIndex += 1

    #Record leader density
    for i in range(nxLocal):
        recordArray[recordNumber, stateIndex] = l[i]
        stateIndex += 1


'''
Calculate the total follower and leader mass
'''
@jit(nopython=True, cache=True, inline='always')
def totalMass(f, l, dx, dp):
    #Dimensions of follower density
    nxLocal = f.shape[0]
    nPolLocal = f.shape[1]
    #Follower and leader mass
    massF = 0.0
    massL = 0.0

    #Integrate follower density over physical and polarisation space
    for i in range(nxLocal):
        for j in range(nPolLocal):
            massF += f[i, j]
        massL += l[i]

    #Return total cell mass
    return massF * dx * dp + massL * dx


'''
Run the sparse Forward Euler simulation with accumulated-displacement leader advection
'''
@jit(nopython=True, cache=True)
def simulateSparse(f0, l0, pGrid, absPGrid, oneMinusP2, uGrid, kernelW, \
                   lambda_, pLead, Df, Dp, dx, dp, dt, nSteps, \
                   recordIndices, leaderShiftIncrement, printProgress):
    #Mesh dimensions
    nxLocal = f0.shape[0]
    nPolLocal = f0.shape[1]
    #State dimensions
    fSize = nxLocal * nPolLocal
    stateDim = fSize + nxLocal
    nRecordLocal = recordIndices.shape[0]

    #Current and updated follower densities
    f = f0.copy()
    fNew = np.empty_like(f)
    #Leader density
    l = l0.copy()

    #Arrays for non-local polarity alignment
    weightedPolarity = np.empty(nxLocal, dtype=np.float64)
    weightedDensity = np.empty(nxLocal, dtype=np.float64)
    pTilde = np.empty(nxLocal, dtype=np.float64)
    #Polarity drift for one spatial row
    driftRow = np.empty(nPolLocal, dtype=np.float64)

    #Sparse output array
    recordArray = np.zeros((nRecordLocal, stateDim), dtype=np.float64)
    recordState(recordArray, 0, f, l)

    #Recording and diagnostic counters
    nextRecord = 1
    errorFlag = False
    accumulatedLeaderDisplacement = 0.0
    completedLeaderShifts = 0
    printInterval = max(1, nSteps // 10)
    nextPrint = printInterval

    #Frequently used constants
    invDx = 1.0 / dx
    invDp = 1.0 / dp
    invDx2 = invDx * invDx
    invDp2 = invDp * invDp
    absPLead = abs(pLead)
    kernelHalf = kernelW.shape[0] // 2

    #Main Forward Euler loop
    for step in range(1, nSteps + 1):

        #Calculate local density, polarity, and polarity-weighted density
        for i in range(nxLocal):
            followerDensity = 0.0
            followerPolarity = 0.0
            followerAbsPolarity = 0.0

            for j in range(nPolLocal):
                fValue = f[i, j]
                followerDensity += fValue
                followerPolarity += fValue * pGrid[j]
                followerAbsPolarity += fValue * absPGrid[j]

            rho = followerDensity * dp + l[i]
            q = followerPolarity * dp + l[i] * pLead
            cNumerator = followerAbsPolarity * dp + l[i] * absPLead

            if rho > 0.0:
                c = cNumerator / rho
            else:
                c = 0.0

            weightedPolarity[i] = c * q
            weightedDensity[i] = c * rho

        #Calculate the two zero-padded non-local convolutions
        for i in range(nxLocal):
            qTilde = 0.0
            mTilde = 0.0

            sourceStart = max(0, i - kernelHalf)
            sourceEnd = min(nxLocal - 1, i + kernelHalf)

            for sourceIndex in range(sourceStart, sourceEnd + 1):
                #Kernel reversal reproduces np.convolve / jnp.convolve
                kernelIndex = i - sourceIndex + kernelHalf
                weight = kernelW[kernelIndex]
                qTilde += weightedPolarity[sourceIndex] * weight
                mTilde += weightedDensity[sourceIndex] * weight

            qTilde *= dx
            mTilde *= dx

            if mTilde > 0.0:
                pTilde[i] = qTilde / mTilde
            else:
                pTilde[i] = 0.0

        #Calculate the follower update at every physical and polarisation point
        for i in range(nxLocal):

            #Polarity drift at mesh points for this spatial row
            for j in range(nPolLocal):
                driftRow[j] = lambda_ * oneMinusP2[j] * \
                              (pTilde[i] - pGrid[j])

            for j in range(nPolLocal):
                fCentre = f[i, j]

                #Follower advection in physical space
                if i == 0:
                    fluxXLeft = 0.0
                else:
                    if uGrid[j] >= 0.0:
                        fluxXLeft = uGrid[j] * f[i - 1, j]
                    else:
                        fluxXLeft = uGrid[j] * fCentre

                if i == nxLocal - 1:
                    fluxXRight = 0.0
                else:
                    if uGrid[j] >= 0.0:
                        fluxXRight = uGrid[j] * fCentre
                    else:
                        fluxXRight = uGrid[j] * f[i + 1, j]

                dfDtX = -(fluxXRight - fluxXLeft) * invDx

                #Follower advection in polarisation space
                if j == 0:
                    fluxPLeft = 0.0
                else:
                    driftLeft = 0.5 * (driftRow[j - 1] + driftRow[j])
                    if driftLeft >= 0.0:
                        fluxPLeft = driftLeft * f[i, j - 1]
                    else:
                        fluxPLeft = driftLeft * fCentre

                if j == nPolLocal - 1:
                    fluxPRight = 0.0
                else:
                    driftRight = 0.5 * (driftRow[j] + driftRow[j + 1])
                    if driftRight >= 0.0:
                        fluxPRight = driftRight * fCentre
                    else:
                        fluxPRight = driftRight * f[i, j + 1]

                dfDtP = -(fluxPRight - fluxPLeft) * invDp

                #Follower diffusion in physical space
                if i == 0:
                    lapX = (f[i + 1, j] - fCentre) * invDx2
                elif i == nxLocal - 1:
                    lapX = (f[i - 1, j] - fCentre) * invDx2
                else:
                    lapX = (f[i + 1, j] - 2.0 * fCentre + \
                            f[i - 1, j]) * invDx2

                #Follower diffusion in polarisation space
                if j == 0:
                    lapP = (f[i, j + 1] - fCentre) * invDp2
                elif j == nPolLocal - 1:
                    lapP = (f[i, j - 1] - fCentre) * invDp2
                else:
                    lapP = (f[i, j + 1] - 2.0 * fCentre + \
                            f[i, j - 1]) * invDp2

                #Forward Euler update in the same term order as jaxPDE.py
                dfDt = dfDtP + Dp * lapP + Df * lapX + dfDtX
                updatedValue = fCentre + dt * dfDt
                fNew[i, j] = updatedValue

                #Detect numerical instability without copying the state to Python
                if not errorFlag and not np.isfinite(updatedValue):
                    errorFlag = True

        #Exchange old and updated follower arrays
        fTemporary = f
        f = fNew
        fNew = fTemporary

        #Accumulate leader displacement and shift whenever floor(S) increases
        accumulatedLeaderDisplacement += leaderShiftIncrement
        requiredLeaderShifts = int(np.floor(accumulatedLeaderDisplacement + 1e-12))

        while completedLeaderShifts < requiredLeaderShifts:
            for i in range(nxLocal - 1, 0, -1):
                l[i] = l[i - 1]
            l[0] = 0.0
            completedLeaderShifts += 1

        #Record the state at the requested sparse output times
        if nextRecord < nRecordLocal and step == recordIndices[nextRecord]:
            recordState(recordArray, nextRecord, f, l)
            nextRecord += 1

        #Print progress and mass diagnostics at ten equally spaced times
        if printProgress and (step == nextPrint or step == nSteps):
            percentage = int((step / nSteps) * 100.0)
            print('Progress:', percentage, '% (step', step, '/', nSteps, ')')
            print('Total mass:', totalMass(f, l, dx, dp))
            nextPrint += printInterval

    #Return recorded state and numerical-error flag
    return recordArray, errorFlag


'''
Run all parameter combinations and save outputs
'''
def runSimulations():
    #Create output directory when required
    os.makedirs(outputFolder, exist_ok=True)

    #Physical and polarisation grids
    xGrid = np.linspace(0.0, L, nx, dtype=np.float64)
    pGrid = np.linspace(pMin, pMax, nPol, dtype=np.float64)

    #Precomputed polarity arrays
    absPGrid = np.abs(pGrid)
    oneMinusP2 = 1.0 - pGrid**2
    uGrid = v0 * pGrid

    #Initial follower and leader densities
    f0, l0 = initialConditions(xGrid, pGrid, dp)

    #Run parameter sweep
    for z in lmbdVals:
        for q in interactionVals:
            lambda_ = float(z)
            interactionLength = float(q)

            #Construct interaction kernel
            kernelW = gaussianKernel(dx, interactionLength)

            #Simulation information
            print('')
            print('Running sparse Numba simulation with accumulated-displacement leader advection')
            print('using Forward Euler...')
            print('lambda =', lambda_, ', interaction length =', \
                  interactionLength, 'µm')

            #Run simulation
            startTime = time.perf_counter()
            trajSparse, errFlag = simulateSparse(
                f0,
                l0,
                pGrid,
                absPGrid,
                oneMinusP2,
                uGrid,
                kernelW,
                lambda_,
                pLead,
                Df,
                Dp,
                dx,
                dp,
                dt,
                nSteps,
                recordIndices,
                leaderShiftIncrement,
                printProgress
            )
            runTime = time.perf_counter() - startTime

            #Calculate final follower density and mean local polarisation
            yFinal = trajSparse[-1]
            fFinal = yFinal[:nx * nPol].reshape((nx, nPol))
            lFinal = yFinal[nx * nPol:]
            rhoF = np.sum(fFinal, axis=1) * dp
            pFNumerator = np.sum(fFinal * pGrid[np.newaxis, :], axis=1) * dp
            pFLocal = np.divide(pFNumerator, rhoF, out=np.zeros_like(rhoF), \
                                where=rhoF > 0.0)

            #Suppress unused-variable warnings while retaining the JAX post-processing
            _ = lFinal, pFLocal

            #Output times and parameter tag
            tOut = timeStart + recordIndices * dt
            tag = 'lam_{:.3g}_xi_{:.3g}'.format(lambda_, interactionLength)

            #Save state and time arrays in the same orientation and text format
            np.savetxt(
                os.path.join(outputFolder, \
                             'solution_Y_numba_euler_{}.txt'.format(tag)),
                trajSparse.T,
                delimiter=','
            )
            np.savetxt(
                os.path.join(outputFolder, \
                             'solution_T_numba_euler_{}.txt'.format(tag)),
                tOut,
                delimiter=','
            )

            #Report simulation status
            if errFlag:
                print('Warning: numerical instability detected ' \
                      '(NaN or Inf encountered during simulation)')
            else:
                print('Simulation complete without numerical errors.')
            print('Runtime:', runTime / 60.0, 'minutes')


if __name__ == '__main__':
    runSimulations()
