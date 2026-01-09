import sys
import numpy as np
from scipy.signal import fftconvolve
import time

############################## Helper functions ################################

# Compute CFL condition (with additional safety factor)
def cfl_dt(h, d_f, safety=0.01):
    lims = []
    if d_f > 0: lims.append(h*h/(2.0*d_f))
    if not lims:
        return 1.0
    return safety * min(lims)

# Compute laplacian
def laplacian_neumann(Z, h):
    lap = np.zeros_like(Z)
    lap[1:-1] = (Z[:-2] + Z[2:] - 2.0*Z[1:-1]) / (h*h)
    lap[0]    = 2 * (Z[1]  - Z[0]) / (h*h)
    lap[-1]   = 2 * (Z[-2] - Z[-1]) / (h*h)
    return lap

# Build exponential interaction kernel
def exp_kernel(L, h, interaction_length):
    three_sigma = 3 * interaction_length
    box_cap = max(0.0, L/2.0 - h)
    radius = min(three_sigma, box_cap)

    npts = max(1, int(np.floor(radius / h)))
    x = np.arange(-npts, npts + 1) * h

    ker = np.exp(-0.5 * (np.abs(x) / interaction_length)**2)
    # Normalisation so that sum * h ≈ ξ
    ker *= interaction_length / (np.sum(ker) * h)
    return ker

# Compute non-local convolution using FFT
def nonlocal_convolution(Z, kernel, h):
    return h * fftconvolve(Z, kernel, mode='same')

# Leader cell shifting function (advection)
def shift_vacuum(arr, n):
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

#################################### Model #####################################

class ExplicitNeuralCrest1D:
    """
    Forward-Euler integrator.
    Leaders: pure translation by integer-cell rolling.
      - boundary_mode='vacuum' (no wrap).
      - boundary_mode='wrap' uses periodic wrap via np.roll.
    Followers: diffusion + nonlocal-taxis with volume filling.
    """
    def __init__(
        self, l0, f0, v_l, d_f, mu_fl, mu_ff, K,
        xi_fl, xi_ff, L, mesh_points,
        time_span, time_evaluations,
        dt=None, safety=0.01, clip_nonneg=True,
        boundary_mode='vacuum'
    ):
        self.l0, self.f0 = l0, f0
        self.v_l = float(v_l)
        self.d_f = float(d_f)
   
        self.mu_fl, self.mu_ff = float(mu_fl), float(mu_ff)
        self.K = float(K)
        self.xi_fl, self.xi_ff = float(xi_fl), float(xi_ff)

        self.L = float(L)
        self.N = int(mesh_points)
        self.x = np.linspace(0.0, self.L, self.N)
        self.h = self.x[1] - self.x[0] if self.N > 1 else self.L

        self.t0, self.tf = time_span
        self.t_eval = np.asarray(time_evaluations, dtype=float)

        self.kernel_fl = exp_kernel(self.L, self.h, self.xi_fl)
        self.kernel_ff = exp_kernel(self.L, self.h, self.xi_ff)

        self.l, self.f = self.default_initial_condition()

        self.clip_nonneg = clip_nonneg
        self.boundary_mode = boundary_mode 

        # Displacement accumulator in grid-cell units
        self._shift_accum = 0.0

        # Timestep
        self.dt = cfl_dt(self.h, self.d_f, safety) \
                  if dt is None else float(dt)

    def default_initial_condition(self):
        x = self.x
        l = np.zeros_like(x)
        f = np.zeros_like(x)
        x_lower = 65.0
        x_upper = 870.0
        delta = 5.0
        f[:] = 0.5 * self.f0 * (np.tanh((x - x_lower)/delta) - \
               np.tanh((x - x_upper)/delta))
        x0 = 895.0
        sigma = 12.0
        base = np.exp(-0.5*((x - x0)/sigma)**2)
        l[:] = (base / base.max()) * f.max()
        return l, f

    def _shift_leaders(self):
        if self.v_l == 0.0:
            return
        # Accumulate displacement in cell units
        self._shift_accum += self.v_l * self.dt / self.h
        nshift = int(np.trunc(self._shift_accum))  
        if nshift != 0:
            if self.boundary_mode == 'wrap':
                self.l = np.roll(self.l, nshift)
            else:
                self.l = shift_vacuum(self.l, nshift)
            self._shift_accum -= nshift  

    def step(self):
        # Leader shift
        self._shift_leaders()

        # Follower PDE update
        l, f = self.l, self.f

        C = (self.mu_ff * nonlocal_convolution(f, self.kernel_ff, self.h)
            + self.mu_fl * nonlocal_convolution(l, self.kernel_fl, self.h))

        lap_f = laplacian_neumann(f, self.h)

        # Finite volume taxis
        P = self.K - l - f                         
        dCdx_face = (C[1:] - C[:-1]) / self.h       
        P_face = 0.5 * (P[1:] + P[:-1])            
        u_face = P_face * dCdx_face                

        f_up = np.where(u_face >= 0.0, f[:-1], f[1:])  

        F_face = np.zeros(self.N + 1)               
        F_face[1:-1] = f_up * u_face

        dflux_dx = (F_face[1:] - F_face[:-1]) / self.h    

        df_dt = self.d_f * lap_f - dflux_dx
        
        self.f = f + self.dt * df_dt     

        if self.clip_nonneg:
            np.maximum(self.f, 0.0, out=self.f)

    def simulate_with_progress(self, y_file, t_file, checks=50):
        start = time.time()
        t_grid = self.t_eval
        out_Y = np.empty((2*self.N, t_grid.size), dtype=float)

        t_curr = self.t0
        out_Y[:self.N, 0] = self.l
        out_Y[self.N:, 0] = self.f

        total = t_grid[-1] - t_grid[0]
        next_check = 1

        for k in range(1, t_grid.size):
            t_target = t_grid[k]
            while t_curr + self.dt < t_target - 1e-15:
                self.step()
                t_curr += self.dt
                print(t_curr)

            dt_last = t_target - t_curr
            if dt_last > 1e-15:
                dt_save = self.dt
                self.dt = dt_last
                self.step()
                self.dt = dt_save
                t_curr = t_target

            out_Y[:self.N, k] = self.l
            out_Y[self.N:, k] = self.f

            frac = (t_grid[k] - t_grid[0]) / total if total > 0 else 1.0
            if frac >= next_check / checks:
                next_check += 1

        np.savetxt(y_file, out_Y, delimiter=',')
        np.savetxt(t_file, t_grid, delimiter=',')

############################## CLI wrapper #####################################

if __name__ == "__main__":
    l0 = 0.95
    f0 = 0.95
    v_l = 1.0
    d_f = 10.0
    mu_fl = int(sys.argv[1])
    mu_ff = int(sys.argv[2])
    xi_fl = int(sys.argv[3])
    xi_ff = int(sys.argv[4])
    K = 1.0
    L = 2000.0
    mesh_points = 20000
    time_span = (0.0, 12.0 * 60.0)
    time_evaluations = np.linspace(time_span[0], time_span[1], 24)

    sim = ExplicitNeuralCrest1D(
        l0, f0, v_l, d_f, mu_fl, mu_ff, K,
        xi_fl, xi_ff, L, mesh_points,
        time_span, time_evaluations,
        dt=None,          
        safety=0.01,
        clip_nonneg=True,
        boundary_mode='vacuum'  
    )

    filename_y = f"Y_mu_fl{mu_fl}_mu_ff{mu_ff}_xi_fl{xi_fl}_xi_ff{xi_ff}.txt"
    filename_t = f"T_mu_fl{mu_fl}_mu_ff{mu_ff}_xi_fl{xi_fl}_xi_ff{xi_ff}.txt"

    sim.simulate_with_progress(filename_y, filename_t)

