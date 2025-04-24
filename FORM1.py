import numpy as np
from scipy.stats import norm
from sympy import symbols, solve, Eq, exp, diff, lambdify
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde


class ReliabilityAnalysis:
    def __init__(self):
        pass



    def create_R_S_functions(self, n, model):
        # --- Define statistical properties ---
        # Resistance inputs (lognormal)
        mu_fy= model['fy'] * 1.2
        sigma_fy= 0.05 * mu_fy
            # Convert to log-space
        sigma_ln_fy = np.sqrt(np.log(1 + (sigma_fy**2 / mu_fy**2)))
        mu_ln_fy = np.log(mu_fy) - 0.5 * sigma_ln_fy**2



        
        # Imposed load inputs (gumble)
        mu_q= 0.4 * model['q']
        sigma_q= 0.49 * mu_q

            # Convert to Gumbel parameters
        beta = sigma_q * np.sqrt(6) / np.pi
        gamma = 0.5772  # Euler-Mascheroni constant
        loc = mu_q - beta * gamma

        # Permanent load inputs (normal)
        mu_gsw= 0.985 * model['gsw']
        sigma_gsw= 0.045 * mu_gsw

        mu_gnsl= 1 * model['gnsl']
        sigma_gnsl= 0.1 * mu_gsw             

        # Uncertainties (lognormal)
        mu_tq, mu_tE= 1 , 1
        sigma_tq, sigma_tE= 0.1 , 0.1 

        mu_tr= 1.15
        sigma_tr= 0.05 * 1.15 

        sigma_ln_tq = np.sqrt(np.log(1 + (sigma_tq**2 / mu_tq**2)))
        mu_ln_tq = np.log(mu_tq) - 0.5 * sigma_ln_tq**2

        sigma_ln_tE = np.sqrt(np.log(1 + (sigma_tE**2 / mu_tE**2)))
        mu_ln_tE = np.log(mu_tE) - 0.5 * sigma_ln_tE**2

        sigma_ln_tr = np.sqrt(np.log(1 + (sigma_tr**2 / mu_tr**2)))
        mu_ln_tr = np.log(mu_tr) - 0.5 * sigma_ln_tr**2



        # --- Generate samples ---
        fy = np.random.lognormal(mu_ln_fy, sigma_ln_fy, n)
        q = np.random.gumbel(loc, beta, n)
        gsw = np.random.normal(mu_gsw, sigma_gsw, n)
        gnsl = np.random.normal(mu_gnsl, sigma_gnsl, n)
        teta_E= np.random.lognormal(mu_ln_tE, sigma_ln_tE, n)
        teta_q= np.random.lognormal(mu_ln_tq, sigma_ln_tq, n)
        teta_R= np.random.lognormal(mu_ln_tr, sigma_ln_tr, n)

        # --- Compute R and S ---
        R = fy * teta_R * float(model['p'])  # e.g. bending resistance
        S = teta_E * (gsw + gnsl + teta_q*q) * model['D'] * model['L']**2 / 8  # e.g. simply supported uniform load

        # --- Sort values for empirical CDF ---
        R_sorted = np.sort(R)
        S_sorted = np.sort(S)
        p_vals = np.linspace(0, 1, n)

        # --- Create interpolated CDF functions ---
        F_R = interp1d(R_sorted, p_vals, bounds_error=False, fill_value=(0.0, 1.0))
        F_S = interp1d(S_sorted, p_vals, bounds_error=False, fill_value=(0.0, 1.0))



        # Solve inverse of CDFs
            # Empirical CDF values (from 0 to 1)
        F_vals = np.linspace(0, 1, len(R_sorted), endpoint=False)
        F_vals_S = np.linspace(0, 1, len(S_sorted), endpoint=False)

        self.inv_cdf_R = interp1d(F_vals, R_sorted, bounds_error=False, fill_value=(R_sorted[0], R_sorted[-1]))
        self.inv_cdf_S = interp1d(F_vals_S, S_sorted, bounds_error=False, fill_value=(S_sorted[0], S_sorted[-1]))

        # ---- Derive PDFs from empirical CDFs ----
        kde_R = gaussian_kde(R)
        f_R = lambda x: kde_R(x)

        kde_S = gaussian_kde(S)
        f_S = lambda x: kde_S(x)

        return F_R, F_S, f_R, f_S, R_sorted, S_sorted




    def HLRF_Algorithm(self, N, model):

        # CDFs of R and S

        F_R, F_S, f_R, f_S, R_sorted, S_sorted = self.create_R_S_functions(100000, model)  

        # Safe inverse CDF transformation (norm.ppf)
        def safe_ppf(p, eps=1e-10):
            return norm.ppf(np.clip(p, eps, 1 - eps))

        # Gradient of limit state function: g(R, S) = R - S
        grad_g = np.array([1, -1])  # constant for linear g(x)



        beta_list = []
        R0= model['fy'] * float(model['p'])
        S0= (model['gsw'] + model['gnsl'] + model['q']) * model['D'] * model['L']**2 / 8

        for i in range(N):
            # Step 1: Convert x to u-space
            U = np.array([
                safe_ppf(F_R(R0)),
                safe_ppf(F_S(S0))
            ])

            # Step 2: Get PDFs
            f_X = np.array([f_R(R0).item(), f_S(S0).item()])

            # Step 3: Standard normal PDFs
            phi = np.array([norm.pdf(U[0]), norm.pdf(U[1])])

            # Step 4: Jacobian J_XU = diag(φ(u) / f_X(x))
            J_XU = np.array([
                [phi[0] / f_X[0], 0],
                [0, phi[1] / f_X[1]]
            ])

            # Step 5: Gradient in u-space
            grad_h = np.dot(J_XU.T, grad_g)

            # Step 6: Direction cosines
            norm_grad_h = np.linalg.norm(grad_h)
            alpha = -grad_h / norm_grad_h

            # Step 7: Reliability index
            beta = np.dot(alpha, U)

            # Step 8: Evaluate g(X)
            h_u = R0 - S0

            # Step 9: Update u
            next_u = alpha * (beta + h_u / norm_grad_h)

            # Step 10: Back-transform to x-space (inverse CDF)
            phi_u = norm.cdf(next_u)


            # Step 11: Update physical variables
            next_x = np.array([
                self.inv_cdf_R(phi_u[0]),
                self.inv_cdf_S(phi_u[1])
            ])

            beta_list.append(beta)
            X0 = next_x

        pf=norm.cdf(-beta)
        return pf


