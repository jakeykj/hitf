"""
Implementation of Hidden Interaction Tensor Factorization (HITF) model based on
pytorch.

"""

import time
import logging
import numpy as np
import torch
import torch.nn.functional as f

from functools import partial

eps = 1e-10

def P_plus(X, proj_eps):
    return torch.clamp(X, min=proj_eps)

class HITF:
    def __init__(self, R, proj_eps=1e-5, use_cuda=False, logger=None):
        self.U_init = None
        self.U = None
        self.lmbda = None
        self.R = R
        self.proj_eps = proj_eps
        self.use_cuda = use_cuda
        self.logger = logger or logging.getLogger(__name__)

    def _eval_nll(self, U, M, Dprime, verbose=False):
        Mhat = U[0] @ torch.diag(U[1].sum(dim=0)) @ U[2].t()
        Dhat = U[0] @ torch.diag(U[2].sum(dim=0)) @ U[1].t()
        nll_M = Mhat - M * torch.log(torch.clamp(Mhat, min=1e-15))
        nll_D = Dhat.clone()
        nll_D[nll_D < 35] = torch.log(torch.clamp(torch.exp(nll_D[nll_D < 35]) - 1, min=1e-15))
        nll_D = Dhat - Dprime * nll_D
        if verbose:
            return nll_M.sum(), nll_D.sum(), nll_M.sum() + nll_D.sum()
        return nll_M.sum() + nll_D.sum()

    def calculate_Phi(self, U, M):
        Mhat = U[0] @ torch.diag(U[1].sum(dim=0)) @ U[2].t()
        return M.t() / torch.clamp(Mhat.t(), min=eps)

    def calculate_Psi(self, U, Dprime):
        Dhat = U[0] @ torch.diag(U[2].sum(dim=0)) @ U[1].t()
        return (Dprime / torch.clamp(1 - torch.exp(-Dhat), min=eps)).t()

    def normalize_factors(self):
        for n in range(len(self.U)):
            self.lmbda *= self.U[n].norm(1, dim=0)
            self.U[n] = f.normalize(self.U[n], 1, dim=0)

    def redistribute_factors(self, n):
        self.normalize_factors()
        self.U[n] *= self.lmbda
        self.lmbda = torch.ones(self.R).cuda() if self.use_cuda else torch.ones(self.R)

    def _projected_line_search_armijo(self, X, grad, Sk, eval_func, proj_func,
                                      desc_step, suff_desc, max_steps=100):
        for t in range(max_steps):
            if eval_func(proj_func(X + (desc_step ** t) * Sk)) - eval_func(X) <= suff_desc * ((proj_func(X + (desc_step ** t) * Sk)-X) * grad).sum():
                return proj_func(X + (desc_step ** t) * Sk), t + 1
        return X, -1  # No updating if maximum steps reached.

    def _solve_subproblem(self, n, Xinit, M, Dprime, max_iter, grad_func, desc_step, suff_desc, outer):
        X = Xinit.clone()
        # U = []

        eval_f = lambda X: self._eval_nll([self.U[k] if k != n else X for k in range(len(self.U))], M=M, Dprime=Dprime)
        proj_func = partial(P_plus, proj_eps=self.proj_eps)

        nll_init = eval_f(X)
        nlls = []
        steps = []
        tic = time.time()
        for iter_ in range(max_iter):
            grad = grad_func([self.U[k] if k != n else X for k in range(len(self.U))])
            X, t = self._projected_line_search_armijo(X, grad, -grad, eval_f, proj_func, desc_step, suff_desc)
            nll = eval_f(X)
            self.logger.debug(f'  Outer: {outer}, U{n+1}, iter:{iter_+1}, t={t}, nll: {nll:f}')
            nlls.append(nll)
            steps.append(t)
            if iter_ > 0 and abs((nlls[-2] - nlls[-1]) / nlls[-2]) < 1e-5:
                break
        toc = time.time()
        iter_info = {
            'inner_iter_time': toc - tic,
            'nll_init': nll_init,
            'inner_nlls': nlls,
            'step_size': steps
        }
        self.logger.info(f'  Outer: {outer}, U{n+1} updated with {len(nlls)} iterations, final nll: {nlls[-1]:f}.')
        return X, iter_info


    def decompose(self, M, Dprime, max_outer_iters=200,
                  max_inner_iters=100, desc_step=0.5, suff_desc=1e-4,
                  dump_file=None, random_state=None):
        if isinstance(M, np.ndarray):
            M = torch.from_numpy(M.astype(np.float32))
        if isinstance(Dprime, np.ndarray):
            Dprime = torch.from_numpy(Dprime.astype(np.float32))
        # set random state if specified
        if random_state is None:
            st0 = np.random.get_state()
        else:
            self.logger.warning()

        # for debug
        # seed = 75
        # np.random.seed(seed)
        # dims = (M.shape[0], Dprime.shape[1], M.shape[1])
        # self.U = [torch.from_numpy(np.random.rand(dim, self.R).astype(np.float32)) for dim in dims]
        # self.lmbda = torch.ones(self.R)
        # self.normalize_factors()
        # self.lmbda = torch.ones(self.R)  # to avoid too large initial value


        # initialize
        dims = (M.shape[0], Dprime.shape[1], M.shape[1])
        U_init = [torch.rand(dim, self.R) for dim in dims]
        self.U = [k.clone() for k in U_init]
        self.lmbda = torch.ones(self.R)
        self.normalize_factors()
        self.lmbda = torch.ones(self.R)  # to avoid too large initial value
        if self.use_cuda:
            # self.U_init = [k.cuda() for k in self.U_init]
            self.U = [k.cuda() for k in self.U]
            self.lmbda = self.lmbda.cuda()
            M = M.cuda()
            Dprime = Dprime.cuda()

        Phi = lambda U: self.calculate_Phi(U, M)
        Psi = lambda U: self.calculate_Psi(U, Dprime)
        # define gradients

        if self.use_cuda:
            ones = lambda shape: torch.cuda.FloatTensor(shape).fill_(1)
            eyes = torch.eye(self.R).cuda()
        else:
            ones = lambda shape: torch.ones(shape)
            eyes = torch.eye(self.R)
        grad_funcs = [
            lambda U: 2 * ones(U[0].shape) - Phi(U).t() @ U[2] - Psi(U).t() @ U[1],  # gradient w.r.t. U1
            lambda U: 2 * ones(U[1].shape) - ones(U[1].shape) @ ((U[0].t() @ Phi(U).t() @ U[2]) * eyes) - Psi(U) @ U[0],  # gradient w.r.t. U2
            lambda U: 2 * ones(U[2].shape) - Phi(U) @ U[0] - ones(U[2].shape) @ ((U[0].t() @ Psi(U).t() @ U[1]) * eyes)  ## gradient w.r.t. U3
        ]

        self.iters_info = []
        nll = self._eval_nll(self.U, M, Dprime, verbose=True)
        tic = time.time()
        for iter_ in range(max_outer_iters):
            self.logger.info(f'Start the {iter_+1}-th iteration.')

            iter_tic = time.time()
            iter_infos = [None] * 3
            for n in range(3):
                self.redistribute_factors(n)
                self.U[n], iter_infos[n] = self._solve_subproblem(n, self.U[n], M, Dprime, max_inner_iters, grad_funcs[n], desc_step, suff_desc, iter_)
            time_elapsed = time.time() - iter_tic

            nll_old = nll
            nll = self._eval_nll(self.U, M, Dprime, verbose=True)
            nll_delta = abs((nll_old[-1] - nll[-1]) / nll_old[-1])
            fit_M = torch.norm(M - self.U[0] @ torch.diag(self.U[1].sum(dim=0)) @ self.U[2].t()) ** 2
            Dhat = self.U[0] @ torch.diag(self.U[2].sum(dim=0)) @ self.U[1].t()
            Dhat[Dhat > 0] = 1
            fit_D = torch.norm(Dprime - Dhat) ** 2
            iter_info = {
                'inner_iter_infos': iter_infos,
                'nll': nll,
                'iter_time': time_elapsed
            }

            self.iters_info.append(iter_info)
            self.logger.info(f'Iter {iter_}: {time_elapsed:.1f}s, negtative-ll {nll[-1]:.3f}, nll delta {nll_delta}, M fit {fit_M:.3f}, D fit {fit_D}')

            self.redistribute_factors(0)

            if dump_file:
                np.savez(dump_file, iter=iter_,
                         U1=self.U[0].cpu().numpy(),
                         U2=self.U[1].cpu().numpy(),
                         U3=self.U[2].cpu().numpy())
            if iter_ > 0 and abs(nll_delta) <= 1e-4:
                break
        self.logger.info(f'Decomposition is done, time: {time.time()-tic:.1f}s')

    def project(self, M, Dprime, max_outer_iters=200,
                  max_inner_iters=100, desc_step=0.5, suff_desc=1e-4,
                  dump_file=None, random_state=None):
        if isinstance(M, np.ndarray):
            M = torch.from_numpy(M.astype(np.float32))
        if isinstance(Dprime, np.ndarray):
            Dprime = torch.from_numpy(Dprime.astype(np.float32))
        dims = (M.shape[0], Dprime.shape[1], M.shape[1])

        proj = torch.rand(dims[0], self.R)
        if self.use_cuda:
            proj = proj.cuda()
            M = M.cuda()
            Dprime = Dprime.cuda()
        self.logger.info('Projecting testing data with dims ({}, {}, {}).'.format(*dims))

        Phi = lambda U: self.calculate_Phi(U, M)
        Psi = lambda U: self.calculate_Psi(U, Dprime)
        ones = lambda shape: torch.ones(shape).cuda() if self.use_cuda else torch.ones(shape)
        grad_func = lambda U: 2 * ones(U[0].shape) - Phi(U).t() @ U[2] - Psi(U).t() @ U[1]  # gradient w.r.t. U1

        for iter_ in range(max_outer_iters):
            proj_old = proj.clone()
            proj, iter_info = self._solve_subproblem(0, proj_old, M, Dprime, max_inner_iters, grad_func, desc_step, suff_desc, iter_)
            Xchange = torch.norm(proj - proj_old)**2
            if Xchange < 1e-4:
                break
        self.logger.info('Projection done with {} iterations.'.format(iter_+1))
        return proj






