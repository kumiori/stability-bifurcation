from dolfin import Constant, Expression
from ufl import grad, dot, inner, sym, sqrt, conditional
from ufl import as_tensor, as_vector, Identity, variable, diff, tr, inner, dev
import ufl


class DamageElasticityModel(object):
    def __init__(
        self,
        state,
        E0,
        nu,
        ell,
        sigma_D0,
        k_ell=Constant(1.0e-8),
        user_functional=None,
    ):

        self.u = state[0]
        self.alpha = state[1]
        self.E0 = E0
        self.nu = nu
        self.ell = ell
        self.sigma_D0 = sigma_D0
        self.k_ell = k_ell
        self.lmbda_0 = self.lmbda3D(0)
        self.mu_0 = self.mu3D(0)
        self.dim = state[0].function_space().ufl_element().value_size()
        self.user_functional = user_functional

    def lmbda3D(self, alpha):
        return self.E0*self.a(alpha) * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))

    def mu3D(self, alpha):
        return self.E0*self.a(alpha) / (2.0 * (1.0 + self.nu))

    def eps(self, u):
        """Strain tensor as a function of the displacement"""
        return sym(grad(u))

    def w(self, alpha):
        """Dissipated energy function as a function of the damage """
        return alpha * self.sigma_D0 ** 2 / self.E0

        ## Sigma_D0 = sigma_c Material's critical stress
        ## for AT1 model sigma_c = sqrt(3GcE/8l) --> sigma_c **2/E = 3Gc/8l

    def a(self, alpha):
        """Stiffness modulation as a function of the damage """
        return (1 - alpha) ** 2 + self.k_ell

    def elastic_energy_density(self, eps, alpha):
        lmbda = self.lmbda3D(alpha)
        mu = self.mu3D(alpha)
        return 1.0 / 2.0 * lmbda * tr(eps) ** 2 + mu * inner(eps, eps)

    def damage_dissipation_density(self, alpha):
        w_1 = self.w(1)
        return self.w(alpha) + w_1 * self.ell ** 2 * dot(grad(alpha), grad(alpha))

    def total_energy_density(self, u, alpha):
        # + self.user_energy()
        energy = self.elastic_energy_density(
            self.eps(u), alpha) + self.damage_dissipation_density(alpha)
        if self.user_functional:
           energy += self.user_functional
        return energy

    def user_energy(self, **args):
        return self.user_functional

    def stress(self, eps, alpha):
        eps_ = variable(eps)
        sigma = diff(self.elastic_energy_density(eps_, alpha), eps_)
        return sigma

    def sigma(self, u):
        n = u.geometric_dimension()
        lmbda = self.lmbda3D(0)
        mu = self.mu3D(0)
        return lmbda * tr(self.eps(u)) * Identity(n) + 2*mu * self.eps(u)

    def rP(self, u, alpha, v, beta):
        w_1 = self.w(1)
        a = self.a
        sigma = self.sigma
        eps = self.eps
        return inner(sqrt(a(alpha))*sigma(v) + diff(a(alpha), alpha)/sqrt(a(alpha))*sigma(u)*beta,
                    sqrt(a(alpha))*eps(v) + diff(a(alpha), alpha)/sqrt(a(alpha))*eps(u)*beta) + \
                    2*w_1*self.ell ** 2 * dot(grad(beta), grad(beta))

    def rN(self, u, alpha, beta):
        a = self.a
        w = self.w
        sigma = self.sigma
        eps = self.eps
        da = diff(a(alpha), alpha)
        dda = diff(diff(a(alpha), alpha), alpha)
        ddw = diff(diff(w(alpha), alpha), alpha)

        return -(1./2.*(dda - da**2./a(alpha))*inner(sigma(u), eps(u)) +1./2.*ddw)*beta**2.

class DamagePrestrainedElasticityModel(DamageElasticityModel):
    def __init__(
        self,
        state,
        E0,
        nu,
        ell,
        sigma_D0,
        eps0t=Expression([['t', '0'],['0', 't']], t=0., degree=0),
        k_ell=Constant(1.0e-8),
        user_functional=None,
    ):

        self.u = state[0]
        self.alpha = state[1]
        self.E0 = E0
        self.nu = nu
        self.ell = ell
        self.sigma_D0 = sigma_D0
        self.k_ell = k_ell
        self.lmbda_0 = self.lmbda3D(0)
        self.mu_0 = self.mu3D(0)
        self.eps0t = eps0t
        self.dim = state[0].function_space().ufl_element().value_size()
        self.user_functional = user_functional

    def eps0(self):
        # n = self.dim
        return self.eps0t

    def sigma0(self):
        n = self.dim
        lmbda = self.lmbda3D(0)
        mu = self.mu3D(0)
        return lmbda * tr(self.eps0()) * Identity(n) + 2*mu * self.eps0()

    def elastic_energy_density(self, eps, alpha):
        lmbda = self.lmbda3D(alpha)
        mu = self.mu3D(alpha)
        return 1.0 / 2.0 * lmbda * tr(eps-self.eps0()) ** 2 + mu * inner(eps-self.eps0(), eps-self.eps0())

    def total_energy_density(self, u, alpha):
        energy = self.elastic_energy_density(self.eps(u), alpha) + self.damage_dissipation_density(alpha)
        if self.user_functional:
           energy += self.user_functional
        return energy

    def rP(self, u, alpha, v, beta):
        w_1 = self.w(1)
        a = self.a
        sigma = self.sigma
        eps = self.eps
        eps0 = self.eps0
        sigma0 = self.sigma0
        return inner(sqrt(a(alpha))*sigma(v) + diff(a(alpha), alpha)/sqrt(a(alpha))*(sigma(u) - sigma0())*beta,
                     sqrt(a(alpha))*eps(v) + diff(a(alpha), alpha)/sqrt(a(alpha))*(eps(u) - eps0())*beta) + \
                    2*w_1*self.ell ** 2 * dot(grad(beta), grad(beta))

    def rN(self, u, alpha, beta):
        a = self.a
        w = self.w
        sigma = self.sigma
        eps = self.eps
        da = diff(a(alpha), alpha)
        dda = diff(diff(a(alpha), alpha), alpha)
        ddw = diff(diff(w(alpha), alpha), alpha)
        eps0 = self.eps0
        sigma0 = self.sigma0
        return -(1./2.*(dda - da**2./a(alpha))*inner(sigma(u)-sigma0(), eps(u)-eps0()) + 1./2.*ddw)*beta**2.

class ElasticityModel(object):
    def __init__(
        self,
        state,
        E0,
        nu,
        user_functional=None,
    ):

        self.u = state[0]
        self.E0 = E0
        self.nu = nu

    def eps(self, u):
        """Strain tensor as a functino of the displacement"""
        return sym(grad(u))

    def elastic_energy_density(self, eps):
        lmbda = self.E0* self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        mu = self.E0 / (2.0 * (1.0 + self.nu))
        return 1.0 / 2.0 * lmbda * tr(eps) ** 2 + mu * inner(eps, eps)

    def user_energy(self, **args):
        return self.user_functional

    def stress(self, eps, alpha):
        eps_ = variable(eps)
        sigma = diff(self.elastic_energy_density(eps), eps_)
        return sigma
