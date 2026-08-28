"""Gaussian process modelling of decoupled objectives, on BoTorch."""
# TODO: clean up these comments in this file.
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from botorch.acquisition import PosteriorMean
from botorch.acquisition.objective import ScalarizedPosteriorTransform
from botorch.fit import fit_gpytorch_mll
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ZeroMean
from gpytorch.mlls import ExactMarginalLogLikelihood, SumMarginalLogLikelihood
from gpytorch.priors import LogNormalPrior

from pypolar.optimization.objectives import DecoupledObjectives, Objective

DTYPE           = torch.float64
NOISE_STD       = 0.05      # observation noise, as a fraction of each objective's spread
LENGTH_SCALE    = 0.2       # 20% of the unit action box
SIGNAL_VAR      = 1.0       # prior variance of unit-variance values
PRIOR_SIGMA     = 1.0       # LogNormal noise prior width, matching BoTorch's own
RAW_SAMPLES     = 512       # stage-1 Sobol samples per optimization
NUM_RESTARTS    = 8         # stage-2 L-BFGS-B starting points


@dataclass(frozen=True)
class NoiseModel:
    """How a GP treats its observation noise, as a fraction of the objective's
    own standard deviation.

    Three representations, one per constructor:

    - **`pinned(std)`** fixes the noise through `train_Yvar`, which selects
      `FixedNoiseGaussianLikelihood`, so a fit only ever moves the kernel.
    - **`fitted()`** leaves it free for the marginal likelihood. On a small
      design this can collapse onto interpolation: the GP threads every
      measurement, calls the residual zero, and puts its argmax on whichever
      point drew the luckiest noise.
    - **`prior(median, sigma)`** fits it under a LogNormal centered on `median`,
      pulling a collapsing fit back without forbidding any value outright. This
      is the conventional choice, and what BoTorch's own default likelihood does.

    A pinned noise is never fitted, so it cannot also carry a prior; the two
    constructors are mutually exclusive and the combination raises.

    Attributes:
        std: the pinned level, or None when the noise is fitted.
        median: center of the prior on the noise *standard deviation*, or None
            to fit it unpriored.
        sigma: prior width in log space, on the variance.
    """

    std    : float | None = None
    median : float | None = None
    sigma  : float        = PRIOR_SIGMA

    def __post_init__(self):
        if self.std is not None and self.median is not None:
            raise ValueError('a pinned noise is never fitted, so it cannot carry '
                             'a prior; use NoiseModel.pinned or .prior, not both')
        if self.std is not None and self.std < 0:
            raise ValueError('noise std must be non-negative')
        if self.median is not None and self.median <= 0:
            raise ValueError('a LogNormal prior has positive support, so its '
                             'median must be positive')

    @classmethod
    def pinned(cls, std: float) -> 'NoiseModel':
        return cls(std=std)

    @classmethod
    def fitted(cls) -> 'NoiseModel':
        return cls()

    @classmethod
    def prior(cls, median: float, sigma: float = PRIOR_SIGMA) -> 'NoiseModel':
        return cls(median=median, sigma=sigma)

    @classmethod
    def coerce(cls, noise) -> 'NoiseModel':
        """A `NoiseModel` unchanged, a number as `pinned`, None as `fitted`."""
        if isinstance(noise, cls):
            return noise
        return cls.fitted() if noise is None else cls.pinned(noise)

    @property
    def is_fitted(self) -> bool:
        return self.std is None

    def train_yvar(self, train_Y: torch.Tensor, spread: torch.Tensor):
        """The per-point variance to pin, or None when the noise is fitted.

        `Standardize` divides `train_Yvar` by the same variance it divides
        `train_Y` by, so pre-scaling by the spread is what leaves `std` a
        fraction of the objective's own standard deviation rather than a raw
        magnitude.
        """
        if self.std is None:
            return None
        return torch.full_like(train_Y, (self.std * spread) ** 2)

    def likelihood(self):
        """The likelihood to attach, or None to let `train_Yvar` pick a fixed one.

        Passed explicitly whenever the noise is fitted, because BoTorch's default
        likelihood carries a LogNormal noise prior of its own, centered low
        enough to decide the answer on a small design.
        """
        if self.std is not None:
            return None

        # post-Standardize the noise variance is the squared fraction, so a
        # median noise standard deviation of m is a median variance of m^2
        prior = None if self.median is None else LogNormalPrior(
            loc=2.0 * float(np.log(self.median)), scale=self.sigma
        )
        return GaussianLikelihood(noise_prior=prior).to(DTYPE)


def build_botorch_gp(
    actions          : np.ndarray,
    values           : np.ndarray,
    noise            : 'NoiseModel | float | None',
    signal_var       : float,
    length_scale     : float,
    min_length_scale : float | None = None,
    kernel_wrap_fn   : Callable = None
) -> SingleTaskGP:
    """The one `SingleTaskGP` configuration the package uses.

    Args:
        actions: (n, d) actions, in the normalized frame.
        values: (n,) values, already standardized and larger-is-better.
        noise: a `NoiseModel`, or the shorthands it coerces -- a number for a
            pinned fraction of the values' own spread, None to fit the noise.
        signal_var: starting ScaleKernel outputscale.
        length_scale: starting ARD lengthscale, scalar or one per dimension.
        min_length_scale: lower bound on every lengthscale, in the normalized
            frame. None leaves them bounded only away from zero.

    Returns:
        the `SingleTaskGP`, unfitted.
    """
    train_X = torch.as_tensor(actions, dtype=DTYPE)
    train_Y = torch.as_tensor(values, dtype=DTYPE).reshape(-1, 1)
    noise   = NoiseModel.coerce(noise)

    if(kernel_wrap_fn):
        prior_covar = kernel_wrap_fn(ScaleKernel(RBFKernel(
            ard_num_dims           = train_X.shape[-1],
            lengthscale_constraint = None if min_length_scale is None
                                    else GreaterThan(min_length_scale)
        ))).to(DTYPE)
        if min_length_scale is not None:
            # a GreaterThan cannot represent its own edge, so start strictly inside.
            # elementwise, since length_scale may be one per action dimension
            length_scale = np.maximum(length_scale, 1.5 * min_length_scale)
        # TODO: There must be a better way to handle the "hidden" field of base_kernel's length scale
        prior_covar.base_kernel.base_kernel.lengthscale = torch.tensor(length_scale, dtype=DTYPE)
        prior_covar.base_kernel.outputscale = torch.tensor(signal_var, dtype=DTYPE)
    else:
        prior_covar = ScaleKernel(RBFKernel(
            ard_num_dims           = train_X.shape[-1],
            lengthscale_constraint = None if min_length_scale is None
                                    else GreaterThan(min_length_scale)
        )).to(DTYPE)

        if min_length_scale is not None:
            # a GreaterThan cannot represent its own edge, so start strictly inside.
            # elementwise, since length_scale may be one per action dimension
            length_scale = np.maximum(length_scale, 1.5 * min_length_scale)
        prior_covar.base_kernel.lengthscale = torch.tensor(length_scale, dtype=DTYPE)
        prior_covar.outputscale = torch.tensor(signal_var, dtype=DTYPE)

    # one point has no sample standard deviation, so fall back to 1 rather than
    # propagate a NaN into the likelihood
    spread = train_Y.std() if train_Y.shape[0] > 1 else torch.ones((), dtype=DTYPE)

    return SingleTaskGP(
        train_X             = train_X,
        train_Y             = train_Y,
        train_Yvar          = noise.train_yvar(train_Y, spread),
        likelihood          = noise.likelihood(),
        covar_module        = prior_covar,
        mean_module         = ZeroMean(),
        outcome_transform   = Standardize(m=1),
        input_transform     = None,
    )

@dataclass(frozen=True, eq=False)
class GPHyperparameters:
    """Every hyperparameter of one GP built by `build_botorch_gp`.

    Attributes:
        lengthscale: (d,) ARD lengthscales, in the normalized action frame.
            Scalar when the same value is meant for every dimension.
        signal_var: the ScaleKernel outputscale.
        noise_var: the observation noise, pinned or fitted.
        standardize_scale: the divisor Standardize applied to the values. Both
            variances are in post-Standardize units; multiplying them by
            `standardize_scale ** 2` puts them in the units the GP was handed.
            Defaults to 1, which is the scale of values already at unit spread.
    """

    lengthscale       : np.ndarray | float
    signal_var        : float
    noise_var         : float
    standardize_scale : float = 1.0


def gp_hyperparameters(model: SingleTaskGP) -> GPHyperparameters:
    """The hyperparameters of one GP built by `build_botorch_gp`.

    Args:
        model: a GP built by `build_botorch_gp`.

    Returns:
        the `GPHyperparameters`, read off the model's own tensors.
    """
    kernel = model.covar_module
    return GPHyperparameters(
        lengthscale       = kernel.base_kernel.lengthscale.detach().numpy().ravel(),
        signal_var        = kernel.outputscale.item(),
        # identical at every point by construction, so the mean is that value
        noise_var         = model.likelihood.noise.mean().item(),
        standardize_scale = model.outcome_transform.stdvs.item(),
    )


def _posterior_at(model, frame, action, normalized, chunk, transform=None):
    """Posterior mean and standard deviation at arbitrary actions.

    Args:
        model: the BoTorch model to read.
        frame: the `AffineTransform` normalizing raw actions.
        action: a single (d,) action or an (n, d) array of them.
        normalized: True if `action` is already in the [0, 1]^d frame.
        chunk: actions evaluated per posterior call.
        transform: a BoTorch `PosteriorTransform`, or None.

    Returns:
        (mean, std), each (n, m), in maximization space.
    """
    X = np.atleast_2d(np.asarray(action, dtype=float))
    if not normalized:
        X = frame(X)

    # fed as q=1 batch elements, so each posterior is 1x1 per objective and
    # gpytorch never forms the n x n test-test block
    X = torch.as_tensor(X, dtype=DTYPE).unsqueeze(1)
    with torch.no_grad():
        posteriors = [model.posterior(X[i:i + chunk], posterior_transform=transform)
                      for i in range(0, X.shape[0], chunk)]
        mu  = torch.cat([p.mean.squeeze(1)     for p in posteriors])
        var = torch.cat([p.variance.squeeze(1) for p in posteriors])

    return mu.numpy(), var.sqrt().numpy()


def _sample_paths(model, frame, action, num_paths, normalized, transform=None):
    """Joint posterior draws over arbitrary actions.

    Args:
        model: the BoTorch model to read.
        frame: the `AffineTransform` normalizing raw actions.
        action: a single (d,) action or an (n, d) array of them.
        num_paths: paths drawn.
        normalized: True if `action` is already in the [0, 1]^d frame.
        transform: a BoTorch `PosteriorTransform`, or None.

    Returns:
        (num_paths, n, m) draws, in maximization space.
    """
    X = np.atleast_2d(np.asarray(action, dtype=float))
    if not normalized:
        X = frame(X)

    # fed as one batch element rather than n of them, the opposite of
    # `_posterior_at`, so the test-test covariance is formed and sampled
    X = torch.as_tensor(X, dtype=DTYPE)
    with torch.no_grad():
        posterior = model.posterior(X, posterior_transform=transform)
        return posterior.rsample(torch.Size([num_paths])).numpy()


class BoTorchGP:
    """BoTorch GP in a convenient wrapper.
    """

    # nothing to read the posterior through: this model is already single-output
    transform = None

    def __init__(self, objective: Objective, noise, fit_hyperparameters=True,
                 length_scale=LENGTH_SCALE, signal_var=SIGNAL_VAR,
                 min_length_scale=None, kernel_wrap_fn=None):
        """
        Args:
            objective: the measurements to condition on.
            noise: a `NoiseModel`, or the shorthands it coerces -- a number for
                a pinned fraction of the objective's own standard deviation,
                None to fit the noise.
            fit_hyperparameters: fit the lengthscales and signal variance by
                marginal likelihood, starting from `length_scale`/`signal_var`.
            length_scale: starting ARD lengthscale, in the normalized frame.
            signal_var: starting ScaleKernel outputscale.
            min_length_scale: lower bound on every lengthscale, or None.
            kernel_wrap_fn: A function taking in a kernel and returning another.
                Used if the default RBF kernel needs to  be modified externally.
        """
        self.noise = NoiseModel.coerce(noise)
        if self.noise.is_fitted and not fit_hyperparameters:
            raise ValueError('a fitted noise is determined by the marginal '
                             'likelihood, so fit_hyperparameters must be True')
        self.fit_hyperparameters = fit_hyperparameters
        self.length_scale     = length_scale
        self.signal_var       = signal_var
        self.min_length_scale = min_length_scale
        self.kernel_wrap_fn = kernel_wrap_fn
        self.update_feedback(objective)
        
    def update_feedback(self, objective: Objective):
        """Rebuild GP against the current feedback.

        Args:
            objective: the measurements to condition on.

        Returns:
            The `SingleTaskGP`, in eval mode.
        """
        self.objective = objective
        self.model = build_botorch_gp(
            actions          = objective.normalized_x,
            values           = objective.standard_y,
            noise            = self.noise,
            signal_var       = self.signal_var,
            length_scale     = self.length_scale,
            min_length_scale = self.min_length_scale,
            kernel_wrap_fn   = self.kernel_wrap_fn
        ).eval()

        if self.fit_hyperparameters:
            # one exact GP, so one exact marginal likelihood
            fit_gpytorch_mll(
                ExactMarginalLogLikelihood(self.model.likelihood, self.model)
            )
            self.model.eval()

        return self.model

    def posterior_at(self, action, normalized=False, raw=False, chunk=2048):
        """Posterior mean and standard deviation at arbitrary actions.

        Args:
            action: a single (d,) action or an (n, d) array of them.
            normalized: True if `action` is already in the shared [0, 1]^d frame.
            raw: report in each objective's own units instead of maximization
                space. This undoes the sign too.
            chunk: actions evaluated per posterior call.

        Returns:
            (mean, std), each (n, m), in maximization space: larger-is-better and
            standardized, matching `objectives.feedback()`. With `raw=True`, in
            the units the measurements were taken in.
        """
        mu, std = _posterior_at(self.model, self.frame, action, normalized, chunk)
        return self.objective.to_raw(mu, std) if raw else (mu, std)

    def sample_paths(self, action, num_paths, normalized=False, raw=False):
        """Sample paths of the posterior over arbitrary actions.

        Drawn from the *joint* posterior over the whole set of actions, which is
        what makes each draw a function: sampling every action from its own
        marginal independently would discard the covariance between them and
        return white noise. That is also why this cannot chunk the way
        `posterior_at` does -- the n x n test-test block it avoids forming is
        the very object a joint draw needs.

        The draws come from torch's global generator, so seeding that is what
        makes them repeatable.

        Args:
            action: a single (d,) action or an (n, d) array of them.
            num_paths: paths drawn. Not botorch's `q`, which here is n: the
                actions are one q-batch, and this is how often it is sampled.
            normalized: True if `action` is already in the [0, 1]^d frame.
            raw: report in the objective's own units instead of maximization
                space. This undoes the sign too.

        Returns:
            (num_paths, n, m) draws, in maximization space: larger-is-better and
            standardized, matching `objective.standard_y`. With `raw=True`, in
            the units the measurements were taken in.
        """
        paths = _sample_paths(self.model, self.frame, action, num_paths, normalized)
        return self.objective.to_raw(paths) if raw else paths

    def best_actions(self, num_restarts=NUM_RESTARTS, raw_samples=RAW_SAMPLES,
                     raw=False):
        """The action maximizing each objective's posterior mean.

        Args:
            num_restarts: L-BFGS-B starting points per objective.
            raw_samples: Sobol samples scanned to pick those starting points.
            raw: report the values in each objective's own units instead of
                maximization space. The actions are in raw units either way.

        Returns:
            (actions, mu, std): (m, d) actions in raw units, and the length-m
            posterior mean and standard deviation of each objective at its own
            best action, in maximization space unless `raw` is set.
        """
        d = self.objective.normalized_x.shape[1]
        bounds = torch.stack([torch.zeros(d, dtype=DTYPE), torch.ones(d, dtype=DTYPE)])

        # one single-output problem per objective, over the normalized box
        actions = np.vstack(
            optimize_acqf(
                acq_function    = PosteriorMean(self.model),
                bounds          = bounds,
                q               = 1,
                num_restarts    = num_restarts,
                raw_samples     = raw_samples
            )[0].detach().numpy()
        )

        # (m, m) evaluated at m actions; the diagonal is each objective at its own
        mu, std = self.posterior_at(actions, normalized=True, raw=raw)
        return self.objective.xtransform.inv(actions), np.diag(mu), np.diag(std)

    @property
    def frame(self):
        """The `AffineTransform` normalizing actions into the GP's own frame."""
        return self.objective.xtransform

    @property
    def measured_x(self):
        """(n, d) measured actions, in the normalized frame."""
        return self.objective.normalized_x

    def incumbent(self):
        """The best value measured so far, in maximization space."""
        return self.objective.standard_y.max()

    def get_fitted_hyperparameters(self):
        """The GP's hyperparameters, as described by `gp_hyperparameters`."""
        return gp_hyperparameters(self.model)


class DecoupledMOGP:
    """Independent (decoupled) per-objective GPs over one shared action frame.
    """

    def __init__(self, objectives: DecoupledObjectives, fit_hyperparameters=True,
                 noise=NOISE_STD, length_scale=LENGTH_SCALE,
                 signal_var=SIGNAL_VAR, min_length_scale=None,
                 kernel_wrap_fn=None):
        """
        Args:
            objectives: the measurements to condition on.
            fit_hyperparameters: fit the lengthscales and signal variances by
                marginal likelihood, starting from `length_scale`/`signal_var`.
            noise: a `NoiseModel`, or the shorthands it coerces -- a number for
                a pinned fraction of each objective's own standard deviation,
                None to fit the noise. Shared by every sub-model, though each
                fits its own value from its own term of the likelihood.
            length_scale: starting ARD lengthscale, in the normalized frame.
            signal_var: starting ScaleKernel outputscale.
            min_length_scale: lower bound on every lengthscale, or None.
            kernel_wrap_fn: A function taking in a kernel and returning another.
                Used if the default RBF kernel needs to  be modified externally.
        """
        self.noise = NoiseModel.coerce(noise)
        if self.noise.is_fitted and not fit_hyperparameters:
            raise ValueError('a fitted noise is determined by the marginal '
                             'likelihood, so fit_hyperparameters must be True')
        self.fit_hyperparameters = fit_hyperparameters
        self.length_scale     = length_scale
        self.signal_var       = signal_var
        self.min_length_scale = min_length_scale
        self.kernel_wrap_fn   = kernel_wrap_fn
        self.update_feedback(objectives)

    def update_feedback(self, objectives: DecoupledObjectives):
        """Rebuild every GP against the current feedback.

        Args:
            objectives: the measurements to condition on.

        Returns:
            The `ModelListGP`, in eval mode.
        """
        self.objectives = objectives
        self.model = ModelListGP(*[
            build_botorch_gp(
                actions          = objectives.actions(i),
                values           = objectives.feedback(i),
                noise            = self.noise,
                signal_var       = self.signal_var,
                length_scale     = self.length_scale,
                min_length_scale = self.min_length_scale,
                kernel_wrap_fn   = self.kernel_wrap_fn,
            )
            for i in range(len(objectives))
        ]).eval()

        if self.fit_hyperparameters:
            # the sum splits over the sub-models because the objectives are
            # independent, so one call fits all of them
            fit_gpytorch_mll(
                SumMarginalLogLikelihood(self.model.likelihood, self.model)
            )
            self.model.eval()

        return self.model

    def posterior_at(self, action, normalized=False, raw=False, chunk=2048):
        """Posterior mean and standard deviation at arbitrary actions.

        Args:
            action: a single (d,) action or an (n, d) array of them.
            normalized: True if `action` is already in the shared [0, 1]^d frame.
            raw: report in each objective's own units instead of maximization
                space. This undoes the sign too.
            chunk: actions evaluated per posterior call.

        Returns:
            (mean, std), each (n, m), in maximization space: larger-is-better and
            standardized, matching `objectives.feedback()`. With `raw=True`, in
            the units the measurements were taken in.
        """
        mu, std = _posterior_at(self.model, self.frame, action, normalized, chunk)
        return self.objectives.to_raw(mu, std) if raw else (mu, std)

    def sample_paths(self, action, num_paths, normalized=False, raw=False):
        """Sample paths of the posterior over arbitrary actions.

        Drawn from the *joint* posterior over the whole set of actions, which is
        what makes each draw a function: sampling every action from its own
        marginal independently would discard the covariance between them and
        return white noise. That is also why this cannot chunk the way
        `posterior_at` does -- the n x n test-test block it avoids forming is
        the very object a joint draw needs.

        The objectives are decoupled, so a draw is joint over the actions but
        independent across the objectives: column j of every path comes from
        objective j's own GP and carries no covariance with column k.

        The draws come from torch's global generator, so seeding that is what
        makes them repeatable.

        Args:
            action: a single (d,) action or an (n, d) array of them.
            num_paths: paths drawn. Not botorch's `q`, which here is n: the
                actions are one q-batch, and this is how often it is sampled.
            normalized: True if `action` is already in the shared [0, 1]^d frame.
            raw: report in each objective's own units instead of maximization
                space. This undoes the sign too.

        Returns:
            (num_paths, n, m) draws, in maximization space: larger-is-better and
            standardized, matching `objectives.feedback()`. With `raw=True`, in
            the units the measurements were taken in.
        """
        paths = _sample_paths(self.model, self.frame, action, num_paths, normalized)
        return self.objectives.to_raw(paths) if raw else paths

    def best_actions(self, num_restarts=NUM_RESTARTS, raw_samples=RAW_SAMPLES,
                     raw=False):
        """The action maximizing each objective's posterior mean.

        Args:
            num_restarts: L-BFGS-B starting points per objective.
            raw_samples: Sobol samples scanned to pick those starting points.
            raw: report the values in each objective's own units instead of
                maximization space. The actions are in raw units either way.

        Returns:
            (actions, mu, std): (m, d) actions in raw units, and the length-m
            posterior mean and standard deviation of each objective at its own
            best action, in maximization space unless `raw` is set.
        """
        d = self.objectives.actions(0).shape[1]
        bounds = torch.stack([torch.zeros(d, dtype=DTYPE), torch.ones(d, dtype=DTYPE)])

        # one single-output problem per objective, over the normalized box
        actions = np.vstack([
            optimize_acqf(PosteriorMean(gp), bounds=bounds, q=1,
                          num_restarts=num_restarts, raw_samples=raw_samples
                          )[0].detach().numpy()
            for gp in self.model.models
        ])

        # (m, m) evaluated at m actions; the diagonal is each objective at its own
        mu, std = self.posterior_at(actions, normalized=True, raw=raw)
        return self.objectives.xtransform.inv(actions), np.diag(mu), np.diag(std)

    @property
    def frame(self):
        """The `AffineTransform` normalizing actions into the shared frame."""
        return self.objectives.xtransform

    def scalarized(self, weights) -> 'ScalarizedGP':
        """These GPs read through fixed weights, as one single-output GP."""
        return ScalarizedGP(self, weights)

    def get_fitted_hyperparameters(self):
        """Each objective's GP hyperparameters, in objective order.

        Returns:
            a length-m list of `GPHyperparameters`. The objectives are
            decoupled, so no entry is shared between them.
        """
        return [gp_hyperparameters(gp) for gp in self.model.models]


class ScalarizedGP:
    """One `DecoupledMOGP` read through fixed weights, as a single-output GP.

    A linear functional of a GP is itself a GP, so `g(x) = w^T f(x)` has

        mean      w^T mu(x)
        variance  w^T Sigma(x) w = sum_j w_j^2 sigma_j^2(x)
        kernel    k_w(x, x') = sum_j w_j^2 k_j(x, x')

    where the cross terms vanish because `DecoupledMOGP`'s objectives are
    independent and `Sigma` is therefore diagonal. This is exact rather than an
    approximation, and it is a read-out of models that are already fit: changing
    `weights` refits nothing and adds no hyperparameters.

    `feedback()` is standardized and sign-flipped to larger-is-better before any
    GP sees it, which is what makes `w^T f` dimensionally meaningful -- on raw
    units a weighted sum of a speed, a power and a height would mean nothing.

    Attributes:
        mogp: the `DecoupledMOGP` being read.
        objectives: its `DecoupledObjectives`.
        model: its `ModelListGP`, unchanged and shared.
        weights: (m,) weights, in maximization space.
        transform: the `ScalarizedPosteriorTransform` BoTorch reads them through.
    """

    def __init__(self, mogp: DecoupledMOGP, weights):
        """
        Args:
            mogp: the fitted per-objective GPs to scalarize.
            weights: (m,) weights, in maximization space, one per objective.
        """
        weights = np.asarray(weights, dtype=float).ravel()
        if weights.size != len(mogp.objectives):
            raise ValueError(
                f'{len(mogp.objectives)} objectives take {len(mogp.objectives)} '
                f'weights, got {weights.size}'
            )

        self.mogp       = mogp
        self.objectives = mogp.objectives
        self.model      = mogp.model
        self.weights    = weights
        self.transform  = ScalarizedPosteriorTransform(
            torch.as_tensor(weights, dtype=DTYPE)
        )

    def posterior_at(self, action, normalized=False, chunk=2048):
        """Posterior mean and standard deviation of `w^T f` at arbitrary actions.

        There is deliberately no `raw` flag: `w^T y` mixes the objectives' units
        and has no single objective's frame to invert into.

        Args:
            action: a single (d,) action or an (n, d) array of them.
            normalized: True if `action` is already in the shared [0, 1]^d frame.
            chunk: actions evaluated per posterior call.

        Returns:
            (mean, std), each (n, 1), in maximization space.
        """
        return _posterior_at(self.model, self.frame, action, normalized, chunk,
                             transform=self.transform)

    def sample_paths(self, action, num_paths, normalized=False):
        """Sample paths of `w^T f` over arbitrary actions.

        Drawn from the joint posterior over the whole set of actions, exactly as
        `DecoupledMOGP.sample_paths` is, so each draw is a function rather than
        noise. There is no `raw` flag, for the reason `posterior_at` gives.

        Args:
            action: a single (d,) action or an (n, d) array of them.
            num_paths: paths drawn.
            normalized: True if `action` is already in the shared [0, 1]^d frame.

        Returns:
            (num_paths, n, 1) draws, in maximization space.
        """
        return _sample_paths(self.model, self.frame, action, num_paths, normalized,
                             transform=self.transform)

    def best_actions(self, num_restarts=NUM_RESTARTS, raw_samples=RAW_SAMPLES):
        """The action maximizing the scalarized posterior mean.

        One optimization rather than `DecoupledMOGP`'s m of them, since the
        weights have already collapsed the m objectives to one.

        Args:
            num_restarts: L-BFGS-B starting points.
            raw_samples: Sobol samples scanned to pick them.

        Returns:
            (actions, mu, std): a (1, d) action in raw units, and the length-1
            posterior mean and standard deviation there, in maximization space.
        """
        d = self.objectives.action_dim
        bounds = torch.stack([torch.zeros(d, dtype=DTYPE), torch.ones(d, dtype=DTYPE)])

        actions, _ = optimize_acqf(
            acq_function = PosteriorMean(self.model, posterior_transform=self.transform),
            bounds       = bounds,
            q            = 1,
            num_restarts = num_restarts,
            raw_samples  = raw_samples
        )
        actions = actions.detach().numpy()

        mu, std = self.posterior_at(actions, normalized=True)
        return self.frame.inv(actions), np.diag(mu), np.diag(std)

    @property
    def frame(self):
        """The `AffineTransform` normalizing actions into the shared frame."""
        return self.objectives.xtransform

    @property
    def measured_x(self):
        """(n, d) union of every objective's measured actions, normalized.

        A union rather than one objective's actions because the objectives are
        decoupled: nothing requires them to share a design.
        """
        return np.unique(
            np.vstack([self.objectives.actions(i) for i in range(len(self.objectives))]),
            axis=0
        )

    def incumbent(self):
        """The best scalarized value inferred so far, in maximization space.

        Read off the posterior mean rather than the measurements: with decoupled
        data no action generally carries all m of them, so there is no observed
        `w^T y` to take a max over. This is the standard noisy-EI incumbent.
        """
        return self.posterior_at(self.measured_x, normalized=True)[0].max()

    def get_fitted_hyperparameters(self):
        """Each objective's GP hyperparameters, as `DecoupledMOGP` reports them.

        The scalarization is a read-out, not a fit, so the weights appear
        nowhere in them.
        """
        return self.mogp.get_fitted_hyperparameters()
