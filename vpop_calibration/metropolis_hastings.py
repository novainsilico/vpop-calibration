import torch
from typing import NamedTuple
import numpy as np

from vpop_calibration.pynlme.model import StatisticalModel
from vpop_calibration.config import device


class MetropolisHastingsState(NamedTuple):
    etas: torch.Tensor
    gaussian_params: torch.Tensor
    prediction: torch.Tensor
    log_prob: torch.Tensor
    step_size: float
    complete_likelihood: torch.Tensor


def mh_step(
    nlme_model: StatisticalModel,
    previous_state: MetropolisHastingsState,
    learning_rate: float,
    target_acceptance_rate: float = 0.234,
    verbose: bool = False,
) -> MetropolisHastingsState:
    """Perform one step of a Metropolis-Hastings transition kernel

    Args:
        nlme_model (StatisticalModel): The non-linear mixed effects model to use to compute the likelihoods
        previous_state (MetropolisHastingsState): Tuple containing the information from the previous step of Metropolis-Hastings
        target_acceptance_rate (float, optional): Target for the MCMC acceptance rate. Defaults to 0.234 [1]
        verbose (bool, optional): If true, print debug information to the console

        [1] Sherlock C. Optimal Scaling of the Random Walk Metropolis: General Criteria for the 0.234 Acceptance Rule. Journal of Applied Probability. 2013;50(1):1-15. doi:10.1239/jap/1363784420

    Returns:
        MetropolisHastingsState: the algorithm state to use for next iteration
    """
    nb_chains = previous_state.etas.shape[0]
    # Propose new etas
    proposal_noise = (
        torch.randn_like(previous_state.etas, device=device)
        @ nlme_model.omega_pop_lower_chol
    )
    proposal_etas = previous_state.etas + previous_state.step_size * proposal_noise
    # Compute their log posterior likelihood
    # This is the computation-heavy step:
    proposal = nlme_model.log_posterior_etas_all_patients(proposal_etas)
    proposal_log_prob = proposal.log_posterior

    assert proposal_log_prob.shape == previous_state.log_prob.shape

    # Define acceptance masks
    deltas: torch.Tensor = proposal_log_prob - previous_state.log_prob
    log_u: torch.Tensor = torch.log(torch.rand_like(deltas, device=device))
    accept_mask: torch.Tensor = log_u < deltas
    assert accept_mask.shape == (nb_chains, nlme_model.nb_patients)
    # Create a mask for parameters: last dimension of size nb of pdus
    accept_mask_parameters = accept_mask.unsqueeze(-1).expand(
        -1, -1, previous_state.etas.shape[-1]
    )
    # Create a mask for predictions, mapping patients index to row index in the predictions
    accept_mask_predictions = accept_mask.index_select(
        1, nlme_model.data.full_obs.obs_index.id.index_values
    )
    # Accept the different variables using the masks
    new_etas = torch.where(
        accept_mask_parameters, proposal_etas, previous_state.etas
    ).to(device)
    new_gaussian_params = torch.where(
        accept_mask_parameters, proposal.gaussian_params, previous_state.gaussian_params
    ).to(device)
    new_log_prob = torch.where(
        accept_mask, proposal_log_prob, previous_state.log_prob
    ).to(device)
    # Compute the complete likelihood by averaging over chains and summing over patients
    new_complete_likelihood = -2 * new_log_prob.mean(dim=0).sum(dim=0)
    new_pred = torch.where(
        accept_mask_predictions, proposal.predictions, previous_state.prediction
    ).to(device)
    new_acceptance_rate: float = accept_mask.cpu().float().mean().mean().item()
    if verbose:
        print(f"  Acceptance rate: {new_acceptance_rate:.2f}")
    new_step_size: float = previous_state.step_size * np.exp(
        learning_rate * (new_acceptance_rate - target_acceptance_rate)
    )

    new_state = MetropolisHastingsState(
        etas=new_etas,
        gaussian_params=new_gaussian_params,
        prediction=new_pred,
        log_prob=new_log_prob,
        step_size=new_step_size,
        complete_likelihood=new_complete_likelihood,
    )

    return new_state
