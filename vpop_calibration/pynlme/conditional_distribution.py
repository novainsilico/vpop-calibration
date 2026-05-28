from tqdm import tqdm
import torch


from vpop_calibration.pynlme.model import NlmeModel
from vpop_calibration.config import smoke_test
from vpop_calibration.metropolis_hastings import MetropolisHastingsState, mh_step


def sample_conditional_distribution(
    nlme_model: NlmeModel,
    nb_samples: int = 1000,
    nb_burn_in: int = 50,
):
    """
    Returns: cond_dist_samples: dim(nb_samples, nb_patients, nb_PDU)
    """

    if smoke_test:
        nb_samples = 2
        nb_burn_in = 1

    init_etas = nlme_model.sample_etas(1)
    output = nlme_model.log_posterior_etas(init_etas)
    current_state = MetropolisHastingsState(
        etas=init_etas,
        gaussian_params=output.gaussian_params,
        prediction=output.predictions,
        log_prob=output.log_posterior,
        step_size=0.1,
        complete_likelihood=output.predictions.sum(dim=0),
    )
    sample_list = []
    print(f"Sampling conditional distribution on {nb_samples} samples:")
    for i in tqdm(range(nb_burn_in + nb_samples)):
        current_state = mh_step(
            nlme_model=nlme_model,
            previous_state=current_state,
            learning_rate=0.0,
        )

        if i >= nb_burn_in:
            sample_list.append(current_state.etas)

    return torch.stack(sample_list).squeeze(1)
