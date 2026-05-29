from tqdm import tqdm
import torch


from vpop_calibration.pynlme.model import NlmeModel
from vpop_calibration.config import smoke_test
from vpop_calibration.metropolis_hastings import MetropolisHastingsState, mh_step


def sample_conditional_distribution_nlme(
    nlme_model: NlmeModel,
    nb_samples: int = 1000,
    nb_burn_in: int = 50,
) -> torch.Tensor:
    """
    Sample random effects from the conditional distribution
    """

    if smoke_test:
        nb_samples = 2
        nb_burn_in = 1

    init_etas = nlme_model.sample_etas(1)
    init_predictions = nlme_model.log_posterior_etas_all_patients(init_etas)
    current_state = MetropolisHastingsState(
        etas=init_etas,
        gaussian_params=init_predictions.gaussian_params,
        prediction=init_predictions.predictions,
        log_prob=init_predictions.log_posterior,
        step_size=0.1,
        complete_likelihood=init_predictions.predictions.sum(dim=0),
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

    output = torch.stack(sample_list).squeeze(1)
    assert output.shape == (nb_samples, nlme_model.nb_patients, nlme_model.nb_pdu)
    return output
