from vpop_calibration.saem.scheduler import (
    SaemScheduler,
)


def test_scheduler():
    scheduler = SaemScheduler(
        nb_iter_burnin=2,
        nb_iter_learning=2,
        nb_iter_smoothing=2,
        init_step_adaptation=0.5,
        learning_rate_power=0.8,
        patience=5,
    )

    output = []
    for _ in scheduler:
        output.append(
            (
                scheduler.iteration,
                scheduler.phase,
                scheduler.mh_learning_rate,
                scheduler.stochastic_approximation_rate,
            )
        )

    expected_output = [
        (0, "burnin", 0.5, 1.0),
        (1, "burnin", 0.5, 1.0),
        (2, "learning", 0.5, 1.0),
        (3, "learning", 0.5 / (2**0.5), 1.0),
        (4, "smoothing", 0.0, 1.0),
        (5, "smoothing", 0.0, 1 / 2**0.8),
    ]

    assert output == expected_output
