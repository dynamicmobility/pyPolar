import argparse
import time
from datetime import datetime
from pathlib import Path
import socketio
from dataclasses import asdict
import numpy as np
import torch
import pypolar as plr
from tablet.survey import Survey
import logging
import hilo.shared.log as log

logger = logging.getLogger(__name__)
EXO_IP      = "192.168.1.122:5000"
sio = socketio.Client()
hilo = None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Fit a multi-objective GP over a HILO run.')
    parser.add_argument('--subject', required=True,
                        help='subject id; names the log, the dataset and its json')
    parser.add_argument('--resume', type=Path, default=None, metavar='DATASET',
                        help='a dataset directory (or json) to carry the run on from')
    parser.add_argument('--connect', action=argparse.BooleanOptionalAction, default=True,
                        help='send actions to the exo over the socket')
    parser.add_argument('--emulate', action=argparse.BooleanOptionalAction, default=False,
                        help='simulate the objectives instead of measuring them')

    return parser.parse_args(argv)


def load_backend(emulate: bool):
    global hilo
    if emulate:
        import hilo.shared.simulation as backend
    else:
        import hilo.shared.hardware as backend

    hilo = backend

def connect_to_exo(exo_ip: str = EXO_IP, retries: int = 5):
    """Dials the exo, and exits the run when the retries are spent."""
    for attempt in range(1, retries + 1):
        try:
            sio.connect(f'ws://{exo_ip}')
            return sio
        except socketio.exceptions.ConnectionError:
            # prints rather than logs: this runs before setup_logger
            print(f'Connection failed ({attempt}/{retries}). Retrying...')
            time.sleep(1)

    print('Exceeded maximum number of retries. Exiting...')
    exit()


def send_to_exo(action):
    logger.info(f'Exo got action {action}')
    ans = log.logged_input(f'Send {action} (y/n)? ')
    while True and ans.lower() != 'y':
        action = log.logged_input(f'Enter an alternative action as an array, like [1, 2, 3]: ')
        try:
            action = np.array(eval(action))
            if np.any(action < hilo.BOUNDS[0]) or np.any(action > hilo.BOUNDS[1]):
                raise ValueError('Action out of bounds')
            logger.info(f'Got {action}. Sending to exo...', extra=log.TO_BOTH)
            break
        except ValueError as e:
            logger.error(f'Action out of bounds. Note that bounds (low, high) = {hilo.BOUNDS}', extra=log.TO_BOTH)
            continue
        except Exception as e:
            logger.error(e, extra=log.TO_BOTH)
            logger.error(f'{action} did not compile. Try again.', extra=log.TO_BOTH)
            continue

    if not sio.connected:
        logger.info('Disabled!', extra=log.TO_BOTH)
        return True

    action_dict = {
        'h_flex_torque_scale': action[0], # make this a dict when sending to the exo
        'h_ext_torque_scale' : action[1],
        'hip_delay_idx'      : action[2]
    }
    sio.emit("update_inputs", action_dict)
    logger.info('Successfully sent action.', extra=log.TO_BOTH)

    return True


def find_dataset(resume: Path):
    """The dataset json a `--resume` argument names.

    Raises:
        FileNotFoundError: it is not a file. A run is resumed from a dataset,
            not from the directory holding one.
    """
    resume = Path(resume)
    if not resume.is_file():
        raise FileNotFoundError(f'{resume} is not a dataset file')

    return resume


def fit_gp(
    objective   : plr.DecoupledObjectives,
    noise       : plr.NoiseModel = None,
    hypers      : plr.GPHyperparameters = None
):
    return plr.DecoupledMOGP(
        objectives          = objective,
        noise               = hilo.GP_NOISE,
        fit_hyperparameters = True,
        min_length_scale    = hilo.MIN_LENGTHSCALE,
    )

def run_experiment(
    experiment        : plr.Logger,
    acqf              : plr.AcquisitionFunction,
    dataset           : plr.ExperimentDataset,
    send_to_exo       = None,
    start             : int = 0,
):
    # a resumed run already has measurements, so it has a GP before its first trial
    gp = fit_gp(experiment.objectives) if start else None
    # NUM_QUERIES is what one session asks for, not a total: resuming a finished
    # run of 45 takes 45 more rather than stopping the moment it starts
    for i in range(start, start + hilo.NUM_QUERIES):
        if i < hilo.NUM_RANDOM:
            # randomly sample if no data is collected
            source = 'random'
            action = plr.sample_actions(
                bounds = hilo.BOUNDS,
                n      = 1,
                kind   = 'uniform',
                seed   = hilo.SEED + i,
                dim    = hilo.DIM
            )[0]
        else:
            # fit gp + Acquisition strategy for the rest
            source = dataset.acquisition.strategy
            action = acqf.query(gp, q=1)[0]

        experiment.begin_trial(
            action         = action,
            device_send_fn = send_to_exo,
            args           = {
                hilo.METABOLIC: (action, i + 1),
                hilo.PUSH:   (action, i + 1, hilo.SURVEY_TIMEOUT, hilo.SURVEY_PERIOD)
            }
        )
        
        experiment.wait_for_measurements()
        experiment.end_trial() # updates the objectives
        
        gp = fit_gp(experiment.objectives)
        dataset.add_trial(
            objectives  = experiment.objectives,
            gp          = gp,
            action      = action,
            source      = source,
        )
        logger.info(f'Wrote {dataset.save()}')
    
    # the run's final state: every measurement, and the fit to all of them
    gp = fit_gp(experiment.objectives)
    dataset.add_trial(
        objectives  = experiment.objectives,
        gp          = gp,
    )
    
    return experiment, dataset

def connect_to_ipad(emulate: bool):
    if emulate:
        return None
    ipad = Survey(
        timeout = hilo.SURVEY_TIMEOUT,
        period  = hilo.SURVEY_PERIOD,
        logger  = logger
    )
    logger.info('Waiting for the iPad...', extra=log.TO_BOTH)
    ipad.wait_for_ipad()
    return ipad

def setup_experiment(ipad):
    probes     = hilo.make_probes_mo(ipad, multithread=hilo.MULTITHREAD)
    experiment = hilo.make_experiment_mo(
        probes    = probes,
        maximize  = hilo.MAXIMIZE
    )
    acqf_params = plr.AcquisitionParams(
        strategy          = hilo.ACQ_STRAT,
        seed              = hilo.SEED,
        num_objectives    = hilo.NUM_OBJECTIVES,
        raw_ref_point     = hilo.REF_POINT,
        **hilo.ACQ_KWARGS
    )
    return experiment, acqf_params

def main(argv=None):
    args = parse_args(argv)
    load_backend(args.emulate)

    torch.manual_seed(hilo.SEED)
    hilo.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log.setup_logger(hilo.OUTPUT_DIR / f'{args.subject}.log')

    if args.connect:
        connect_to_exo()
    ipad                    = connect_to_ipad(args.emulate)
    experiment, acqf_params = setup_experiment(ipad)
    
    dataset = plr.ExperimentDataset(
        name         = args.subject,
        subject      = args.subject,
        timestamp    = datetime.now().isoformat(timespec='seconds'),
        acquisition  = acqf_params,
        config       = asdict(
            hilo.Config(
                gp_noise          = plr.NoiseModel.coerce(hilo.GP_NOISE),
                min_lengthscale   = hilo.MIN_LENGTHSCALE,
                num_queries       = hilo.NUM_QUERIES,
                repeats           = hilo.REPEATS,
                dim               = hilo.DIM,
            )
        ),
        groundtruth  = hilo.GROUND_TRUTH_PARAMS if args.emulate else None,
        path         = hilo.OUTPUT_DIR / (args.subject + f'.json')
    )

    start = 0
    if args.resume is not None:
        prior = plr.ExperimentDataset.load(find_dataset(args.resume))
        logger.info(f'Resuming {prior}', extra=log.TO_BOTH)
        start = dataset.resume(prior)          # the trials
        experiment.resume(prior)               # the measurements they hold
        logger.info(f'Carrying on at trial {start}, through {start + hilo.NUM_QUERIES}',
                    extra=log.TO_BOTH)

    try:
        experiment, dataset = run_experiment(
            experiment        = experiment,
            acqf              = acqf_params.build(),
            dataset           = dataset,
            send_to_exo       = None if args.emulate else send_to_exo,
            start             = start,
        )
    finally:
        logger.info(f'Wrote {dataset.save()}', extra=log.TO_BOTH) # change this to save per trial...
        if ipad is not None: ipad.close()
        sio.disconnect()


if __name__ == '__main__':
    main()
