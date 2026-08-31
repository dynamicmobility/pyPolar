"""Whether a fit's Pareto front really does beat what it calls dominated.

Reads a finished run, takes K actions off its posterior's Pareto front and K
off its anti-front, writes them beside that run as a csv, and then measures
every one of them on the subject. There is no GP in the loop: the fit chose the
actions, and after that this is a plain measurement script.

The order is drawn at random, so the two sets interleave rather than arriving
as blocks -- a subject who can see the blocks is rating the block, not the
action. That draw is made once, by the plain run, and the csv it writes is the
record of it: `--reversed` reads that csv back and reverses the order it finds,
rather than drawing again. Running one subject each way is what separates a
genuine front effect from an ordering or fatigue one.

Nothing is ever written over. A run whose files are already in the directory
raises instead, since the alternative is quietly destroying a subject's data --
`--resume` is how a session that died partway is picked up, continuing the same
files from the trial it stopped on.
"""

import argparse
import csv
from ast import literal_eval
from datetime import datetime
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
import torch
import pypolar as plr
import logging
import hilo.shared.log as log
from hilo.fit_mogp import connect_to_exo, connect_to_ipad, find_dataset, send_to_exo, sio
import hilo.fit_mogp as fit_mogp

logger = logging.getLogger(__name__)
K       = 3        # actions per set
SCAN    = 2**14    # Sobol points the fronts are read off
SEED    = 95
PARETO  = 'pareto'
ANTI    = 'antipareto'
hilo = None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Measure a fit\'s Pareto front against its anti-front.')
    parser.add_argument('--dataset', required=True, type=Path,
                        help='the run to read the fronts off; everything is written beside it')
    parser.add_argument('--reversed', action='store_true',
                        help='present the actions last-to-first, into evaluation_reversed')
    parser.add_argument('--resume', action='store_true',
                        help='carry on a session that stopped partway, from its own files')
    parser.add_argument('--connect', action=argparse.BooleanOptionalAction, default=True,
                        help='send actions to the exo over the socket')
    parser.add_argument('--emulate', action=argparse.BooleanOptionalAction, default=False,
                        help='simulate the objectives instead of measuring them')

    return parser.parse_args(argv)


def load_backend(emulate: bool):
    # the device layer is fit_mogp's, so its backend is set alongside this one
    global hilo
    fit_mogp.load_backend(emulate)
    hilo = fit_mogp.hilo


def peel(mu, k, sign, exclude=None):
    """Indices of at least k points on the `sign` end of `mu`: non-dominated
    layers peeled until enough have accumulated. A smooth posterior often puts
    a single point on the first anti-Pareto layer.

    Args:
        mu: (n, m) objective values, higher is better.
        sign: +1 for the Pareto front, -1 for the anti-front.
        exclude: indices to leave out. An extreme point can sit on the Pareto
            front and the anti-front at once, so the second end is peeled from
            what the first did not claim; without this the same action can be
            handed to a subject twice, once under each label.
    """
    left = np.setdiff1d(np.arange(len(mu)), [] if exclude is None else exclude)
    out  = np.array([], dtype=int)
    while len(out) < k and len(left):
        layer = left[plr.get_nondominated(sign * mu[left])]
        out   = np.concatenate([out, layer])
        left  = np.setdiff1d(left, layer)
    return out


def spread(idxs, F, k):
    """k of `idxs` covering the set: the most central member, then greedily
    whichever is furthest from everything already chosen.

    Args:
        idxs: (n,) candidate indices into F.
        F: (N, m) objective values, normalized onto [0, 1] per column.

    Returns:
        (k,) of `idxs`.
    """
    if len(idxs) <= k:
        return idxs

    S      = F[idxs]
    chosen = [int(np.argmin(cdist(S, S.mean(axis=0, keepdims=True))))]
    while len(chosen) < k:
        chosen.append(int(np.argmax(cdist(S, S[chosen]).min(axis=1))))

    return idxs[chosen]


def front_samples(dataset: plr.ExperimentDataset, rng=None):
    """The actions to measure, in the order they are presented.

    The run's own final fit is rebuilt from its state dict rather than refit, so
    the fronts are the ones that run actually inferred. A Sobol scan of the
    whole box is read for them, K taken off each end.

    The order is then drawn at random, which is what interleaves the two sets: a
    subject given three of one and then three of the other is rating the block
    as much as the action. Which actions are chosen stays deterministic under
    SEED -- only their order is drawn.

    Args:
        dataset: the finished run to read.
        rng: the generator the order is drawn from. Defaults to a fresh one, so
            two subjects do not share an order.

    Returns:
        (actions, estimates, kinds, names) -- (2K, d) actions in raw units, the
        (2K, m) posterior means at them in the objectives' own units, the set
        each came from, and the objective names the estimates are of.
    """
    mogp      = dataset.get_model()
    X         = plr.sample_actions(mogp.action_bounds, SCAN, 'sobol', SEED)
    mu, _     = mogp.posterior_at(X)             # maximization space
    raw_mu, _ = mogp.posterior_at(X, raw=True)   # the objectives' own units

    # a shared [0, 1] metric per objective, so both fronts are selected alike
    F     = (mu - mu.min(axis=0)) / np.ptp(mu, axis=0)
    front = peel(mu, K, +1)
    idxs  = np.concatenate([spread(front, F, K),
                            spread(peel(mu, K, -1, exclude=front), F, K)])
    kinds = np.array([PARETO] * K + [ANTI] * K)

    order = (rng or np.random.default_rng()).permutation(len(idxs))

    return X[idxs[order]], raw_mu[idxs[order]], kinds[order], mogp.objectives.names


def write_csv(path: Path, actions, estimates, kinds, names):
    """The actions to be measured, one row per evaluation trial."""
    with path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['trial', 'action', *(f'{name} est' for name in names), 'type'])
        for i, (action, estimate, kind) in enumerate(zip(actions, estimates, kinds), start=1):
            writer.writerow([
                i,
                '[' + ', '.join(f'{a:.6g}' for a in action) + ']',
                *(f'{v:.6g}' for v in estimate),
                kind,
            ])

    return path


def read_samples(path: Path):
    """A written csv read back: the inverse of `write_csv`.

    This is what `--reversed` reverses. The order was drawn once, by the plain
    run, so the file is the record of it -- drawing again would give a
    different order and the two runs would no longer be the same experiment
    backwards.

    Raises:
        FileNotFoundError: the plain run has not been done yet.
    """
    if not path.is_file():
        raise FileNotFoundError(f'{path} is not there; run without --reversed '
                                'first, so there is an order to reverse')

    with path.open(newline='') as f:
        header, *rows = list(csv.reader(f))

    names     = [column.removesuffix(' est') for column in header[2:-1]]
    actions   = np.array([literal_eval(row[1]) for row in rows], dtype=float)
    estimates = np.array([row[2:-1] for row in rows], dtype=float)
    kinds     = np.array([row[-1] for row in rows])

    return actions, estimates, kinds, names


def plan(actions, kinds, done: int = 0):
    """The order to be run, one line per trial, `>` marking the next one up.

    Printed before the first trial and again if the run dies, so a session that
    fails partway can be picked up by hand from the log.
    """
    return '\n'.join(
        f'  {">" if i == done else " "} {i + 1:>2}  {kind:<10} '
        f'{np.array2string(action, precision=4, suppress_small=True)}'
        for i, (action, kind) in enumerate(zip(actions, kinds))
    )


def refuse_overwrite(*paths: Path):
    """Raises rather than write over a run that has already been taken.

    Raises:
        FileExistsError: any of `paths` is already there.
    """
    taken = [str(path) for path in paths if Path(path).exists()]
    if taken:
        raise FileExistsError(f'{", ".join(taken)} already there; move or delete '
                              'what is worth keeping, this will not overwrite it')


def run_experiment(
    experiment        : plr.Logger,
    dataset           : plr.ExperimentDataset,
    actions           : np.ndarray,
    kinds             : np.ndarray,
    send_to_exo       = None,
    start             : int = 0,
):
    for i in range(start, len(actions)):
        action, kind = actions[i], kinds[i]
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

        # no gp: the fit that chose these actions was the one being tested
        dataset.add_trial(
            objectives  = experiment.objectives,
            action      = action,
            source      = kind,
        )
        logger.info(f'trial {i + 1}/{len(actions)} ({kind}) done, '
                    f'wrote {dataset.save()}', extra=log.TO_BOTH)

    # the run's final state: every measurement, and nothing fit to them
    dataset.add_trial(objectives = experiment.objectives)

    return experiment, dataset

def setup_experiment(ipad):
    probes     = hilo.make_probes_mo(ipad, multithread=hilo.MULTITHREAD)
    experiment = hilo.make_experiment_mo(
        probes    = probes,
        maximize  = hilo.MAXIMIZE
    )
    return experiment

def main(argv=None):
    args = parse_args(argv)
    load_backend(args.emulate)

    torch.manual_seed(hilo.SEED)
    source  = plr.ExperimentDataset.load(find_dataset(args.dataset))
    out_dir = Path(args.dataset).parent
    stem    = 'evaluation_reversed' if args.reversed else 'evaluation'

    # every precondition is settled before anything is created: opening the
    # logger writes the log file, and a half-made run would block the next try
    prior = None
    if args.resume:
        # the plan and the trials already taken are this run's own files
        actions, estimates, kinds, names = read_samples(out_dir / f'{stem}.csv')
        prior = plr.ExperimentDataset.load(find_dataset(out_dir / f'{stem}.json'))
    else:
        refuse_overwrite(*(out_dir / f'{stem}.{suffix}' for suffix in ('csv', 'json', 'log')))
        if args.reversed:
            # the plain run drew the order; this one only turns it around
            actions, estimates, kinds, names = read_samples(out_dir / 'evaluation.csv')
            actions, estimates, kinds = actions[::-1], estimates[::-1], kinds[::-1]
        else:
            actions, estimates, kinds, names = front_samples(source)

    log.setup_logger(out_dir / f'{stem}.log')   # appends, so a resumed run keeps one log
    if not args.resume:
        logger.info(f'Wrote {write_csv(out_dir / f"{stem}.csv", actions, estimates, kinds, names)}',
                    extra=log.TO_BOTH)

    if args.connect:
        connect_to_exo()
    ipad       = connect_to_ipad(args.emulate)
    experiment = setup_experiment(ipad)

    dataset = plr.ExperimentDataset(
        name         = stem,
        subject      = source.subject or source.name,
        # a resumed session is the same run continued in place, not a new one,
        # so it keeps the moment it began rather than the moment it restarted
        timestamp    = (prior or source).timestamp if args.resume
                       else datetime.now().isoformat(timespec='seconds'),
        config       = {
            'evaluates' : str(args.dataset),
            'k'         : K,
            'scan'      : SCAN,
            'seed'      : SEED,
            'reversed'  : args.reversed,
            'repeats'   : hilo.REPEATS,
            'dim'       : hilo.DIM,
        },
        groundtruth  = hilo.GROUND_TRUTH_PARAMS if args.emulate else None,
        path         = out_dir / f'{stem}.json'
    )

    start = 0
    if prior is not None:
        start = dataset.resume(prior)   # the trials, and where they stopped
        experiment.resume(prior)        # the measurements they hold

    logger.info(f'{stem}: {start}/{len(actions)} measured\n' + plan(actions, kinds, start),
                extra=log.TO_BOTH)

    ans = log.logged_input('Proceed? (y/n) ')
    if ans == 'n':
        print('Quitting')
        quit()
        
    try:
        experiment, dataset = run_experiment(
            experiment        = experiment,
            dataset           = dataset,
            actions           = actions,
            kinds             = kinds,
            send_to_exo       = None if args.emulate else send_to_exo,
            start             = start,
        )
    finally:
        done = len(dataset.get_actions())
        logger.info(f'{done}/{len(actions)} trials measured, wrote {dataset.save()}\n'
                    + plan(actions, kinds, done), extra=log.TO_BOTH)
        if ipad is not None: ipad.close()
        sio.disconnect() # a no-op on a client that never connected


if __name__ == '__main__':
    main()
