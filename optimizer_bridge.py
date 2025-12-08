import sys
import math
import time
import statistics
import warnings
from io import StringIO

import optuna
from optuna.samplers import CmaEsSampler
import iOpt


def forward_kinematics(angles, lengths):
    phi = x = y = 0.0
    angle_iter = iter(angles)
    length_iter = iter(lengths)
    for theta, L in zip(angle_iter, length_iter):
        phi += theta
        x += L * math.cos(phi)
        y += L * math.sin(phi)
    return x, y


def manipulator_cost(angles, lengths, target_x, target_y, min_theta):
    x, y = forward_kinematics(angles, lengths)
    dx, dy = x - target_x, y - target_y
    dist = math.hypot(dx, dy)
    scale = 2.0 / (min_theta + 1e-6)
    pen_c = arch_pen = 0.0

    for theta in angles:
        v = abs(theta) - min_theta
        if v > 0.0:
            scaled = scale * v
            if scaled < 100.0:
                pen_c += 0.05 * (1 << int(scaled))
            else:
                pen_c += 0.05 * (2.0 ** scaled - 1.0)
        t = -theta * 3.0
        if t > 10.0:
            arch_pen += 0.02 * t
        else:
            arch_pen += 0.02 * math.log1p(math.exp(t))

    return dist + pen_c + arch_pen


def build_angles_lengths(trial, n_seg, var_len, baseLength, stretchFactor):
    angles = []
    for i in range(n_seg):
        low = -1.0471975511965977 if i == 0 else -2.6179938779914944
        angles.append(trial.suggest_float(f"theta_{i}", low, 2.6179938779914944))

    if var_len:
        length_lower = baseLength / stretchFactor
        length_upper = baseLength * stretchFactor
        lengths = [trial.suggest_float(f"L_{i}", length_lower, length_upper)
                  for i in range(n_seg)]
    else:
        lengths = [baseLength] * n_seg

    return angles, lengths


def run_optuna(n_seg, var_len, min_theta, tx, ty, max_iter,
              baseLength, stretchFactor):
    warnings.filterwarnings('ignore')
    optuna.logging.set_verbosity(optuna.logging.ERROR)

    dimension = n_seg * (2 if var_len else 1)
    sampler = CmaEsSampler(
        seed=42,
        n_startup_trials=(dimension - 1) ** 2 + 3,
        consider_pruned_trials=False
    )

    def objective(trial):
        angles, lengths = build_angles_lengths(trial, n_seg, var_len,
                                              baseLength, stretchFactor)
        return manipulator_cost(angles, lengths, tx, ty, min_theta)

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=optuna.pruners.NopPruner()
    )

    start_time = time.perf_counter()
    study.optimize(objective, n_trials=max_iter, show_progress_bar=False,
                  gc_after_trial=False)
    elapsed_micros = (time.perf_counter() - start_time) * 1e6

    best_trial = study.best_trial
    params = best_trial.params

    angles = [params[f"theta_{i}"] for i in range(n_seg)]
    lengths = [params[f"L_{i}"] for i in range(n_seg)] if var_len else [1.0] * n_seg

    best_x, best_y = forward_kinematics(angles, lengths)
    iterations = len(study.trials)

    recent_trials = iterations // 10
    recent_values = [t.value for t in study.trials[-recent_trials:]]
    achieved_eps = statistics.stdev(recent_values) if recent_trials > 1 else 0.0

    return {
        "BEST_F": float(best_trial.value),
        "BEST_X": float(best_x),
        "BEST_Y": float(best_y),
        "ITERATIONS": int(iterations),
        "EPS": float(achieved_eps),
        "TIME": float(elapsed_micros),
        "ANGLES": angles,
        "LENGTHS": lengths
    }


def run_iopt(n_seg, var_len, min_theta, tx, ty, levels, max_iter,
             r_param, eps, adaptive, baseLength, stretchFactor):
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = StringIO(), StringIO()

    def objective(trial):
        angles, lengths = build_angles_lengths(trial, n_seg, var_len,
                                              baseLength, stretchFactor)
        return manipulator_cost(angles, lengths, tx, ty, min_theta)

    study = iOpt.create_study()
    params = iOpt.SolverParameters(eps=eps, r=r_param, iters_limit=max_iter,
                                  evolvent_density=levels, refine_solution=adaptive)
    study.optimize(objective=objective, solver_parameters=params,
                  type_of_painter="none", console_mode="off")

    sys.stdout, sys.stderr = old_stdout, old_stderr

    best_params = study.best_float_params_()
    angles = best_params[:n_seg]
    lengths = best_params[n_seg:n_seg*2] if var_len else [1.0] * n_seg

    best_f = study.best_values_()
    best_x, best_y = forward_kinematics(angles, lengths)

    solution = getattr(study, "solution", None)
    achieved_eps = solution.solution_accuracy
    iterations = solution.number_of_global_trials
    elapsed_micros = solution.solving_time * 1e6

    return {
        "BEST_F": float(best_f),
        "BEST_X": float(best_x),
        "BEST_Y": float(best_y),
        "ITERATIONS": int(iterations),
        "EPS": float(achieved_eps),
        "TIME": float(elapsed_micros),
        "ANGLES": angles,
        "LENGTHS": lengths
    }


def main():
    argv = sys.argv
    backend, n_seg, var_len = argv[1], int(argv[2]), bool(int(argv[3]))
    min_theta, tx, ty = float(argv[4]), float(argv[5]), float(argv[6])
    max_iter = int(argv[8])
    baseLength = float(argv[9])
    stretchFactor = float(argv[10])

    if backend == "optuna":
        run_optuna(n_seg, var_len, min_theta, tx, ty, max_iter,
                  baseLength, stretchFactor)
    else:
        levels, r_param, eps, adaptive = int(argv[7]), float(argv[11]), float(argv[12]), bool(int(argv[13]))
        run_iopt(n_seg, var_len, min_theta, tx, ty, levels, max_iter,
                 r_param, eps, adaptive, baseLength, stretchFactor)
