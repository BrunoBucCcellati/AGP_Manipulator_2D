import sys
import math
import time
import statistics
from io import StringIO

import optuna
import iOpt


def forward_kinematics(angles, lengths):
    n = len(angles)
    phi = 0.0
    x = 0.0
    y = 0.0
    cos = math.cos
    sin = math.sin
    for i in range(n):
        phi += angles[i]
        L = lengths[i]
        x += L * cos(phi)
        y += L * sin(phi)
    return x, y


def manipulator_cost(angles, lengths, target_x, target_y, min_theta):
    x, y = forward_kinematics(angles, lengths)
    dx = x - target_x
    dy = y - target_y
    sqrt = math.sqrt
    pow_ = math.pow
    log1p = math.log1p
    exp = math.exp
    abs_ = abs
    dist = sqrt(dx * dx + dy * dy)
    arch_bias_w = 0.02
    arch_bias_k = 3.0
    sharp_w = 0.05
    scale = 2.0 / (min_theta + 1e-6)
    pen_c = 0.0
    arch_pen = 0.0
    for theta in angles:
        a = abs_(theta)
        v = a - min_theta
        if v > 0.0:
            pen_c += sharp_w * (pow_(2.0, scale * v) - 1.0)
        t = -theta * arch_bias_k
        if t > 10.0:
            sp = t
        else:
            sp = log1p(exp(t))
        arch_pen += arch_bias_w * sp
    return dist + pen_c + arch_pen


def build_angles_lengths(trial, n_seg, var_len):
    theta0_min = -1.0471975511965977
    theta0_max = 2.6179938779914944
    theta_min = -2.6179938779914944
    theta_max = 2.6179938779914944
    suggest_float = trial.suggest_float
    angles = []
    if n_seg > 0:
        angles.append(suggest_float("theta_0", theta0_min, theta0_max))
        for i in range(1, n_seg):
            name = "theta_" + str(i)
            angles.append(suggest_float(name, theta_min, theta_max))
    if var_len:
        lengths = []
        for i in range(n_seg):
            name = "L_" + str(i)
            lengths.append(suggest_float(name, 0.5, 2.0))
    else:
        lengths = [1.0] * n_seg
    return angles, lengths


def run_optuna(n_seg, var_len, min_theta, tx, ty, max_iter, r_param, eps):
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    build = build_angles_lengths
    cost = manipulator_cost

    def objective(trial):
        angles, lengths = build(trial, n_seg, var_len)
        return cost(angles, lengths, tx, ty, min_theta)

    study = optuna.create_study(direction="minimize")
    start_time = time.perf_counter()
    study.optimize(objective, n_trials=max_iter, show_progress_bar=False)
    elapsed_micros = (time.perf_counter() - start_time) * 1e6
    best_trial = study.best_trial
    best_f = best_trial.value
    params = best_trial.params
    angles = [params["theta_" + str(i)] for i in range(n_seg)]
    if var_len:
        lengths = [params["L_" + str(i)] for i in range(n_seg)]
    else:
        lengths = [1.0] * n_seg
    best_x, best_y = forward_kinematics(angles, lengths)
    trials = study.trials
    iterations = len(trials)
    recent_trials = max(1, iterations // 10)
    recent_values = [t.value for t in trials[-recent_trials:] if t.value is not None]
    achieved_eps = statistics.stdev(recent_values)
    q_values = angles + lengths
    print("BEST_F", best_f)
    print("BEST_X", best_x)
    print("BEST_Y", best_y)
    print("ITERATIONS", iterations)
    print("EPS", achieved_eps)
    print("TIME", int(elapsed_micros))
    print("Q", " ".join(str(q) for q in q_values))


def run_iopt(n_seg, var_len, min_theta, tx, ty, max_iter, r_param, eps, adaptive):
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = StringIO()
    sys.stderr = StringIO()
    build = build_angles_lengths
    cost = manipulator_cost

    def objective(trial):
        angles, lengths = build(trial, n_seg, var_len)
        return cost(angles, lengths, tx, ty, min_theta)

    study = iOpt.create_study()
    params = iOpt.SolverParameters(r=r_param, eps=eps, iters_limit=max_iter, refine_solution=adaptive)
    study.optimize(objective=objective, solver_parameters=params, type_of_painter="none", console_mode="off")
    sys.stdout = old_stdout
    sys.stderr = old_stderr
    best_params = study.best_float_params_()
    angles = best_params[:n_seg]
    if var_len:
        lengths = best_params[n_seg:n_seg * 2]
    else:
        lengths = [1.0] * n_seg
    best_f = study.best_values_()
    best_x, best_y = forward_kinematics(angles, lengths)
    solution = getattr(study, "solution", None)
    achieved_eps = solution.solution_accuracy
    iterations = solution.number_of_global_trials
    elapsed_micros = solution.solving_time * 1e6
    q_values = angles + lengths
    print("BEST_F", best_f)
    print("BEST_X", best_x)
    print("BEST_Y", best_y)
    print("ITERATIONS", iterations)
    print("EPS", achieved_eps)
    print("TIME", int(elapsed_micros))
    print("Q", " ".join(str(q) for q in q_values))


def main():
    argv = sys.argv
    backend = argv[1]
    n_seg = int(argv[2])
    var_len = bool(int(argv[3]))
    min_theta = float(argv[4])
    tx = float(argv[5])
    ty = float(argv[6])
    _levels = int(argv[7])
    max_iter = int(argv[8])
    r_param = float(argv[9])
    eps = float(argv[10])
    adaptive = bool(int(argv[11]))
    if backend == "optuna":
        run_optuna(n_seg, var_len, min_theta, tx, ty, max_iter, r_param, eps)
    elif backend == "iopt":
        run_iopt(n_seg, var_len, min_theta, tx, ty, max_iter, r_param, eps, adaptive)


if __name__ == "__main__":
    main()
