import sys
import math
import time
import statistics
import warnings
from io import StringIO

import numpy as np
import optuna
from optuna.samplers import CmaEsSampler, TPESampler
from optuna.trial import TrialState

import iOpt
from iOpt.problem import Problem
from iOpt.trial import FunctionType


MANIP_CONSTRAINT_NONE = 0
MANIP_CONSTRAINT_OBSTACLES = 1
MANIP_CONSTRAINT_NONLINEAR = 2
MANIP_CONSTRAINT_OBSTACLES_AND_NONLINEAR = 3

OBSTACLE_CLEARANCE = 0.05
GEOM_EPS = 1e-12


def uses_obstacles(constraint_mode, obstacles):
    return ((constraint_mode & MANIP_CONSTRAINT_OBSTACLES) != 0) and bool(obstacles)


def uses_nonlinear(constraint_mode):
    return (constraint_mode & MANIP_CONSTRAINT_NONLINEAR) != 0


def nonlinear_constraints_for_pose(x, y):
    c0 = (x - 5.0) ** 2 + y ** 2 - 25.0
    c1 = 7.7 - ((x - 8.0) ** 2 + (y + 3.0) ** 2)
    return (c0, c1)


def clamp01(v):
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def wrap_pi(a):
    two_pi = 2.0 * math.pi
    w = math.remainder(float(a), two_pi)
    if w <= -math.pi:
        w += two_pi
    elif w > math.pi:
        w -= two_pi
    return w


def choose_best_optuna_trial(study):
    completed = [
        t for t in study.trials
        if t.state == TrialState.COMPLETE and t.value is not None
    ]
    if not completed:
        raise RuntimeError("Optuna did not produce any completed trials.")

    feasible = [
        t for t in completed
        if all(float(v) <= 0.0 for v in t.user_attrs.get("constraint", ()))
    ]
    if feasible:
        return min(feasible, key=lambda t: float(t.value))

    def violation_key(trial):
        cons = tuple(float(v) for v in trial.user_attrs.get("constraint", ()))
        max_viol = max((max(0.0, v) for v in cons), default=float("inf"))
        sum_viol = sum(max(0.0, v) for v in cons)
        return (max_viol, sum_viol, float(trial.value))

    return min(completed, key=violation_key)


def point_aabb_distance_sq(px, py, min_x, min_y, max_x, max_y):
    dx = 0.0
    if px < min_x:
        dx = min_x - px
    elif px > max_x:
        dx = px - max_x
    dy = 0.0
    if py < min_y:
        dy = min_y - py
    elif py > max_y:
        dy = py - max_y
    return dx * dx + dy * dy


def point_segment_distance_sq(px, py, ax, ay, bx, by):
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay
    ab2 = abx * abx + aby * aby
    if ab2 <= GEOM_EPS:
        return apx * apx + apy * apy
    t = clamp01((apx * abx + apy * aby) / ab2)
    qx = ax + t * abx
    qy = ay + t * aby
    dx = px - qx
    dy = py - qy
    return dx * dx + dy * dy


def orient2d(ax, ay, bx, by, cx, cy):
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def on_segment(ax, ay, bx, by, px, py):
    eps = 1e-6
    return (
        min(ax, bx) - eps <= px <= max(ax, bx) + eps
        and min(ay, by) - eps <= py <= max(ay, by) + eps
    )


def segments_intersect(ax, ay, bx, by, cx, cy, dx, dy):
    o1 = orient2d(ax, ay, bx, by, cx, cy)
    o2 = orient2d(ax, ay, bx, by, dx, dy)
    o3 = orient2d(cx, cy, dx, dy, ax, ay)
    o4 = orient2d(cx, cy, dx, dy, bx, by)
    eps = 1e-6
    if (((o1 > eps and o2 < -eps) or (o1 < -eps and o2 > eps))
            and ((o3 > eps and o4 < -eps) or (o3 < -eps and o4 > eps))):
        return True
    if abs(o1) <= eps and on_segment(ax, ay, bx, by, cx, cy):
        return True
    if abs(o2) <= eps and on_segment(ax, ay, bx, by, dx, dy):
        return True
    if abs(o3) <= eps and on_segment(cx, cy, dx, dy, ax, ay):
        return True
    if abs(o4) <= eps and on_segment(cx, cy, dx, dy, bx, by):
        return True
    return False


def segment_segment_distance_sq(ax, ay, bx, by, cx, cy, dx, dy):
    if segments_intersect(ax, ay, bx, by, cx, cy, dx, dy):
        return 0.0
    best = point_segment_distance_sq(ax, ay, cx, cy, dx, dy)
    best = min(best, point_segment_distance_sq(bx, by, cx, cy, dx, dy))
    best = min(best, point_segment_distance_sq(cx, cy, ax, ay, bx, by))
    best = min(best, point_segment_distance_sq(dx, dy, ax, ay, bx, by))
    return best


def point_in_aabb(px, py, min_x, min_y, max_x, max_y):
    return min_x <= px <= max_x and min_y <= py <= max_y


def segment_aabb_distance_sq(ax, ay, bx, by, min_x, min_y, max_x, max_y):
    if point_in_aabb(ax, ay, min_x, min_y, max_x, max_y) or point_in_aabb(bx, by, min_x, min_y, max_x, max_y):
        return 0.0
    if (segments_intersect(ax, ay, bx, by, min_x, min_y, max_x, min_y)
            or segments_intersect(ax, ay, bx, by, max_x, min_y, max_x, max_y)
            or segments_intersect(ax, ay, bx, by, max_x, max_y, min_x, max_y)
            or segments_intersect(ax, ay, bx, by, min_x, max_y, min_x, min_y)):
        return 0.0
    best = point_aabb_distance_sq(ax, ay, min_x, min_y, max_x, max_y)
    best = min(best, point_aabb_distance_sq(bx, by, min_x, min_y, max_x, max_y))
    best = min(best, segment_segment_distance_sq(ax, ay, bx, by, min_x, min_y, max_x, min_y))
    best = min(best, segment_segment_distance_sq(ax, ay, bx, by, max_x, min_y, max_x, max_y))
    best = min(best, segment_segment_distance_sq(ax, ay, bx, by, max_x, max_y, min_x, max_y))
    best = min(best, segment_segment_distance_sq(ax, ay, bx, by, min_x, max_y, min_x, min_y))
    return best


def forward_kinematics_with_points(angles, lengths):
    phi = 0.0
    x = 0.0
    y = 0.0
    points = [(0.0, 0.0)]
    prefix = []
    for theta, length in zip(angles, lengths):
        phi += theta
        prefix.append(phi)
        x += length * math.cos(phi)
        y += length * math.sin(phi)
        points.append((x, y))
    return x, y, points, prefix


def polyline_square_violation(points, obstacle, clearance=OBSTACLE_CLEARANCE):
    cx, cy, half = obstacle
    min_x = cx - half
    max_x = cx + half
    min_y = cy - half
    max_y = cy + half
    best_d2 = float("inf")
    for (ax, ay), (bx, by) in zip(points, points[1:]):
        d2 = segment_aabb_distance_sq(ax, ay, bx, by, min_x, min_y, max_x, max_y)
        if d2 < best_d2:
            best_d2 = d2
        if best_d2 <= 0.0:
            break
    return clearance - math.sqrt(best_d2)


def polyline_circle_violation(points, cx, cy, radius, clearance=0.0):
    rr = radius + clearance
    best_d2 = float("inf")
    for (ax, ay), (bx, by) in zip(points, points[1:]):
        d2 = point_segment_distance_sq(cx, cy, ax, ay, bx, by)
        if d2 < best_d2:
            best_d2 = d2
        if best_d2 <= rr * rr:
            break
    return rr - math.sqrt(best_d2)


def global_angle_lower(index):
    return -1.0471975511965977 if index == 0 else -2.6179938779914944


def global_angle_upper(index):
    return 2.6179938779914944


def positioning_cost_from_pose(x, y, target_x, target_y):
    return math.hypot(x - target_x, y - target_y)


def build_positioning_constraints(angles, lengths, x, y, points, obstacles,
                                  constraint_mode, min_theta,
                                  base_length, stretch_factor):
    out = []
    for theta in angles:
        out.append(abs(theta) - min_theta)
    if lengths:
        lo = base_length / stretch_factor
        hi = base_length * stretch_factor
        for L in lengths:
            if L < lo:
                out.append(lo - L)
            elif L > hi:
                out.append(L - hi)
            else:
                out.append(0.0)
    if uses_obstacles(constraint_mode, obstacles):
        for obs in obstacles:
            out.append(polyline_square_violation(points, obs, OBSTACLE_CLEARANCE))
    if uses_nonlinear(constraint_mode):
        radius_inner = math.sqrt(7.7)
        out.append(polyline_circle_violation(points, 8.0, -3.0, radius_inner, 0.0))
        c0, c1 = nonlinear_constraints_for_pose(x, y)
        out.append(c0)
        out.append(c1)
    return tuple(float(v) for v in out)


def evaluate_positioning_configuration(angles, lengths, target_x, target_y, min_theta,
                                       obstacles, constraint_mode,
                                       base_length, stretch_factor):
    x, y, points, _ = forward_kinematics_with_points(angles, lengths)
    value = positioning_cost_from_pose(x, y, target_x, target_y)
    constraints = build_positioning_constraints(
        angles, lengths, x, y, points, obstacles,
        constraint_mode, min_theta,
        base_length, stretch_factor
    )
    return x, y, value, constraints


def params_to_angles_lengths(params, n_seg, var_len, base_length):
    angles = [float(params[f"theta_{i}"]) for i in range(n_seg)]
    if var_len:
        lengths = [float(params[f"L_{i}"]) for i in range(n_seg)]
    else:
        lengths = [base_length] * n_seg
    return angles, lengths


def build_angles_lengths(trial, n_seg, var_len, base_length, stretch_factor):
    angles = []
    for i in range(n_seg):
        low = global_angle_lower(i)
        high = global_angle_upper(i)
        angles.append(trial.suggest_float(f"theta_{i}", low, high))
    if var_len:
        lo = base_length / stretch_factor
        hi = base_length * stretch_factor
        lengths = [trial.suggest_float(f"L_{i}", lo, hi) for i in range(n_seg)]
    else:
        lengths = [base_length] * n_seg
    return angles, lengths


def finalize_optuna_result(study, n_seg, var_len, base_length, evaluator):
    best_trial = choose_best_optuna_trial(study)
    angles, lengths = params_to_angles_lengths(best_trial.params, n_seg, var_len, base_length)
    best_x, best_y, best_f, _ = evaluator(angles, lengths)
    iterations = len(study.trials)
    recent_cnt = max(1, iterations // 10)
    recent_vals = [
        float(t.value) for t in study.trials[-recent_cnt:]
        if t.state == TrialState.COMPLETE and t.value is not None
    ]
    achieved_eps = statistics.stdev(recent_vals) if len(recent_vals) > 1 else 0.0
    return {
        "BEST_F": best_f,
        "BEST_X": best_x,
        "BEST_Y": best_y,
        "ITERATIONS": iterations,
        "EPS": achieved_eps,
        "ANGLES": angles,
        "LENGTHS": lengths,
    }


def create_optuna_sampler(max_iter, dimension, hard_constraints_active, optuna_seed):
    if optuna_seed is not None:
        optuna_seed = int(optuna_seed)
    if hard_constraints_active:
        def constraints_func(frozen_trial):
            return frozen_trial.user_attrs["constraint"]
        return TPESampler(
            seed=optuna_seed,
            n_startup_trials=min(max_iter, max(12, 2 * dimension)),
            multivariate=False,
            constraints_func=constraints_func,
        )
    return CmaEsSampler(
        seed=optuna_seed,
        n_startup_trials=(dimension - 1) ** 2 + 3,
        consider_pruned_trials=False,
    )


def run_optuna(n_seg, var_len, min_theta, tx, ty, max_iter, base_length,
               stretch_factor, obstacles, constraint_mode, optuna_seed=None):
    warnings.filterwarnings("ignore")
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    active_obs = list(obstacles) if uses_obstacles(constraint_mode, obstacles) else []
    hard_constraints = bool(active_obs) or uses_nonlinear(constraint_mode)
    dimension = n_seg * (2 if var_len else 1)
    sampler = create_optuna_sampler(max_iter, dimension, hard_constraints, optuna_seed)

    def evaluator(angles, lengths):
        return evaluate_positioning_configuration(
            angles, lengths, tx, ty, min_theta,
            active_obs, constraint_mode,
            base_length, stretch_factor
        )

    def objective(trial):
        angles, lengths = build_angles_lengths(trial, n_seg, var_len, base_length, stretch_factor)
        _, _, value, constraints = evaluator(angles, lengths)
        trial.set_user_attr("constraint", constraints)
        return value

    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=optuna.pruners.NopPruner())
    start_time = time.perf_counter()
    study.optimize(objective, n_trials=max_iter, show_progress_bar=False, gc_after_trial=False)
    elapsed_micros = (time.perf_counter() - start_time) * 1e6
    result = finalize_optuna_result(study, n_seg, var_len, base_length, evaluator)
    result["TIME"] = elapsed_micros
    return result


class ManipIOptProblem(Problem):
    def __init__(self, n_seg, var_len, min_theta, tx, ty, base_length,
                 stretch_factor, obstacles, constraint_mode):
        super().__init__()
        self.n_seg = n_seg
        self.var_len = var_len
        self.min_theta = min_theta
        self.tx = tx
        self.ty = ty
        self.base_length = base_length
        self.stretch_factor = stretch_factor
        self.obstacles = list(obstacles) if uses_obstacles(constraint_mode, obstacles) else []
        self.constraint_mode = constraint_mode

        self.number_of_objectives = 1
        self.number_of_discrete_variables = 0
        self.number_of_float_variables = n_seg * (2 if var_len else 1)
        self.dimension = self.number_of_float_variables

        c_cnt = n_seg
        if var_len:
            c_cnt += n_seg
        if uses_nonlinear(constraint_mode):
            c_cnt += 3
        if uses_obstacles(constraint_mode, obstacles):
            c_cnt += len(self.obstacles)
        self.number_of_constraints = c_cnt

        names = []
        lower = []
        upper = []
        for i in range(n_seg):
            names.append(f"theta_{i}")
            lower.append(global_angle_lower(i))
            upper.append(global_angle_upper(i))
        if var_len:
            lo = base_length / stretch_factor
            hi = base_length * stretch_factor
            for i in range(n_seg):
                names.append(f"L_{i}")
                lower.append(lo)
                upper.append(hi)
        self.float_variable_names = np.array(names, dtype=object)
        self.lower_bound_of_float_variables = np.array(lower, dtype=float)
        self.upper_bound_of_float_variables = np.array(upper, dtype=float)

    def point_to_angles_lengths(self, point):
        floats = point.float_variables
        angles = [float(floats[i]) for i in range(self.n_seg)]
        if self.var_len:
            lengths = [float(floats[self.n_seg + i]) for i in range(self.n_seg)]
        else:
            lengths = [self.base_length] * self.n_seg
        return angles, lengths

    def evaluate_with_constraints(self, angles, lengths):
        x, y, points, _ = forward_kinematics_with_points(angles, lengths)
        value = positioning_cost_from_pose(x, y, self.tx, self.ty)
        constraints = build_positioning_constraints(
            angles, lengths, x, y, points,
            self.obstacles, self.constraint_mode,
            self.min_theta,
            self.base_length, self.stretch_factor
        )
        return x, y, value, constraints

    def calculate(self, point, function_value):
        angles, lengths = self.point_to_angles_lengths(point)
        _, _, obj, cons = self.evaluate_with_constraints(angles, lengths)
        if function_value.type == FunctionType.CONSTRAINT:
            function_value.value = float(cons[int(function_value.functionID)])
        else:
            function_value.value = float(obj)
        return function_value

    def calculateAllFunction(self, point, function_values):
        angles, lengths = self.point_to_angles_lengths(point)
        _, _, obj, cons = self.evaluate_with_constraints(angles, lengths)
        for fv in function_values:
            if fv.type == FunctionType.CONSTRAINT:
                fv.value = float(cons[int(fv.functionID)])
            else:
                fv.value = float(obj)
        return function_values


def run_iopt(n_seg, var_len, min_theta, tx, ty, levels, max_iter, r_param, eps,
             adaptive, base_length, stretch_factor, obstacles, constraint_mode):
    problem = ManipIOptProblem(
        n_seg, var_len, min_theta, tx, ty, base_length,
        stretch_factor, obstacles, constraint_mode
    )
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = StringIO()
    sys.stderr = StringIO()
    try:
        eps_r = max(eps, 1e-3) if problem.number_of_constraints > 0 else 0.0
        study = iOpt.create_study()
        params = iOpt.SolverParameters(
            eps=eps,
            r=r_param,
            iters_limit=max_iter,
            evolvent_density=levels,
            refine_solution=adaptive,
            eps_r=eps_r,
        )
        study.optimize(
            objective=problem,
            solver_parameters=params,
            type_of_painter="none",
            console_mode="off",
        )
        best_params = study.best_float_params_()
        angles = [float(best_params[i]) for i in range(n_seg)]
        if var_len:
            lengths = [float(best_params[n_seg + i]) for i in range(n_seg)]
        else:
            lengths = [base_length] * n_seg
        best_x, best_y, best_f, _ = problem.evaluate_with_constraints(angles, lengths)
        solution = getattr(study, "solution", None)
        if solution is not None:
            achieved_eps = float(getattr(solution, "solution_accuracy", 0.0))
            iterations = int(getattr(solution, "number_of_global_trials", 0))
            elapsed_micros = float(getattr(solution, "solving_time", 0.0)) * 1e6
        else:
            achieved_eps = 0.0
            iterations = 0
            elapsed_micros = 0.0
        return {
            "BEST_F": best_f,
            "BEST_X": best_x,
            "BEST_Y": best_y,
            "ITERATIONS": iterations,
            "EPS": achieved_eps,
            "TIME": elapsed_micros,
            "ANGLES": angles,
            "LENGTHS": lengths,
        }
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


def main():
    pass


if __name__ == "__main__":
    main()
