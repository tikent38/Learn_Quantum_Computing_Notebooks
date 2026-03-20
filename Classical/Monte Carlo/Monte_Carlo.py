import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

def animate_pi_cumulative(points_per_trial=2000, n_trials=45, seed=1, interval_ms=200, save=None):
    rng = np.random.default_rng(seed)
    pi_true = np.pi

    total_points = 0
    total_inside = 0
    cumulative_estimates = np.full(n_trials, np.nan, dtype=float)
    batch_estimates = np.full(n_trials, np.nan, dtype=float)

    trials = np.arange(1, n_trials + 1)

    xs_in, ys_in = [], []
    xs_out, ys_out = [], []

    fig = plt.figure(figsize=(13, 6), constrained_layout=True)
    # 3-column grid: scatter gets col 0-1 (wide), right panels share col 2
    gs = fig.add_gridspec(2, 3, width_ratios=[0.05, 1.0, 1.1], height_ratios=[1.0, 1.0])
    ax_scatter = fig.add_subplot(gs[:, 1])   # centre column, both rows
    ax_line    = fig.add_subplot(gs[0, 2])
    ax_hist    = fig.add_subplot(gs[1, 2])

    theta = np.linspace(0, 2 * np.pi, 800)
    circle_line, = ax_scatter.plot(np.cos(theta), np.sin(theta), linewidth=2)

    scat_in  = ax_scatter.scatter([], [], s=6, alpha=0.9, label="inside")
    scat_out = ax_scatter.scatter([], [], s=6, alpha=0.9, label="outside")

    ax_scatter.set_title(
        "Cumulative drops into the square\n"
        f"Batch size = {points_per_trial}, Trials = {n_trials}"
    )
    ax_scatter.set_xlim(-1, 1)
    ax_scatter.set_ylim(-1, 1)
    ax_scatter.set_aspect("equal", adjustable="box")
    ax_scatter.grid(True, alpha=0.25)
    ax_scatter.legend(loc="upper right")

    ax_line.set_title("Cumulative π estimate vs trial")
    ax_line.set_xlabel("# of trials")
    ax_line.set_ylabel("π estimate")
    ax_line.grid(True, alpha=0.25)
    ax_line.set_xlim(1, n_trials)
    ax_line.set_ylim(2.7, 3.6)
    ax_line.axhline(pi_true, color="red", linewidth=2, label="Actual π")

    (line_est,) = ax_line.plot([], [], marker="o", linestyle="-", markersize=4, label="Cumulative estimate")
    ax_line.legend()

    def draw_hist(vals, current_cum):
        ax_hist.cla()
        ax_hist.hist(vals, bins=12)
        ax_hist.axvline(pi_true, color="red", linewidth=2, label="Actual π")
        ax_hist.axvline(np.mean(vals), linewidth=2, linestyle="--", label="Mean of batches")
        ax_hist.set_title("Histogram of per-batch π estimates")
        ax_hist.set_xlabel("Approximation to π")
        ax_hist.set_ylabel("Counts")
        ax_hist.grid(True, alpha=0.25)
        ax_hist.legend()

    def drop_points(n):
        x = rng.uniform(-1.0, 1.0, size=n)
        y = rng.uniform(-1.0, 1.0, size=n)
        inside = (x * x + y * y) <= 1.0
        return x, y, inside

    def init():
        scat_in.set_offsets(np.empty((0, 2)))
        scat_out.set_offsets(np.empty((0, 2)))
        line_est.set_data([], [])
        ax_hist.cla()
        ax_hist.set_title("Histogram of per-batch π estimates")
        ax_hist.set_xlabel("Approximation to π")
        ax_hist.set_ylabel("Counts")
        ax_hist.grid(True, alpha=0.25)
        return scat_in, scat_out, line_est, circle_line

    def update(frame_idx):
        nonlocal total_points, total_inside

        x, y, inside = drop_points(points_per_trial)

        inside_batch = int(inside.sum())
        pi_hat_batch = 4.0 * (inside_batch / points_per_trial)
        batch_estimates[frame_idx] = pi_hat_batch

        total_points += points_per_trial
        total_inside += inside_batch
        pi_hat_cum = 4.0 * (total_inside / total_points)
        error_pct = abs(pi_hat_cum - pi_true) / pi_true * 100.0

        ax_scatter.set_title(
            "Cumulative drops into the square\n"
            f"Batch size = {points_per_trial}, Trials = {n_trials} | Error: {error_pct:.3f}%"
        )

        cumulative_estimates[frame_idx] = pi_hat_cum

        xs_in.append(x[inside]);   ys_in.append(y[inside])
        xs_out.append(x[~inside]); ys_out.append(y[~inside])

        xin  = np.concatenate(xs_in)  if xs_in  else np.array([])
        yin  = np.concatenate(ys_in)  if ys_in  else np.array([])
        xout = np.concatenate(xs_out) if xs_out else np.array([])
        yout = np.concatenate(ys_out) if ys_out else np.array([])

        scat_in.set_offsets(np.column_stack([xin, yin])   if xin.size  else np.empty((0, 2)))
        scat_out.set_offsets(np.column_stack([xout, yout]) if xout.size else np.empty((0, 2)))

        valid_cum = ~np.isnan(cumulative_estimates)
        line_est.set_data(trials[valid_cum], cumulative_estimates[valid_cum])

        ax_line.set_title(
            "Cumulative π estimate vs trial\n"
            f"After {total_points:,} points: π ≈ {pi_hat_cum:.6f}"
        )

        valid_batch = ~np.isnan(batch_estimates)
        draw_hist(batch_estimates[valid_batch], current_cum=pi_hat_cum)

        return scat_in, scat_out, line_est, circle_line

    anim = FuncAnimation(
        fig, update, frames=n_trials, init_func=init,
        interval=interval_ms, blit=False, repeat=False
    )

    if save:
        fps = max(1, int(1000 / interval_ms))
        if save.lower().endswith(".gif"):
            anim.save(save, writer=PillowWriter(fps=fps), dpi=150)
        else:
            anim.save(save, writer="ffmpeg", fps=fps, dpi=200)

    plt.show()
    return anim


if __name__ == "__main__":
    animate_pi_cumulative(points_per_trial=2000, n_trials=100, seed=1, interval_ms=200)
