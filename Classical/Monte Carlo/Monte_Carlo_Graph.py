import numpy as np
import matplotlib.pyplot as plt

def simulate_pi_over_N(N_list, n_runs=200, seed=0, chunk=5000):
    rng = np.random.default_rng(seed)
    N = np.array(sorted(set(int(x) for x in N_list)))
    m = len(N)
    inside = np.zeros(n_runs, dtype=np.int64)
    total  = np.zeros(n_runs, dtype=np.int64)
    pi_hat = np.empty((n_runs, m), dtype=float)
    cur = 0
    for j, target in enumerate(N):
        remaining = target - cur
        while remaining > 0:
            n = min(chunk, remaining)
            x = rng.uniform(-1.0, 1.0, size=(n_runs, n))
            y = rng.uniform(-1.0, 1.0, size=(n_runs, n))
            inside += ((x * x + y * y) <= 1.0).sum(axis=1)
            total  += n
            remaining -= n
        pi_hat[:, j] = 4.0 * (inside / total)
        cur = target
    return N, pi_hat


def plot_value_with_errorbars(N, pi_hat, ci=(0.1, 0.9)):
    pi_true = np.pi
    lo_q, hi_q = np.quantile(pi_hat, ci, axis=0)
    center = pi_hat.mean(axis=0)
    err_low  = center - lo_q
    err_high = hi_q - center

    # 1/sqrt(N) convergence envelope anchored to pi_true
    half_width_at_first = max(err_low[0], err_high[0])
    C = half_width_at_first * np.sqrt(N[0])
    N_smooth = np.logspace(np.log10(N[0]), np.log10(N[-1]), 300)
    upper_bound = pi_true + C / np.sqrt(N_smooth)
    lower_bound = pi_true - C / np.sqrt(N_smooth)

    fig, ax = plt.subplots(figsize=(8.5, 5))

    # Convergence envelope
    ax.plot(N_smooth, upper_bound, color="green", linewidth=1.5,
            linestyle="--", label=r"$1/\sqrt{N}$ scaling")
    ax.plot(N_smooth, lower_bound, color="green", linewidth=1.5,
            linestyle="--")

    # True value
    ax.axhline(pi_true, color="red", linewidth=2, linestyle="--",
               label="True π")

    # Error bars
    ax.errorbar(N, center, yerr=[err_low, err_high],
                fmt="o", capsize=5, color="steelblue",
                ecolor="steelblue", elinewidth=1.2, markersize=5,
                label="MC estimate", zorder=3)

    # Log x-axis: only decade ticks, no minor ticks
    ax.set_xscale("log")
    decade_ticks = [t for t in [10, 100, 1000, 10000] if N[0] <= t <= N[-1]]
    ax.set_xticks(decade_ticks)
    ax.set_xticklabels([f"{t:,}" for t in decade_ticks])
    ax.xaxis.set_minor_locator(plt.NullLocator())

    # Clean y-axis: ticks every 0.5
    max_dev = max(err_low.max(), err_high.max()) * 1.4
    ax.set_ylim(pi_true - max_dev, pi_true + max_dev)
    ylo = np.floor((pi_true - max_dev) / 0.5) * 0.5
    yhi = np.ceil( (pi_true + max_dev) / 0.5) * 0.5
    ax.set_yticks(np.arange(ylo, yhi + 1e-9, 0.5))
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))

    ax.set_xlabel("N (sample size)", fontsize=12)
    ax.set_ylabel("Estimated value", fontsize=12)
    ax.set_title(r"Monte Carlo $\pi$ Estimates ($1/\sqrt{N}$ Scaling)", fontsize=13)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(loc="upper right", fontsize=10)

    plt.tight_layout()
    plt.show()
    return fig, ax


if __name__ == "__main__":
    N_list = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    N, pi_hat = simulate_pi_over_N(N_list, n_runs=300, seed=1, chunk=5000)
    plot_value_with_errorbars(N, pi_hat, ci=(0.1, 0.9))
