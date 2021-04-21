import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy.integrate import solve_ivp
import torch
from tqdm import tqdm

# Should be None, 'pgf', or 'png'
MODE = 'png'

if MODE == 'pgf':
    # From https://timodenk.com/blog/exporting-matplotlib-plots-to-latex/
    # and https://tex.stackexchange.com/questions/410173/
    matplotlib.use('pgf')
    matplotlib.rcParams.update({
        'pgf.texsystem': 'pdflatex',
        # 'pgf.preamble': r'\DeclareUnicodeCharacter{2212}{-}',
        'font.family': 'serif',
        'text.usetex': True,
        # 'pgf.rcfonts': False
    })

# From https://jwalton.info/Matplotlib-latex-PGF/
def set_size(width_pt=430.00462, fraction=1, subplots=(1, 1), golden_ratio=1):
    """Set figure dimensions to sit nicely in our document.

    Parameters
    ----------
    width_pt: float
            Document width in points
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)

def plt_save_or_show(name):
    if MODE == None:
        plt.show()
    elif MODE == 'pgf':
        plt.savefig(name + '.pgf', bbox_inches='tight')
    elif MODE == 'png':
        plt.savefig(name + '.png', bbox_inches='tight', dpi=500)
    else:
        raise ValueError('Unrecognized MODE value')

def build_difs_matrix(xs):
    """Given an (n x 2) matrix of (x, y) ordered pairs, return an
    (n x n x 2) matrix M of differences so that m_{j,k} = x_j - x_k."""
    n = len(xs)
    difs = np.zeros((n, n, 2))
    for row_id in range(n):
        ident = -np.eye(n)
        ident[:, row_id] += 1
        difs[row_id, :, 0] = ident @ xs[:, 0]
        difs[row_id, :, 1] = ident @ xs[:, 1]
    return difs

def make_rhs1(a, b, c, p):
    def rhs1(t, state):
        """Compute the right-hand side for the 1st order
        system. The state vector has the following format:

        [pred_x,  pred_y,
         prey1_x, prey1_y,
         prey2_x, prey2_y,
         .        .
         .        .
         .        .
         preyn_x, preyn_y]."""

        # Get the number of prey
        n = (len(state) // 2) - 1
        # Extract the prey positions
        xs = state[2:].reshape(n, 2)

        # Extract the predator position
        z = state[:2]
        # Compute the differences between the prey
        # positions and the predator position
        xz_difs = xs - np.expand_dims(z, 0)
        xz_len2s = np.sum(xz_difs**2, axis=1)#**(p/2)
        lens = xz_len2s**(p/2)
        pred_dxdt = c / n * np.sum(xz_difs / np.expand_dims(lens, 1), axis=0)

        # Compute the differences between the prey positions
        dif_mat = build_difs_matrix(xs)
        # Compute the squared lengths between prey positions
        len2_mat = np.sum(dif_mat**2, axis=2)
        # Set the diagonal to 1 to avoid dividing by zero
        len2_mat[np.diag_indices(n)] = 1
        # Compute the prey->prey effects
        m = 1 / n * (dif_mat / np.expand_dims(len2_mat, 2) - a * dif_mat)
        prey_dxdt = np.sum(m, axis=1)
        # Compute the predator->prey effects
        prey_dxdt += b * xz_difs / np.expand_dims(xz_len2s, 1)

        # Flatten the derivatives the way solve_ivp wants
        dxdt = np.zeros(state.shape)
        dxdt[:2] = pred_dxdt.flatten()
        dxdt[2:] = prey_dxdt.flatten()
        return dxdt
    return rhs1

def solve_ivp_tqdm(fun, t_span, y0, t_eval, **kwargs):
    """Wrapper for scipy.integrate.solve_ivp
    that uses tqdm to monitor progress."""
    sol = None
    n = len(t_eval) - 1
    for i in tqdm(range(n)):
        t0 = t_eval[i]
        tf = t_eval[i + 1]
        s = solve_ivp(fun, [t0, tf], y0, t_eval=[t0, tf], **kwargs)
        y0 = s.y[:, 1]
        if sol is None:
            sol = s
        else:
            sol.t = np.append(sol.t, s.t[1])
            sol.y = np.hstack((sol.y, s.y[:, -1:]))
    return sol

def bake_regimes_figure():
    """Bake the simulation for figure 2 from the paper."""

    d = {'n': 400, # 400,
         'a': 1,
         'b': 0.2,
         'p': 3,
         'cs': np.asarray([0.15, 0.4, 0.8, 1.5, 2.5]),
         'times': np.asarray([[0, 45, 55, 60, 70, 80, 100],
                              [0, 0.35, 0.85, 2.35, 4.85, 9.85, 21.05],
                              [0, 4, 123.8, 125.8, 127.8, 129.8, 131.8],
                              [0, 0.6, 1.2, 1.8, 2.4, 3, 3.6],
                              [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2]]),
         'states': []}

    # Simulate each regime
    for row_id, (c, ts) in enumerate(zip(d['cs'], d['times'])):
        print('Simulating regime %d/%d' % (row_id + 1, len(d['cs'])))
        rhs1 = make_rhs1(d['a'], d['b'], c, d['p'])
        # Initialize the predator position at (0.5, 0.5),
        # and initialize the prey positions randomly
        # inside the square [-1, 1] x [-1, 1]
        state0 = np.random.uniform(-1, 1, size=2 * (d['n'] + 1))
        state0[:2] = 0
        # Solve the equations
        s = solve_ivp_tqdm(rhs1,
                           t_span=(ts[0], ts[-1]),
                           y0=state0,
                           # method='RK23',
                           t_eval=ts)
        # Save the results to the dictionary
        d['states'].append(s.y)

    # Save the dictionary to the disk
    pickle.dump(d, open('regimes.pickle', 'wb'))

def compute_xy_lims(center, width):
    """Return (left, right) and (bottom, top) tuples
    specifying a square viewport of sidelength `width`
    centered at `center`."""
    x, y = center
    w2 = width / 2
    return (x - w2, x + w2), (y - w2, y + w2)

def set_ax_lims(ax, xs, ys, width=2.5):
    """Set the viewport to be a square of sidelength
    `width` centered at the mean of the data."""
    center = (xs.mean(), ys.mean())
    xlim, ylim = compute_xy_lims(center, width)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

SMALL_FONT_SIZE = 6

def plot_regime(row_id, axes, c, times, states):
    for col_id, (ax, t) in enumerate(zip(axes, times)):
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)
        # ax.axis('off')
        # ax.tick_params(axis=u'both', which=u'both',length=0)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        # Turn off the black border
        for key in ['left', 'right', 'bottom', 'top']:
            ax.spines[key].set_visible(False)
        ax.set_title(r'$t = %s$' % str(t), size=SMALL_FONT_SIZE,
                     y=0.9)
        if col_id == 0:
            ax.annotate(r'(%d) $c = %s$' % (row_id + 1, str(c)),
                        # xy=(2.2, 0.5),
                        xy=(0, 0),
                        xytext=(-ax.yaxis.labelpad - 30, 0),
                        xycoords=ax.yaxis.label,
                        textcoords='offset points',
                        size=SMALL_FONT_SIZE,
                        ha='left', va='center')

        scatter_kwargs = {'edgecolors': None,
                          'linewidths': 0}
        # Plot the prey
        prey_xs = states[2::2, col_id]
        prey_ys = states[3::2, col_id]
        ax.scatter(prey_xs, prey_ys, color='black', s=0.8, **scatter_kwargs)
        # Plot the predator
        pred_x, pred_y = states[:2, col_id]
        ax.scatter([pred_x], [pred_y], color='red', s=1.5, **scatter_kwargs)
        set_ax_lims(ax, prey_xs, prey_ys)

def make_regimes_figure():
    """Replicate figure 2 from the paper."""

    d = pickle.load(open('regimes.pickle', 'rb'))
    fig, ax_matrix = plt.subplots(*d['times'].shape,
                                  figsize=set_size(fraction=1,
                                                   subplots=d['times'].shape))

    for row_id, (axes, c, ts) in enumerate(zip(ax_matrix, d['cs'], d['times'])):
        print('Plotting regime %d/%d' % (row_id + 1, len(d['cs'])))
        plot_regime(row_id, axes, c, ts, d['states'][row_id])

    plt.subplots_adjust(hspace=0.3)
    plt.subplots_adjust(wspace=0)
    # plt.tight_layout()
    plt_save_or_show('regimes')
