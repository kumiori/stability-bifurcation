import matplotlib.pyplot as plt
from dolfin import plot, dot
from ufl import ln
import os
from dolfin import MPI
import numpy as np

def plotMatrix(J, outdir='data', name='matrix'):
    c = J.size(1)
    r = J.size(0)
    m = np.inf
    M = -np.inf
    data  = [ [0] * c for i in range(r) ]
    for rr in range(J.size(0)):
        m = min(J.getrow(rr)[1]) if min(J.getrow(rr)[1]) < m else m
        M = max(J.getrow(rr)[1]) if max(J.getrow(rr)[1]) > M else M
        for cc in J.getrow(rr)[0]:
            idx = np.where(J.getrow(rr)[0]==cc)[0][0]
            data[rr][cc]=J.getrow(rr)[1][idx]
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_aspect('equal')
    plt.imshow(data, interpolation='nearest', cmap=plt.cm.RdGy,
        vmin=-max(abs(m), abs(M)), vmax=max(abs(m), abs(M)))
    plt.colorbar()
    plt.savefig(os.path.join(outdir, name))
    pass    

def plot_global_data(time_data_pd, load, outdir):
    if MPI.size(MPI.comm_world):
        plt.figure()
        plt1 = time_data_pd.plot(
            x="load",
            y=["iterations"],
            marker=".",
            logy=True,
            logx=False,
        )
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(os.path.join(outdir, "plot_err.png"))
        plt.figure()
        plt2 = time_data_pd.plot(
            x="load",
            y=[
                "dissipated_energy",
                "elastic_energy"
            ],
        )
        plt.savefig(os.path.join(outdir, "plot_energies.png"))
        #plt.figure()
        #plt3 = time_data_pd.plot(x="load", y=["force"], marker=".")
        #plt.savefig(os.path.join(outdir, "plot_force.png"))
        plt.close("all")

def make_figures(time_data_pd, u, alpha, load, outdir):
    if MPI.size(MPI.comm_world):
        plt.figure(1)
        pltu = plt.colorbar(plot(u, mode="displacement", title=u.name()))
        plt.savefig(os.path.join(outdir, "plot_u_{:3.4f}.png".format(load)))
        plt.figure(2)
        pltalpha = plt.colorbar(plot(alpha, title=alpha.name()))
        plt.subplots_adjust(hspace=0.8)
        plt.savefig(os.path.join(outdir, "plot_alpha_{:3.4f}.png".format(load)))
        plt.close("all")

def plot_eigenmodes(eigendata, alpha, load, outdir):
    nmodes = len(eigendata)
    fig = plt.figure(figsize=((nmodes+1)*3, 3), dpi=80, facecolor='w', edgecolor='k')
    fig.suptitle('Load {:3f}'.format(load), fontsize=16)
    plt.subplot(2, nmodes+1, 1)
    plt.title('$\\alpha$ (max = {:2.2f})'.format(max(alpha.vector()[:])))
    plt.set_cmap('coolwarm')
    plt.axis('off')
    plot(alpha, vmin=0., vmax=1.)

    plt.set_cmap('hot')

    for i,mode in enumerate(eigendata):
        plt.subplot(2, nmodes+1, i+2)
        plt.axis('off')
        plot(mode['beta_n'])
        # import pdb; pdb.set_trace()
        plt.title('mode {}\n$\\lambda_{{{}}}={:.1e},$\n$h^*={:.3f}, \\Delta E={:.2%}$%'.format(
            i, i, mode['lambda_n'], mode['hstar'], mode['en_diff']))
        # print('plot mode {}'.format(i))
        # plt.tight_layout(h_pad=0.0, pad=1.5)
        # plt.savefig(os.path.join(outdir, "modes-{:3.4f}.png".format(load)))

    for i,mode in enumerate(eigendata):
        plt.subplot(2, nmodes+1, nmodes+2+1+i)
        plt.axis('off')
        bounds = mode['interval']
        # import pdb; pdb.set_trace()
        if bounds[0] == bounds[1] == 0:
            plt.plot(bounds[0], mode['energy'])
        else:
            hs = np.linspace(bounds[0], bounds[1], 100)
            z = np.polyfit(np.linspace(bounds[0], bounds[1],
                len(mode['energy'])), mode['energy'], mode['order'])
            p = np.poly1d(z)
            plt.plot(hs, p(hs))
            plt.plot(np.linspace(bounds[0], bounds[1],
                len(mode['energy'])), mode['energy'], marker='o')
            plt.axvline(mode['hstar'])
            plt.axvline(0, lw=.5, c='k')
        # plt.title('{}'.format(i))
        # plt.tight_layout(h_pad=0.0, pad=1.5)
    # plt.legend()
    plt.savefig(os.path.join(outdir, "modes-{:3.4f}.png".format(load)))
    plt.close(fig)
    plt.clf()
    # plt.colorbar(plot(mode['v_n'], mode='magnitude'))
    # import pdb; pdb.set_trace()

    plt.colorbar(plot(dot(eigendata[0]['v_n'],eigendata[0]['v_n'])**(.5)))
    plt.title('mode 0')
    plt.savefig(os.path.join(outdir, "modes-vn-{:3.4f}.png".format(load)))
    plt.close()

def plot_spectrum(params, outdir, data, tc=None):
    # E0 = params['material']['E']
    # w1 = params['material']['sigma_D0']**2/E0

    fig = plt.figure(dpi=80, facecolor='w', edgecolor='k')
    for i,d in enumerate(data['eigs']):
        if d is not (None or np.inf or np.nan):
            lend = len(d) if isinstance(d, list) or isinstance(d, np.ndarray) else 1
            plt.scatter([(data['load'].values)[i]]*lend, d,
                       c=np.where(np.array(d)<-1e-8, 'C1', 'C2'))

    # plt.ylim(-6e-4, 3e-4)
    plt.axhline(0, c='k', lw=1.)
    plt.xlabel('$t$')
    plt.ylabel('$\\lambda_m$')
    if tc: plt.axvline(tc, lw=.5, c='k')
    ax1 = plt.gca()
    ax2 = plt.twinx()

    ax2.plot(data['load'].values,
        data['alpha_max'].values)
    ax2.axhline(1., c='k')
    ax2.set_ylabel('$max \\alpha$')
    plt.savefig(os.path.join(outdir, "spectrum.pdf"))

    return fig, ax1, ax2
