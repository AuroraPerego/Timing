import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt

# a and b are two opposite vertices of the parallepiped
def plot_cube(a, b, ax):
    x, y, z = 0, 2, 1 # to plot y vertical

    vertices = [
        # XZ
        [(a[x], a[y], a[z]), (b[x], a[y], a[z]), (b[x], a[y], b[z]), (a[x], a[y], b[z])],
        [(a[x], b[y], a[z]), (b[x], b[y], a[z]), (b[x], b[y], b[z]), (a[x], b[y], b[z])],

        # YZ
        [(a[x], a[y], a[z]), (a[x], b[y], a[z]), (a[x], b[y], b[z]), (a[x], a[y], b[z])],
        [(b[x], a[y], a[z]), (b[x], b[y], a[z]), (b[x], b[y], b[z]), (b[x], a[y], b[z])],

        # XY
        [(a[x], a[y], a[z]), (b[x], a[y], a[z]), (b[x], b[y], a[z]), (a[x], b[y], a[z])],
        [(a[x], a[y], b[z]), (b[x], a[y], b[z]), (b[x], b[y], b[z]), (a[x], b[y], b[z])],
    ]

    #fig = plt.figure()

    #ax = fig.add_subplot(111, projection='3d')

    ax.plot([a[x], b[x]], [a[y], b[y]], [a[z], b[z]], 'cyan', alpha=.2)
    ax.add_collection3d(Poly3DCollection(
        vertices, facecolors='cyan', linewidths=1, edgecolors='k', alpha=.2))

    #set_axes_equal(ax)
    #plt.show()

def hist(X, weights=None, bins=30, title='title', xlabel='time (ns)', ylabel='Counts'):
    plt.figure(dpi=100)
    if weights == None:
        plt.hist(ak.flatten(X), bins=bins, color='dodgerblue')
    else:
        plt.hist(ak.flatten(X), bins=bins, color='dodgerblue', weights=ak.flatten(weights))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.show()
    
def scatter(X, Y, s=10, title='title', xlabel='time (ns)', ylabel='z (cm)'):
    plt.figure(dpi=100)
    try:
        plt.scatter(ak.flatten(X), ak.flatten(Y), s=ak.flatten(s), color='dodgerblue')
    except:
        plt.scatter(X, Y, s=s, color='dodgerblue')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.show()
    
def subplot(axs, X, bins=30, title='title', xlabel='time (ns)', ylabel='Counts'):
    #bins=ak.sum(X>-98)
    axs.hist(ak.flatten(X), bins=bins, color='dodgerblue')
    axs.set_title(title)
    axs.grid()

def multiplot(nplots, *arg, title='SimTracksters from CaloParticle', xlabel='time (ns)'):
    fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(18,4))
    fig.suptitle(title, size=18, y=1.05)
    i=0
    while(i+1 < len(arg)):
        subplot(axs[i], arg[i])
        axs[i].set_title(arg[i+1])

    for ax in axs.flat:
        ax.set_xlabel(xlabel, fontsize = 16.0)    
        ax.set_ylabel('Counts', fontsize = 16.0)
    
def nphist(X, weights=None, bins=30, title='title', xlabel='time (ns)', ylabel='Counts'):
    plt.figure(dpi=100)
    try:
        plt.hist(np.array(X).flatten(), bins=bins, color='dodgerblue')
    except:
        plt.hist(np.array(X), bins=bins, color='dodgerblue')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.show()    