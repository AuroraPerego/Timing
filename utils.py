import numpy as np
from wpca import WPCA, EMPCA
import matplotlib.pyplot as plt
import awkward as ak

C = 29.9792458 #cm/ns

# compute barycenter time: propagate back all LCs times to 0,0,0 and recompute tk time, than go to barycenter position
def barTime(bx, by, bz, vx, vy, vz, ve, vt, vm=None):
    x = vx[vt>-99]
    y = vy[vt>-99]
    z = vz[vt>-99]
    e = ve[vt>-99]
    t = vt[vt>-99]
    d = (x*x + y*y + z*z)**0.5
    new_t = t - d/C
    # media pesata dei tempi
    tot_en = ak.sum(e, axis=2) 
    if vm == None:
        w = e
    else:
        m = vm[vt>-99]
        w = e / m
    mean = ak.sum(new_t * w, axis=2) / ak.sum(w, axis=2) + (bx*bx+by*by+bz*bz)**0.5/C
    error = 1. / ((ak.sum(w, axis=2))**0.5)
    #TODO: gestire i nan
    return mean, error

def printLen(x):
    print([len(x[i]) for i in range(len(x))])
    
# given all the MergeTracksters in an event and the index of the CLUE3D trackster
# returns the index of the MergeTrackster it belongs to
def find_trackster_in_candidate(j, TICLtrackster):
    for i, tk in enumerate(TICLtrackster):
        try:
            j = np.where(tk==j)[0][0]
            return i
        except: 
            continue
    return -1 # if we get here sth is wrong

# gves track index of a recoTrackster, if none returns -1
def trackster_to_track(ev, tr_id, TICLtracksters, TICLtracks, verbosity=False):
    tk = find_trackster_in_candidate(tr_id, TICLtracksters[ev])
    if tk == -1:
        print('no trackster in candidate!!')
        return -99
    trackId = TICLtracks[ev][tk]
    if trackId < 4000000:
        if (verbosity):
            print('track ', trackId, ' ev ', ev, ' tk ', tr_id)
        return trackId
    return -1

# return n-th smallest in an array
def nsmall(a, n):
    smallest = np.partition(a, n-1)[n-1]
    return np.argwhere(a==smallest)[0][0]