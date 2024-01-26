import awkward as ak
import numpy as np
import uproot as uproot
import matplotlib.pyplot as plt
import sys
import matplotlib as mpl
import matplotlib.patches as mpatches

mpl.rc('xtick', labelsize=20) 
mpl.rc('ytick', labelsize=20) 
mpl.rc('axes', labelsize=20, titlesize=22)
mpl.rcParams["legend.title_fontsize"]=20

filename = sys.argv[1] # 'new_root_files/D99/histo_from1.root'
file = uproot.open(filename)

simtrackstersSC = file["ticlDumper/simtrackstersSC"]
simtrackstersCP = file["ticlDumper/simtrackstersCP"]
tracksters  = file["ticlDumper/tracksters"]
trackstersMerged = file["ticlDumper/trackstersMerged"]
associations = file["ticlDumper/associations"]
tracks = file["ticlDumper/tracks"]
simTICLCandidate = file["ticlDumper/simTICLCandidate"]
TICLCandidate = file["ticlDumper/candidates"]
clusters = file["ticlDumper/clusters"]



simTICLCandidate_time = simTICLCandidate['simTICLCandidate_time'].array()
simTICLCandidate_raw_energy = simTICLCandidate['simTICLCandidate_raw_energy'].array()
simTICLCandidate_regressed_energy = simTICLCandidate['simTICLCandidate_regressed_energy'].array()
simTICLCandidate_track_in_candidate = simTICLCandidate['simTICLCandidate_track_in_candidate'].array()

candidate_pdgId         = TICLCandidate["candidate_pdgId"].array()
candidate_id_prob       = TICLCandidate["candidate_id_probabilities"].array()
tracksters_in_candidate = TICLCandidate["tracksters_in_candidate"].array()
track_in_candidate      = TICLCandidate["track_in_candidate"].array()
candidate_energy        = TICLCandidate["candidate_energy"].array()
candidate_raw_energy        = TICLCandidate["candidate_raw_energy"].array()
candidate_time          = TICLCandidate["candidate_time"].array()
candidate_timeErr       = TICLCandidate["candidate_timeErr"].array()
tracsktersMerged_rawEne       = trackstersMerged["raw_energy"].array()

track_pt = tracks["track_pt"].array()
track_id = tracks["track_id"].array()
track_hgcal_eta = tracks["track_hgcal_eta"].array()
track_hgcal_pt = tracks["track_hgcal_pt"].array()
track_missing_outer_hits = tracks["track_missing_outer_hits"].array()
track_nhits = tracks["track_nhits"].array()
track_quality = tracks["track_quality"].array()
track_time_mtd_err = tracks["track_time_mtd_err"].array()
track_isMuon = tracks["track_isMuon"].array()
track_isTrackerMuon = tracks["track_isTrackerMuon"].array()

colors = ["green", "blue", "red", "yellow", "orange"]
labels = ["isNone", "isTkMuon", "isMuon", "isBoth", "isNotMuon"]
counts = [0,0,0,0,0]
nevents = len(track_pt)
ncandidates = 0

fig = plt.figure(figsize=(8,8), dpi=100)
#for tk_in_simCand, tkM_rawEne, tk_in_cand, ts_in_cand, trk_pt, trk_eta, trk_id, trk_miss_out, trk_isMuon, trk_isTkMuon in zip(simTICLCandidate_track_in_candidate, tracsktersMerged_rawEne, track_in_candidate, tracksters_in_candidate, track_hgcal_pt,track_hgcal_eta, track_id, track_missing_outer_hits, track_isMuon, track_isTrackerMuon):
#    for i, tk in enumerate(tk_in_cand):
#        if tk != -1:
#            try:
#                idx = np.where(trk_id==tk)[0][0]
#            except:
#                continue
#            ncandidates += 1
#            pt = trk_pt[idx]
#            eta = trk_eta[idx]
#            missing_out = trk_miss_out[idx]
#            isMuon = trk_isMuon[idx]
#            isTkMuon = trk_isTkMuon[idx]
#            colorMuon = isMuon *2 + isTkMuon
#            if colorMuon < 0: colorMuon = 4
#            if len(ts_in_cand[i])== 0 and not missing_out and tk_in_simCand[0]!=-1:
#                plt.scatter(pt*np.cosh(eta), ak.sum(tkM_rawEne), c=colors[colorMuon], alpha=0.4, marker="o")
#                counts[colorMuon] += 1

for tk_in_simCand, tkM_rawEne, tk_in_cand, ts_in_cand, raw_energy, trk_pt, trk_eta, trk_id, trk_miss_out, trk_isMuon, trk_isTkMuon in zip(simTICLCandidate_track_in_candidate, tracsktersMerged_rawEne, track_in_candidate, tracksters_in_candidate, candidate_raw_energy, track_hgcal_pt, track_hgcal_eta, track_id, track_missing_outer_hits, track_isMuon, track_isTrackerMuon):
    for i, tk in enumerate(tk_in_simCand):
        if tk != -1:
            try:
                idx = np.where(trk_id==tk)[0][0]
            except:
                continue
            ncandidates += 1
            pt = trk_pt[idx]
            eta = trk_eta[idx]
            missing_out = trk_miss_out[idx]
            isMuon = trk_isMuon[idx]
            isTkMuon = trk_isTkMuon[idx]
            colorMuon = isMuon *2 + isTkMuon
            if colorMuon < 0: colorMuon = 4
            tracksters =  ts_in_cand[tk_in_cand == tk][0]
            if len(tracksters)== 0:
                plt.scatter(pt*np.cosh(eta), ak.max(tkM_rawEne), c="red", alpha=0.4, marker="o") #c=colors[missing_inn],
                counts[colorMuon] += 1
            else:
                plt.scatter(pt*np.cosh(eta), raw_energy[tk_in_cand == tk], c="blue", alpha=0.4, marker="s")

            #if len(ts_in_cand[i])== 0 and not missing_out and tk_in_simCand[0]!=-1:
plt.xlabel("track energy (GeV) = pT*cosh(eta)")
plt.ylabel("trackstersMerged raw energy (Gev)")
x = np.linspace(0,1500,2)
plt.plot(x, x, c="red")
plt.axis("equal")
plt.title("track-only TICLCandidates\n("+str(ncandidates)+ " tracks that could be linked)\n"+ filename.split("/")[1] + " - " + filename.split("/")[-1].replace(".root", ""))
plt.xlim(-50,1500)
patches = []
for label, color, count in zip(labels, colors, counts):
    patches.append(mpatches.Patch(color=color, label=label+" ("+str(count)+")"))
plt.legend(handles=patches, fontsize=18)
fig.tight_layout()
plt.savefig("linking_plots/track_only/"+filename.split("/")[-1].replace(".root", "")+"_"+filename.split("/")[1]+".png")
