import awkward as ak
import numpy as np
import uproot as uproot
import matplotlib.pyplot as plt
import sys
import matplotlib as mpl

mpl.rc('xtick', labelsize=20) 
mpl.rc('ytick', labelsize=20) 
mpl.rc('axes', labelsize=20, titlesize=22)
mpl.rcParams["legend.title_fontsize"]=20

filename = sys.argv[1] # 'new_root_files/D99/histo_from1.root'
file = uproot.open(filename)

tracks = file["ticlDumper/tracks"]

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

denom = []
ismuon = []
isTkmuon = []
isboth = []
isNone = []
isNotMuon = []
for ev in range(len(track_pt)):
    for pt, eta, muon, tkMuon in zip(track_hgcal_pt[ev], track_hgcal_eta[ev], track_isMuon[ev], track_isTrackerMuon[ev]):
        if pt > 300:
            continue
        en = pt *np.cosh(eta)
        denom.append(en)
        if muon == -1:
            isNotMuon.append(en)
        elif muon==1 and tkMuon==0:
            ismuon.append(en)
        elif tkMuon==1 and muon == 0:
            isTkmuon.append(en)
        elif muon==1 and tkMuon==1:
            isboth.append(en)
        elif muon==0 and tkMuon==0:    
            isNone.append(en)

den_h, bins = np.histogram(denom, bins=10)
isNotMuon_h, _ = np.histogram(isNotMuon, bins=bins)
ismuon_h, _ = np.histogram(ismuon, bins=bins)
isTkmuon_h, _ = np.histogram(isTkmuon, bins=bins)
isboth_h, _ = np.histogram(isboth, bins=bins)
isNone_h, _ = np.histogram(isNone, bins=bins)
plt.figure(figsize=(10,8))
bin_centers = 0.5 * (bins[1:] + bins[:-1])
bin_width = (bin_centers[1] - bin_centers[0])/1.5
plt.bar(bin_centers, ismuon_h/den_h,width= bin_width , color="red")
plt.bar(bin_centers, isTkmuon_h/den_h, bottom=ismuon_h/den_h,width= bin_width , color="blue")
plt.bar(bin_centers, isboth_h/den_h, bottom=ismuon_h/den_h+isTkmuon_h/den_h,width= bin_width , color="yellow")
plt.bar(bin_centers, isNone_h/den_h, bottom=ismuon_h/den_h+isTkmuon_h/den_h+isboth_h/den_h,width= bin_width , color="green")
plt.bar(bin_centers, isNotMuon_h/den_h, bottom=ismuon_h/den_h+isTkmuon_h/den_h+isboth_h/den_h+isNone_h/den_h, width= bin_width , color="orange")
plt.legend(["isMuon", "isTkMuon", "isBoth", "isNone", "isNotMuon"], fontsize=18)
plt.title("tracks and MuonId - "+ filename.split("/")[1] + " - " + filename.split("/")[-1].replace(".root", ""), y=1.05)
plt.xlabel("Energy (GeV)")
plt.ylabel("Counts / bin")
plt.savefig("linking_plots/muon_id/"+filename.split("/")[-1].replace(".root", "")+"_"+filename.split("/")[1]+".png")
