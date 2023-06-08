from iminuit import cost, Minuit
import matplotlib.pyplot as plt
import numpy as np
import awkward as ak

def myhist(X, bins=30, title='title', xlabel='time (ns)', ylabel='Counts', range=None):
    #plt.figure(dpi=100)
    if range==None:
        plt.hist(np.array(X), bins=bins, color='dodgerblue')
    else:
        plt.hist(np.array(X), bins=bins, color='dodgerblue', range=range)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()  

#Define the Gaussian function
def model(x, A, x0, sigma):
    return A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

# returns res that contains the parameters, the chi squared and 
# the list of bins used t oplot the data
def gauss_fit(data, init_parms, bins=300):
    hist, nbins = np.histogram(data, bins=bins)
    errors = [np.sqrt(oh+1) for oh in hist]
    init_parameters = init_parms
    cost_func = cost.LeastSquares(nbins[:-1], hist, errors, model)
    min_obj = Minuit(cost_func, *init_parameters)
    res = min_obj.migrad()
    chi2 = min_obj.fval/(len(nbins[:-1])-3)
    return res, chi2, hist, nbins[:-1]

def gauss_fit_and_plot(data, init_parms, label="data", colors=["midnightblue","dodgerblue"], bins=300):
    res, chi2, hists, newbins = gauss_fit(data, init_parms, bins=bins)
    y = model(newbins, *res.values)
    plt.plot(newbins, y, label=f'gauss fit $\sigma$ = {res.values[2]:.3f} $\pm$ {res.errors[2]:.3f}\n   $x_0$ = {res.values[1]:.3f} $\pm$ {res.errors[1]:.3f} \n   $\chi^2_0$ = {chi2:.3f}', color=colors[0], linewidth=2)
    #plt.plot(newbins, hist, label=label, color=colors[1], alpha=0.7)
    plt.hist(np.array(data), bins=bins, color=colors[1], alpha=0.7)
    plt.legend()
    plt.grid()
    return res, chi2

def track_efficiency(ele_prompt, ele_track, ele_sim_pt, ele_dz, ele_dxy, ele_barrel, ele_time, ele_timeErr, track_sim_pt, track_dz_ele, track_time, track_timeErr, SIGNAL=True, BARREL=True, ELE_DZ=0.2):
    time_ele_eff = []
    all_ele_eff = []
    time_track_eff = []
    all_track_eff = []
    for ev in range(len(ele_prompt)):
        for ele_idx in range(len(ele_prompt[ev])): 
            # check sulla trackref (se -1 skip)
            if (ele_track[ev][ele_idx]==-1):
                continue
            pt = ele_sim_pt[ev][ele_idx]
            # cut on dxy, dz wrt to the PV
            if (ele_dz[ev][ele_idx]>0.5 or ele_dxy[ev][ele_idx]>0.2):
                continue
            # if prompt -> signal, if not -> bkg
            if SIGNAL:
                if not ele_prompt[ev][ele_idx]:
                    continue
            else:
                if ele_prompt[ev][ele_idx]:
                    continue    
            if BARREL:
                if not ele_barrel[ev][ele_idx]:
                    continue
            else:
                if ele_barrel[ev][ele_idx]:
                    continue

            all_ele_eff.append(pt)
            eleTime = ele_time[ev][ele_idx]
            eleErr  = ele_timeErr[ev][ele_idx]
            if eleErr != -1:
                time_ele_eff.append(pt)
            for trk_idx in range(len(track_sim_pt[ev][ele_idx])):
                trackPT = track_sim_pt[ev][ele_idx][trk_idx]
                if (track_dz_ele[ev][ele_idx][trk_idx] > ELE_DZ):
                    continue
                all_track_eff.append(trackPT)
                trTime  = track_time[ev][ele_idx][trk_idx]
                trErr   = track_timeErr[ev][ele_idx][trk_idx]
                if trErr != -1:
                    time_track_eff.append(trackPT)
    return time_ele_eff, all_ele_eff, time_track_eff, all_track_eff 

def track_eff_plot(bins, all_pt, track_pt, title="electrons tracks with time", color="dodgerblue", pos=111):
    pt_tot, bins = np.histogram(all_pt, bins=bins)
    pt_MTD, _ = np.histogram(track_pt, bins=bins) 

    MTD = pt_MTD / pt_tot
    err_MTD = np.sqrt( pt_MTD / (pt_tot*pt_tot) + pt_MTD**2 / (pt_tot**3) )

    plt.subplot(pos)
    plt.errorbar(bins[:-1], MTD, err_MTD, c=color, fmt = "o", markersize=5, mfc=color, mec=color, ecolor=color, capsize=5, linestyle='')
    plt.title(title)
    plt.xlabel("pT sim (GeV)")
    plt.ylabel("efficiency")

def list2hist(hist,den,bins,SCALE=0.2):
    num, _ = np.histogram(hist, bins=bins)
    ratio = num / den
    err = SCALE*np.sqrt( ratio / (den*den) + ratio**2 / (den**3) )
    return ratio, err

def isoefficiency(ele_prompt, ele_track, ele_sim_pt, ele_PT, ele_dz, ele_dxy, track_sim_pt, track_pt, 
                  track_dz_ele, track_sim_time, ele_sim_time, 
                  track_time, track_timeErr, ele_time, ele_timeErr, track_mva, ele_mva, track_gen_matched, 
                  NSIGMA=3, ELE_DZ=0.2, ISO_CUT=0.03, SIGNAL=True):
    ele_pt = []
    ele_pt_noMTD = []
    ele_pt_MTD = []
    ele_sim_pt_MTD = []
    ele_gen_pt_MTD = []
    
    MVA_CUT=0.5
    ERR = (ak.mean(track_timeErr[track_timeErr!=-1])**2+ak.mean(track_timeErr[track_timeErr!=-1])**2)**0.5

    for ev in range(len(ele_prompt)):
        for ele_idx in range(len(ele_prompt[ev])):
            if (ele_sim_pt[ev][ele_idx]==-1):
                continue
            # check sulla trackref (se -1 skip)
            if (ele_track[ev][ele_idx]==-1):
                continue

            pt = ele_sim_pt[ev][ele_idx]
            pt_reco = ele_PT[ev][ele_idx]
            # cut on dxy, dz wrt to the PV
            if (ele_dz[ev][ele_idx]>0.5 or ele_dxy[ev][ele_idx]>0.2):
                continue
            # if prompt -> signal, if not -> bkg
            if SIGNAL:
                if not ele_prompt[ev][ele_idx]:
                    continue
            else:
                if ele_prompt[ev][ele_idx]:
                    continue

            # add pt, different plots for Barrel and Endcap
            ele_pt.append(pt)

            sum_sim_mtd = 0
            sum_noMtd = 0
            sum_mtd = 0  
            sum_gen_mtd = 0
            
            # loop over tracks - SIM
            for trk_idx in range(len(track_sim_pt[ev][ele_idx])):
                if (track_sim_pt[ev][ele_idx][trk_idx]==-1):
                    continue
                # cut in dz con ele, provare diversi valori
                if (track_dz_ele[ev][ele_idx][trk_idx] > ELE_DZ):
                    continue
                # no MTD
                sum_noMtd += track_sim_pt[ev][ele_idx][trk_idx]
                
                trSimTime  = track_sim_time[ev][ele_idx][trk_idx]
                eleSimTime = ele_sim_time[ev][ele_idx]
                trTime  = track_time[ev][ele_idx][trk_idx]
                trErr   = track_timeErr[ev][ele_idx][trk_idx]
                eleTime = ele_time[ev][ele_idx]
                eleErr  = ele_timeErr[ev][ele_idx]
                # SIM
                if (trSimTime != -1 and eleSimTime != -1):
                    # 2. add track and pt for time 
                    if abs(trSimTime-eleSimTime) < (NSIGMA*ERR):
                        sum_sim_mtd += track_sim_pt[ev][ele_idx][trk_idx]
                else:
                    # no time, add anyway 
                    sum_sim_mtd += track_sim_pt[ev][ele_idx][trk_idx]
                
                # mva cut
                if (track_mva[ev][ele_idx][trk_idx] < MVA_CUT):
                    trErr = -1
                if (ele_mva[ev][ele_idx] < MVA_CUT):
                    eleErr = -1
                # RECO
                if (trErr > 0 and eleErr > 0):
                    # 2. add track and pt for time 
                    if (abs(trTime-eleTime) < (NSIGMA*(trErr**2+eleErr**2)**0.5)):
                        sum_mtd += track_pt[ev][ele_idx][trk_idx]
                else:
                    sum_mtd += track_pt[ev][ele_idx][trk_idx]
                # GEN   
                if track_gen_matched[ev][ele_idx][trk_idx]:
                    sum_gen_mtd += track_sim_pt[ev][ele_idx][trk_idx]
                    
            # compute relative iso and check cut            
            if (sum_sim_mtd / pt < ISO_CUT):
                ele_sim_pt_MTD.append(pt)         
            if (sum_noMtd / pt < ISO_CUT):
                ele_pt_noMTD.append(pt)
            if (sum_mtd / pt_reco < ISO_CUT):
                ele_pt_MTD.append(pt)
            if (sum_gen_mtd / pt < ISO_CUT):    
                ele_gen_pt_MTD.append(pt)
    return ele_pt, ele_pt_noMTD, ele_pt_MTD, ele_sim_pt_MTD, ele_gen_pt_MTD
    
def iso_eff_plot(bins, ele_pt, ele_pt_noMTD, ele_pt_MTD, ele_sim_pt_MTD, ele_gen_pt_MTD, title="iso efficiency on electrons", pos=111, ax=None):
    pt_tot, bins = np.histogram(ele_pt, bins=bins)

    MTD_sim, err_sim_MTD = list2hist(ele_sim_pt_MTD, pt_tot, bins)
    MTD_gen, err_gen_MTD = list2hist(ele_gen_pt_MTD, pt_tot, bins)
    MTD, err_MTD = list2hist(ele_pt_MTD, pt_tot, bins)
    noMTD, err_noMTD = list2hist(ele_pt_noMTD, pt_tot, bins)
    if ax == None:
        ax = plt.subplot(pos)
    else:
        plt.subplot(pos, sharey=ax)
    plt.errorbar(bins[:-1], MTD_sim, err_sim_MTD, c="forestgreen", label ="MTD - sim time", fmt = "o", markersize=5, mfc="forestgreen", mec="forestgreen", ecolor="forestgreen", capsize=5, linestyle='')
    plt.errorbar(bins[:-1], MTD_gen, err_gen_MTD, c="grey", label ="gen info", fmt = "*", markersize=5, mfc="grey", mec="grey", ecolor="grey", capsize=5, linestyle='')
    plt.errorbar(bins[:-1], MTD, err_MTD, c="dodgerblue", label ="MTD - reco time", fmt = "s", markersize=5, mfc="dodgerblue", mec="dodgerblue", ecolor="dodgerblue", capsize=5, linestyle='')
    plt.errorbar(bins[:-1], noMTD, err_noMTD, c="red", label ="no MTD", fmt = "^", markersize=5, mfc="red", mec="red", ecolor="red", capsize=5, linestyle='')
    plt.title(title)
    plt.xlabel("pT sim (GeV)")
    plt.ylabel("Counts / bin")
    plt.legend(loc="upper left")
    return ax

def isolation(ele_prompt, ele_track, ele_sim_pt, ele_PT, ele_dz, ele_dxy, track_sim_pt, track_pt, 
              track_dz_ele, track_sim_time, ele_sim_time, 
              track_time, track_timeErr, ele_time, ele_timeErr, track_mva, ele_mva, track_gen_matched, 
              NSIGMA=3,ELE_DZ=0.2,SIGNAL=True):
    ele_pt_noMTD = []
    ele_pt_MTD = []
    ele_sim_pt_MTD = []
    ele_gen_pt_MTD = []
    
    MVA_CUT=0.5
    ERR = (ak.mean(track_timeErr[track_timeErr!=-1])**2+ak.mean(track_timeErr[track_timeErr!=-1])**2)**0.5

    for ev in range(len(ele_prompt)):
        for ele_idx in range(len(ele_prompt[ev])):
            if (ele_sim_pt[ev][ele_idx]==-1):
                continue
            # check sulla trackref (se -1 skip)
            if (ele_track[ev][ele_idx]==-1):
                continue

            pt = ele_sim_pt[ev][ele_idx]
            pt_reco = ele_PT[ev][ele_idx]
            # cut on dxy, dz wrt to the PV
            if (ele_dz[ev][ele_idx]>0.5 or ele_dxy[ev][ele_idx]>0.2):
                continue
            # if prompt -> signal, if not -> bkg
            if SIGNAL:
                if not ele_prompt[ev][ele_idx]:
                    continue
            else:
                if ele_prompt[ev][ele_idx]:
                    continue

            sum_sim_mtd = 0
            sum_noMtd = 0
            sum_mtd = 0  
            sum_gen_mtd = 0
            
            # loop over tracks - SIM
            for trk_idx in range(len(track_sim_pt[ev][ele_idx])):
                if (track_sim_pt[ev][ele_idx][trk_idx]==-1):
                    continue
              # cut in dz con ele, provare diversi valori
                if (track_dz_ele[ev][ele_idx][trk_idx] > ELE_DZ):
                    continue
                # no MTD
                sum_noMtd += track_sim_pt[ev][ele_idx][trk_idx]
                
                trSimTime  = track_sim_time[ev][ele_idx][trk_idx]
                eleSimTime = ele_sim_time[ev][ele_idx]
                trTime  = track_time[ev][ele_idx][trk_idx]
                trErr   = track_timeErr[ev][ele_idx][trk_idx]
                eleTime = ele_time[ev][ele_idx]
                eleErr  = ele_timeErr[ev][ele_idx]
                # SIM
                if (trSimTime != -1 and eleSimTime != -1):
                    # 2. add track and pt for time 
                    if abs(trSimTime-eleSimTime) < (NSIGMA*ERR):
                        sum_sim_mtd += track_sim_pt[ev][ele_idx][trk_idx]
                else:
                    # no time, add anyway 
                    sum_sim_mtd += track_sim_pt[ev][ele_idx][trk_idx]
                
                # mva cut
                if (track_mva[ev][ele_idx][trk_idx] < MVA_CUT):
                    trErr = -1
                if (ele_mva[ev][ele_idx] < MVA_CUT):
                    eleErr = -1
                # RECO
                if (trErr > 0 and eleErr > 0):
                    # 2. add track and pt for time 
                    if (abs(trTime-eleTime) < (NSIGMA*(trErr**2+eleErr**2)**0.5)):
                        sum_mtd += track_pt[ev][ele_idx][trk_idx]
                else:
                    sum_mtd += track_pt[ev][ele_idx][trk_idx]
                # GEN   
                if track_gen_matched[ev][ele_idx][trk_idx]:
                    sum_gen_mtd += track_sim_pt[ev][ele_idx][trk_idx]
                    
            # compute relative iso            
            ele_sim_pt_MTD.append(sum_sim_mtd / pt)         
            ele_pt_noMTD.append(sum_noMtd / pt)
            ele_pt_MTD.append(sum_mtd / pt_reco)
            ele_gen_pt_MTD.append(sum_gen_mtd / pt)
    return ele_pt_noMTD, ele_pt_MTD, ele_sim_pt_MTD, ele_gen_pt_MTD

def isovertexefficiency(ele_prompt, ele_track, ele_sim_pt, ele_dz, ele_dxy, track_sim_pt, track_dz_ele, track_sim_time, ele_sim_time, 
                        track_time, track_timeErr, vertex_time, vertex_timeErr, track_mva, ele_mva, track_gen_matched, 
                        NSIGMA=3, ELE_DZ=0.2, ISO_CUT=0.03, SIGNAL=True):
    ele_pt = []
    ele_pt_noMTD = []
    ele_pt_MTD = []
    ele_sim_pt_MTD = []
    ele_gen_pt_MTD = []
    
    MVA_CUT=0.5
    ERR = (ak.mean(track_timeErr[track_timeErr!=-1])**2+ak.mean(track_timeErr[track_timeErr!=-1])**2)**0.5

    for ev in range(len(ele_prompt)):
        vertexTime = vertex_time[ev]
        vertexTimeErr = vertex_timeErr[ev]
        for ele_idx in range(len(ele_prompt[ev])):
            # check sulla trackref (se -1 skip)
            if (ele_track[ev][ele_idx]==-1):
                continue

            pt = ele_sim_pt[ev][ele_idx]
            # cut on dxy, dz wrt to the PV
            if (ele_dz[ev][ele_idx]>0.5 or ele_dxy[ev][ele_idx]>0.2):
                continue
            # if prompt -> signal, if not -> bkg
            if SIGNAL:
                if not ele_prompt[ev][ele_idx]:
                    continue
            else:
                if ele_prompt[ev][ele_idx]:
                    continue

            # add pt, different plots for Barrel and Endcap
            ele_pt.append(pt)

            sum_sim_mtd = 0
            sum_noMtd = 0
            sum_mtd = 0  
            sum_gen_mtd = 0
            
            # loop over tracks - SIM
            for trk_idx in range(len(track_sim_pt[ev][ele_idx])):
              # cut in dz con ele, provare diversi valori
                if (track_dz_ele[ev][ele_idx][trk_idx] > ELE_DZ):
                    continue
                # no MTD
                sum_noMtd += track_sim_pt[ev][ele_idx][trk_idx]
                
                trSimTime  = track_sim_time[ev][ele_idx][trk_idx]
                eleSimTime = ele_sim_time[ev][ele_idx]
                trTime  = track_time[ev][ele_idx][trk_idx]
                trErr   = track_timeErr[ev][ele_idx][trk_idx]
                # SIM
                if (trSimTime != -1 and eleSimTime != -1):
                    # 2. add track and pt for time 
                    if abs(trSimTime-eleSimTime) < (NSIGMA*ERR):
                        sum_sim_mtd += track_sim_pt[ev][ele_idx][trk_idx]
                else:
                    # no time, add anyway 
                    sum_sim_mtd += track_sim_pt[ev][ele_idx][trk_idx]
                
                # mva cut
                if (track_mva[ev][ele_idx][trk_idx] < MVA_CUT):
                    trErr = -1
                # RECO
                if (trErr > 0 and vertexTimeErr > 0):
                    # 2. add track and pt for time 
                    if (abs(trTime-vertexTime) < (NSIGMA*(trErr**2+vertexTimeErr**2)**0.5)):
                        sum_mtd += track_sim_pt[ev][ele_idx][trk_idx]
                else:
                    sum_mtd += track_sim_pt[ev][ele_idx][trk_idx]
                # GEN   
                if track_gen_matched[ev][ele_idx][trk_idx]:
                    sum_gen_mtd += track_sim_pt[ev][ele_idx][trk_idx]
                    
            # compute relative iso and check cut            
            if (sum_sim_mtd / pt < ISO_CUT):
                ele_sim_pt_MTD.append(pt)         
            if (sum_noMtd / pt < ISO_CUT):
                ele_pt_noMTD.append(pt)
            if (sum_mtd / pt < ISO_CUT):
                ele_pt_MTD.append(pt)
            if (sum_gen_mtd / pt < ISO_CUT):    
                ele_gen_pt_MTD.append(pt)
    return ele_pt, ele_pt_noMTD, ele_pt_MTD, ele_sim_pt_MTD, ele_gen_pt_MTD

def vertexisolation(ele_prompt, ele_track, ele_sim_pt, ele_dz, ele_dxy, track_sim_pt, track_dz_ele, track_sim_time, ele_sim_time, 
              track_time, track_timeErr, vertex_time, vertex_timeErr, track_mva, ele_mva, track_gen_matched, 
              NSIGMA=3,ELE_DZ=0.2,SIGNAL=True):
    ele_pt_noMTD = []
    ele_pt_MTD = []
    ele_sim_pt_MTD = []
    ele_gen_pt_MTD = []
    
    MVA_CUT=0.5
    ERR = (ak.mean(track_timeErr[track_timeErr!=-1])**2+ak.mean(track_timeErr[track_timeErr!=-1])**2)**0.5

    for ev in range(len(ele_prompt)):
        vertexTime = vertex_time[ev]
        vertexTimeErr = vertex_timeErr[ev]
        for ele_idx in range(len(ele_prompt[ev])):
            # check sulla trackref (se -1 skip)
            if (ele_track[ev][ele_idx]==-1):
                continue

            pt = ele_sim_pt[ev][ele_idx]
            # cut on dxy, dz wrt to the PV
            if (ele_dz[ev][ele_idx]>0.5 or ele_dxy[ev][ele_idx]>0.2):
                continue
            # if prompt -> signal, if not -> bkg
            if SIGNAL:
                if not ele_prompt[ev][ele_idx]:
                    continue
            else:
                if ele_prompt[ev][ele_idx]:
                    continue

            sum_sim_mtd = 0
            sum_noMtd = 0
            sum_mtd = 0  
            sum_gen_mtd = 0
            
            # loop over tracks - SIM
            for trk_idx in range(len(track_sim_pt[ev][ele_idx])):
              # cut in dz con ele, provare diversi valori
                if (track_dz_ele[ev][ele_idx][trk_idx] > ELE_DZ):
                    continue
                # no MTD
                sum_noMtd += track_sim_pt[ev][ele_idx][trk_idx]
                
                trSimTime  = track_sim_time[ev][ele_idx][trk_idx]
                eleSimTime = ele_sim_time[ev][ele_idx]
                trTime  = track_time[ev][ele_idx][trk_idx]
                trErr   = track_timeErr[ev][ele_idx][trk_idx]
                # SIM
                if (trSimTime != -1 and eleSimTime != -1):
                    # 2. add track and pt for time 
                    if abs(trSimTime-eleSimTime) < (NSIGMA*ERR):
                        sum_sim_mtd += track_sim_pt[ev][ele_idx][trk_idx]
                else:
                    # no time, add anyway 
                    sum_sim_mtd += track_sim_pt[ev][ele_idx][trk_idx]
                
                # mva cut
                if (track_mva[ev][ele_idx][trk_idx] < MVA_CUT):
                    trErr = -1
                # RECO
                if (trErr > 0 and vertexTimeErr > 0):
                    # 2. add track and pt for time 
                    if (abs(trTime-vertexTime) < (NSIGMA*(trErr**2+vertexTimeErr**2)**0.5)):
                        sum_mtd += track_sim_pt[ev][ele_idx][trk_idx]
                else:
                    sum_mtd += track_sim_pt[ev][ele_idx][trk_idx]
                # GEN   
                if track_gen_matched[ev][ele_idx][trk_idx]:
                    sum_gen_mtd += track_sim_pt[ev][ele_idx][trk_idx]
                    
            # compute relative iso            
            ele_sim_pt_MTD.append(sum_sim_mtd / pt)         
            ele_pt_noMTD.append(sum_noMtd / pt)
            ele_pt_MTD.append(sum_mtd / pt)
            ele_gen_pt_MTD.append(sum_gen_mtd / pt)
    return ele_pt_noMTD, ele_pt_MTD, ele_sim_pt_MTD, ele_gen_pt_MTD