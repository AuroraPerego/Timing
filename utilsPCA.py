import numpy as np
from wpca import WPCA, EMPCA
import matplotlib.pyplot as plt

def multiplicity(svx, svy, svz, sve, svt, svi, svm):
    # keep only the lowest multiplicity per vertex
    plot_set = {}
    for vi, vm in zip(svi, svm):
        for i, m in zip(vi, vm):
            if i in plot_set and plot_set[i] < m:
                continue
            plot_set[i] = m
    _tx1, _tx2 = [], []
    _ty1, _ty2 = [], []
    _tz1, _tz2 = [], []
    _te1, _te2 = [], []
    _tt1, _tt2 = [], []
    for ti, tx, ty, tz, te, tt, vi, vm in zip(range(len(svx)), svx, svy, svz, sve, svt, svi, svm):
        if ti==0:
            for x, y, z, e, t, i, m in zip(tx, ty, tz, te, tt, vi, vm):
                # do not replot points with higher multiplicity
                if plot_set[i] != m:
                    _tx2.append(x)
                    _ty2.append(y)
                    _tz2.append(z)
                    _te2.append(e)
                    _tt2.append(t)
                    continue
                _tx1.append(x)
                _ty1.append(y)
                _tz1.append(z)
                _te1.append(e)
                _tt1.append(t)
        else:
            for x, y, z, e, t, i, m in zip(tx, ty, tz, te, tt, vi, vm):
                # do not replot points with higher multiplicity
                if plot_set[i] != m:
                    _tx1.append(x)
                    _ty1.append(y)
                    _tz1.append(z)
                    _te1.append(e)
                    _tt1.append(t)
                    continue
                _tx2.append(x)
                _ty2.append(y)
                _tz2.append(z)
                _te2.append(e)
                _tt2.append(t)            
    return _tx1, _ty1, _tz1, _te1, _tt1, _tx2, _ty2, _tz2, _te2, _tt2

def distance_matrix(trk_x, trk_y, trk_z):
    v_matrix = np.concatenate(([trk_x], [trk_y], [trk_z]))
    gram = v_matrix.T.dot(v_matrix)

    distance = np.zeros(np.shape(gram))
    for row in range(np.shape(distance)[0]):
        for col in range(np.shape(distance)[1]): #half of the matrix is sufficient, but then mask doesn't work properly
            distance[row][col] = (gram[row][row]-2*gram[row][col]+gram[col][col])**0.5 #= 0 if row==col else
    #print('\n'.join(['\t'.join([str("{:.2f}".format(cell)) for cell in row]) for row in distance]))
    return distance

def project_lc_to_pca(point, segment_start, segment_end):
    """
    `lc` is a 1D NumPy array of length 3 representing the point;
    `segment_start` and `segment_end` are 1D NumPy arrays of length 3 representing the start and end points of the line segment. 
    The function returns the minimum distance between the point and the line segment and the closest point on this segment.
    """
    segment_vector = segment_end - segment_start
    #print(segment_vector)
    point_vector = point - segment_start
    #print(point_vector)
    if not np.any(segment_vector):
        projection = 0
    else:
        projection = np.dot(point_vector, segment_vector) / np.dot(segment_vector, segment_vector)

    closest_point = segment_start + projection * segment_vector
    return np.linalg.norm(point - closest_point), closest_point

def find_dist_to_closest_edge_set(point, edges):
    """
    point: np.array (x,y,z)
    edges: set
    
    """
    min_dist = np.inf
    min_edge = None
    closest_point = None
    closest_endpoint = None
    point = np.array(point)
    
    for (segment_start, segment_end) in edges:
        
        segment_start, segment_end = np.array(segment_start), np.array(segment_end)
        dist, closest_point_tmp, closest_endpoint_tmp = distance_point_to_segment_set(point, segment_start, segment_end)
        
        if dist < min_dist:
            min_dist = dist
            closest_point = closest_point_tmp
            closest_endpoint = closest_endpoint_tmp
            
    return min_dist, min_edge, closest_point, closest_endpoint

def distance_point_to_segment(point, segment_start, segment_end):
    """
    `point` is a 1D NumPy array of length 3 representing the point;
    `segment_start` and `segment_end` are 1D NumPy arrays of length 3 representing the start and end points of the line segment. 
    The function returns the minimum distance between the point and the line segment and the closest point on this segment.
    """
    segment_vector = segment_end - segment_start
    point_vector = point - segment_start
    if not np.any(segment_vector):
        projection = 0
    else:
        projection = np.dot(point_vector, segment_vector) / np.dot(segment_vector, segment_vector)

    if projection <= 0:
        
        return np.linalg.norm(point_vector)
    elif projection > 1:
            
        return np.linalg.norm(point - segment_end)
    else:
        closest_point = segment_start + projection * segment_vector
        
        return np.linalg.norm(point - closest_point)

def distance_point_to_segment_set(point, segment_start, segment_end):
    """
    `point` is a 1D NumPy array of length 3 representing the point;
    `segment_start` and `segment_end` are 1D NumPy arrays of length 3 representing the start and end points of the line segment. 
    The function returns the minimum distance between the point and the line segment and the closest point on this segment.
    """
    segment_vector = segment_end - segment_start
    point_vector = point - segment_start
    if not np.any(segment_vector):
        projection = 0
    else:
        projection = np.dot(point_vector, segment_vector) / np.dot(segment_vector, segment_vector)

    if projection <= 0:
        
        return np.linalg.norm(point_vector), segment_start, segment_start
    elif projection > 1:
            
        return np.linalg.norm(point - segment_end), segment_end, segment_end
    else:
        closest_point = segment_start + projection * segment_vector
        closest_endpoint = segment_start if np.linalg.norm(closest_point - segment_start) < np.linalg.norm(closest_point - segment_end) else segment_end
        
        return np.linalg.norm(point - closest_point), closest_point, closest_endpoint

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    
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

    ax.plot([a[x], b[x]], [a[y], b[y]], [a[z], b[z]], 'cyan', alpha=.2)
    ax.add_collection3d(Poly3DCollection(
        vertices, facecolors='cyan', linewidths=1, alpha=.2)) #edgecolors='k', 

def nodes_pca(vx, vy, vz, ve, vt, vi, vm, bx, by, bz, be, bt, ev, tr_id,
                     ENERGY_RATIO_THR=0.01, PCA_NEIGHBOUR_THR=1, verbosity=True):
    # Project the points on pca
    nodes = []

    vxt = vx[ev][tr_id]
    vyt = vy[ev][tr_id]
    vzt = vz[ev][tr_id]
    vet = ve[ev][tr_id]
    vtt = vt[ev][tr_id]

    bxt = bx[ev][tr_id]
    byt = by[ev][tr_id]
    bzt = bz[ev][tr_id]
    bet = be[ev][tr_id]
    btt = bt[ev][tr_id]
    
    if len(vet)==0:
        b_coord = np.array([bxt, byt, bzt])
        nodes.append(np.array([bxt, byt, bzt, btt]))
        return nodes
    
    # Get distances between the LCs
    dist_matrix = distance_matrix(vxt, vyt, vzt)
    # Calculate PCA
    pca = WPCA(n_components=3)    
    positions = np.array((vxt, vyt, vzt)).T

    vet_array = np.array(vet).reshape(-1,1)
    vet_array = np.tile(vet_array,(1, 3))

    pca.fit(positions, weights = vet_array)
    component = pca.components_[0]
    
    # Barycenter
    b_coord = np.array([bxt, byt, bzt])
    nodes.append(np.array([bxt, byt, bzt, btt]))
  
    segment_end = np.array([bxt+component[0],
                            byt+component[1],
                            bzt+component[2]])

    data = [[x,y,z,e,t,j] for x,y,z,e,t,j in zip(vxt, vyt, vzt, vet, vtt, range(len(vxt)))]
    # sorted by energy
    data = sorted(data, key=lambda a: -a[3])

    # project each LC to the principal component
    min_p, max_p = b_coord, b_coord
    max_en = max(vet)
    num_lc_above_th = len(np.array(vxt)[np.array(vet) > 2*ENERGY_RATIO_THR*max_en])
    
    small_trackster = False
    if (max_en < 1 or num_lc_above_th < 5) and verbosity:
        # If maximum energy of the LCs is lower than 2 GeV or number of layer clusters with at least a ENERGY_RATIO_THR of the maximum energy is less than 5
        small_trackster = True
        print("Small trackster")
        #return nodes

    min_point = np.zeros(4)
    max_point = np.zeros(4)
    # try create main edges
    
    for x,y,z,e,t,j in data:

        point = np.array([x, y, z, t])
        dist, closest_point = project_lc_to_pca(point[:3], b_coord, segment_end)
        
        if e/max_en > ENERGY_RATIO_THR and dist < PCA_NEIGHBOUR_THR:
            # limiting the PCA length
            z_cl = closest_point[2]
            if z_cl > 0:
                if min_p[2] > z_cl and point[3]>-99:
                    min_p = closest_point
                    min_point = point

                if max_p[2] < z_cl:
                    max_p = closest_point  
                    max_point = point  
            else:
                if min_p[2] > z_cl:
                    min_p = closest_point
                    min_point = point
                if max_p[2] < z_cl and point[3]>-99:
                    max_p = closest_point  
                    max_point = point
        
    if np.linalg.norm(min_p-max_p) > 0.5:
    
        if not np.allclose(min_p, b_coord, atol=0.1):
            nodes.append(min_point)

        if not np.allclose(max_p, b_coord, atol=0.1):
            nodes.append(max_point)
#     else:
#         min_p, max_p = b_coord, b_coord
#         min_point[:3], max_point[:3] = b_coord, b_coord
#         min_point[3], max_point[3] = btt, btt
    
    return nodes
    
def create_tr_skeletons(vx, vy, vz, ve, vt, vi, vm, bx, by, bz, be, bt, ev, tr_id, DST_THR=2.5, bubble_size=10, 
                                   NEIGHBOUR_THR=30, ENERGY_RATIO_THR=0.01, MIN_EDGE_LEN=1, PCA_NEIGHBOUR_THR=1, 
                                  secondary_edges = False):
    # Project the points on pca
    edges, nodes = set(), set()
    covered_nodes = set()
    covered_node_idx = []
    times = []
  
    vxt = vx[ev][tr_id]
    vyt = vy[ev][tr_id]
    vzt = vz[ev][tr_id]
    vet = ve[ev][tr_id]
    vtt = vt[ev][tr_id]
    #vit = vi[ev][tr_id]

    bxt = bx[ev][tr_id]
    byt = by[ev][tr_id]
    bzt = bz[ev][tr_id]
    bet = be[ev][tr_id]
    btt = bt[ev][tr_id]
    
    # Get distances between the LCs
    dist_matrix = distance_matrix(vxt, vyt, vzt)
    # Calculate PCA
    pca = WPCA(n_components=3)    
    positions = np.array((vxt, vyt, vzt)).T

    vet_array = np.array(vet).reshape(-1,1)
    vet_array = np.tile(vet_array,(1, 3))

    pca.fit(positions, weights = vet_array)
    component = pca.components_[0]
    
    # Barycenter
    b_coord = np.array([bxt, byt, bzt])
    edges.add((tuple(b_coord), tuple(b_coord)))
    nodes.add(tuple(b_coord))
    times.append(np.array([bxt, byt, bzt, btt]))
    ax.scatter(bxt, byt, bzt, s=10, c="red", label=f"{len(vet)} LC: {sum(vet):.2f} GeV")
    if not np.isnan(btt):
        ax.text(bxt, byt, bzt, '%.3f ns' % (btt), size=12)

    segment_end = np.array([bxt+component[0],
                            byt+component[1],
                            bzt+component[2]])

    data = [[x,y,z,e,i,j] for x,y,z,e,i,j in zip(vxt, vyt, vzt, vet, vtt, range(len(vxt)))]
    # sorted by energy
    data = sorted(data, key=lambda a: -a[3])

    # project each LC to the principal component
    min_p, max_p = b_coord, b_coord
    max_en = max(vet)
    num_lc_above_th = len(np.array(vxt)[np.array(vet) > 2*ENERGY_RATIO_THR*max_en])
    
    small_trackster = False
    if max_en < 1 or num_lc_above_th < 5:
        # If maximum energy of the LCs is lower than 2 GeV or number of layer clusters with at least a ENERGY_RATIO_THR of the maximum energy is less than 5
        small_trackster = True
        print("Small trackster")

    min_point = np.zeros(4)
    max_point = np.zeros(4)
    # try create main edges
    
    for x,y,z,e,t,j in data:
        # plot times
        #if i > -80:
        #    ax.text(x, y, z, '%.3f ns' % (i), size=12) #+ (x**2+y**2+z**2)**0.5/C
        
        point = np.array([x, y, z, t])
        dist, closest_point = project_lc_to_pca(point[:3], b_coord, segment_end)
        
        if e/max_en > ENERGY_RATIO_THR and dist < PCA_NEIGHBOUR_THR:
            # limiting the PCA length
            z_cl = closest_point[2]
            if z_cl > 0:
                if min_p[2] > z_cl and point[3]>-99:
                    min_p = closest_point
                    min_point = point

                if max_p[2] < z_cl:
                    max_p = closest_point  
                    max_point = point  
            else:
                if min_p[2] > z_cl:
                    min_p = closest_point
                    min_point = point
                if max_p[2] < z_cl and point[3]>-99:
                    max_p = closest_point  
                    max_point = point  
        
    if np.linalg.norm(min_p-max_p) > 0.5:
    
        if not np.allclose(min_p, b_coord, atol=0.1):
            edges.add((tuple(min_p), tuple(b_coord)))
            nodes.add(tuple(min_p))
            times.append(min_point)
            ax.plot([bxt, min_p[0]], [byt, min_p[1]], [bzt, min_p[2]], c='red')    

        if not np.allclose(max_p, b_coord, atol=0.1):
            edges.add((tuple(max_p), tuple(b_coord)))
            nodes.add(tuple(max_p))
            times.append(max_point)
            ax.plot([bxt, max_p[0]], [byt, max_p[1]], [bzt, max_p[2]], c='green')
    else:
        min_p, max_p = b_coord, b_coord
        min_point[:3], max_point[:3] = b_coord, b_coord
        min_point[3], max_point[3] = btt, btt
        
    print('times of the nearest LCs: {:.3f} ns, {:.3f} ns'.format(min_point[3], max_point[3]))
    ax.scatter(min_point[0], min_point[1], min_point[2], s=20, alpha=0.8, c="r", zorder=-5)   
    ax.scatter(max_point[0], max_point[1], max_point[2], s=20, alpha=0.8, c="r", zorder=-5)
    ax.text(min_point[0], min_point[1], min_point[2], '%.3f ns' % (min_point[3]), size=12)   
    ax.text(max_point[0], max_point[1], max_point[2], '%.3f ns' % (max_point[3]), size=12)
    
    # speed of the shower
    min_speed = np.linalg.norm(min_point[:3] - b_coord) / abs(min_point[3] - btt)
    max_speed = np.linalg.norm(max_point[:3] - b_coord) / abs(max_point[3] - btt)
    print('speed of shower along PCA segments: {:.3f} cm/ns, {:.3f} cm/ns'.format(min_speed, max_speed))
    speed = np.linalg.norm(max_point[:3] - min_point[:3]) / abs(max_point[3] - min_point[3])
    print('speed of shower: {:.3f} cm/ns'.format(speed))


    #Build secondary edges
    for x,y,z,e,i,j in data:
        
        if e / max_en > ENERGY_RATIO_THR:
            # and not covered by any edge
            point = np.array([x, y, z, i])
            min_dist, min_edge, closest_point_edge, closest_endpoint =  find_dist_to_closest_edge_set(point[:3], edges)
            #print(min_dist)
            
            if min_dist < DST_THR:
                # LC covered by some edge already
                ax.scatter(x, y, z, s=e*bubble_size, alpha=0.2, c="green", zorder=-5)
                covered_nodes.add((x, y, z))
                covered_node_idx.append(j)  
                continue
            else:
                ax.scatter(x, y, z, s=e*bubble_size, alpha=0.2, c="black", zorder=-5)
                pass

            if not small_trackster and secondary_edges:
                # find neares higher
                # go through all points, if the point is within the Neighbour_threshold
                distances = dist_matrix[j]
                indices = np.argsort(distances)

                for idx in indices:

                    if idx in covered_node_idx and distances[idx] < NEIGHBOUR_THR and vet[idx] > e and abs(vzt[idx]) < abs(z):
                        # found the nearest higher
                        # create an edge - to projection of the nearest higher to the pca
                        #edges[(idx, j)] = (np.array([vxt[idx], vyt[idx], vzt[idx]]), point)
                        ax.plot([vxt[idx], x], [vyt[idx], y], [vzt[idx], z], c='blue')
                        nearest_higher = np.array([vxt[idx], vyt[idx], vzt[idx]])
                        
                        # TODO: check that it lies on the axis!!!!!!!!!!!!!!!
                        dist, closest_point = project_lc_to_pca(nearest_higher, b_coord, segment_end)

                        if not np.allclose(closest_point, point[:3], atol=0.1):
                            edges.add((tuple(closest_point), tuple(point[:3])))
                            nodes.add(tuple(closest_point))
                            nodes.add(tuple(point[:3]))
                            times.append(point)
                        ax.plot([closest_point[0], x], [closest_point[1], y], [closest_point[2], z], c='blue')

                        break
        else:
            # if low energy
            pass
            ax.scatter(x, y, z, s=e*bubble_size, alpha=0.2, c="blue", zorder=-5)
    
    edges.remove((tuple(b_coord), tuple(b_coord)))
    set_axes_equal(ax)
    return edges, nodes, covered_node_idx, times
    
def create_tr_skeletons_single_pca(vx, vy, vz, ve, vt, vi, vm, bx, by, bz, be, bt, ev, tr_id, DST_THR=2.5, bubble_size=10, 
                                   NEIGHBOUR_THR=30, ENERGY_RATIO_THR=0.01, MIN_EDGE_LEN=1, PCA_NEIGHBOUR_THR=1, 
                                  secondary_edges = False, SIM = True):
    # Project the points on pca
    edges, nodes = set(), set()
    covered_nodes = set()
    covered_node_idx = []
    times = []
    
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f"Reconstruction of an electron\nLayer Clusters")
    ax.set(xlabel="x (cm)", ylabel="y (cm)", zlabel="z (cm)")
    if SIM:
        eid=ev
        tx1, ty1, tz1, te1, tt1, tx2, ty2, tz2, te2, tt2 = multiplicity(vx[eid], vy[eid], vz[eid], ve[eid], vt[eid], vi[eid], vm[eid])

        if tr_id == 0:
            vxt = tx1
            vyt = ty1
            vzt = tz1
            vet = te1
            vtt = tt1

            bxt = bx[ev][0]
            byt = by[ev][0]
            bzt = bz[ev][0]
            bet = be[ev][0]
            btt = bt[ev][0]
        else:
            vxt = tx2
            vyt = ty2
            vzt = tz2
            vet = te2
            vtt = tt2    

            bxt = bx[ev][1]
            byt = by[ev][1]
            bzt = bz[ev][1]
            bet = be[ev][1]   
            btt = bt[ev][1]
    #ax.scatter(vxt, vyt, vzt, s=vet*bubble_size, alpha=0.2, c="m", zorder=-5)  
    else:
        vxt = vx[ev][tr_id]
        vyt = vy[ev][tr_id]
        vzt = vz[ev][tr_id]
        vet = ve[ev][tr_id]
        vtt = vt[ev][tr_id]
        #vit = vi[ev][tr_id]

        bxt = bx[ev][tr_id]
        byt = by[ev][tr_id]
        bzt = bz[ev][tr_id]
        bet = be[ev][tr_id]
        btt = bt[ev][tr_id]
    # Get distances between the LCs
    dist_matrix = distance_matrix(vxt, vyt, vzt)
    # Calculate PCA
    pca = WPCA(n_components=3)    
    positions = np.array((vxt, vyt, vzt)).T

    vet_array = np.array(vet).reshape(-1,1)
    vet_array = np.tile(vet_array,(1, 3))

    pca.fit(positions, weights = vet_array)
    component = pca.components_[0]
    
    # Barycenter
    b_coord = np.array([bxt, byt, bzt])
    edges.add((tuple(b_coord), tuple(b_coord)))
    nodes.add(tuple(b_coord))
    times.append(np.array([bxt, byt, bzt, btt]))
    ax.scatter(bxt, byt, bzt, s=10, c="red", label=f"{len(vet)} LC: {sum(vet):.2f} GeV")

    segment_end = np.array([bxt+component[0],
                            byt+component[1],
                            bzt+component[2]])

    data = [[x,y,z,e,i,j] for x,y,z,e,i,j in zip(vxt, vyt, vzt, vet, vtt, range(len(vxt)))]
    # sorted by energy
    data = sorted(data, key=lambda a: -a[3])

    # project each LC to the principal component
    min_p, max_p = b_coord, b_coord
    max_en = max(vet)
    num_lc_above_th = len(np.array(vxt)[np.array(vet) > 2*ENERGY_RATIO_THR*max_en])
    
    small_trackster = False
    if max_en < 1 or num_lc_above_th < 5:
        # If maximum energy of the LCs is lower than 2 GeV or number of layer clusters with at least a ENERGY_RATIO_THR of the maximum energy is less than 5
        small_trackster = True
        print("Small trackster")

    min_point = np.zeros(4)
    max_point = np.zeros(4)
    # try create main edges
    
    for x,y,z,e,i,j in data:
        # plot times
        #if i > -80:
        #    ax.text(x, y, z, '%.3f ns' % (i), size=12) #+ (x**2+y**2+z**2)**0.5/C
        
        point = np.array([x, y, z, i])
        dist, closest_point = project_lc_to_pca(point[:3], b_coord, segment_end)
        
        if e/max_en > ENERGY_RATIO_THR and dist < PCA_NEIGHBOUR_THR:
            # limiting the PCA length
            z_cl = closest_point[2]
            if min_p[2] > z_cl and point[3]>-80:
                min_p = closest_point
                min_point = point
            if max_p[2] < z_cl and point[3]>-80:
                max_p = closest_point  
                max_point = point  
        
    if np.linalg.norm(min_p-max_p) > 0.5:
    
        if not np.allclose(min_p, b_coord, atol=0.1):
            edges.add((tuple(min_p), tuple(b_coord)))
            nodes.add(tuple(min_p))
            times.append(min_point)
            ax.plot([bxt, min_p[0]], [byt, min_p[1]], [bzt, min_p[2]], c='red')    

        if not np.allclose(max_p, b_coord, atol=0.1):
            edges.add((tuple(max_p), tuple(b_coord)))
            nodes.add(tuple(max_p))
            times.append(max_point)
            ax.plot([bxt, max_p[0]], [byt, max_p[1]], [bzt, max_p[2]], c='green')
    else:
        min_p, max_p = b_coord, b_coord
        min_point[:3], max_point[:3] = b_coord, b_coord
        min_point[3], max_point[3] = btt, btt
        
    print('times of the nearest LCs: {:.3f} ns, {:.3f} ns'.format(min_point[3], max_point[3]))
    ax.scatter(min_point[0], min_point[1], min_point[2], s=20, alpha=0.8, c="r", zorder=-5)   
    ax.scatter(max_point[0], max_point[1], max_point[2], s=20, alpha=0.8, c="r", zorder=-5)
    #ax.text(min_point[0], min_point[1], min_point[2], '%.3f ns' % (min_point[3]), size=12)   
    #ax.text(max_point[0], max_point[1], max_point[2], '%.3f ns' % (max_point[3]), size=12)
    
    # speed of the particle (?)
    min_speed = np.linalg.norm(min_point[:3] - b_coord) / abs(min_point[3] - btt)
    max_speed = np.linalg.norm(max_point[:3] - b_coord) / abs(max_point[3] - btt)
    print('speed of particle along PCA segments: {:.3f} cm/ns, {:.3f} cm/ns'.format(min_speed, max_speed))
    speed = np.linalg.norm(max_point[:3] - min_point[:3]) / abs(max_point[3] - min_point[3])
    print('speed of particle: {:.3f} cm/ns'.format(speed))

    # compute times
    dist4t = ((min_p[0] - b_coord[0])**2 + (min_p[1] - b_coord[1])**2 + (min_p[2] - b_coord[2])**2)**0.5
    time1 = (dist4t/29.9792458)
    
    dist4t = ((max_p[0] - b_coord[0])**2 + (max_p[1] - b_coord[1])**2 + (max_p[2] - b_coord[2])**2)**0.5
    time2 = (dist4t/29.9792458)
    
    # plot times
    if abs(max_p[2])-abs(min_p[2])>0:
        ax.text(min_p[0], min_p[1], min_p[2], 'PCA: %.3f ns\nLC: %.3f ns' % (btt-time1, min_point[3]), size=12) 
        ax.text(b_coord[0], b_coord[1], b_coord[2], '%.3f ns' % (btt), size=12)
        ax.text(max_p[0], max_p[1], max_p[2], 'PCA: %.3f ns\nLC: %.3f ns' % (btt+time2, max_point[3]), size=12)
        print('times at the speed of light: {:.3f} ns, {:.3f} ns'.format(btt-time1, btt+time2))
    else:
        ax.text(min_p[0], min_p[1], min_p[2], 'PCA: %.3f ns\nLC: %.3f ns' % (btt+time1, min_point[3]), size=12) 
        ax.text(b_coord[0], b_coord[1], b_coord[2], '%.3f ns' % (btt), size=12)
        ax.text(max_p[0], max_p[1], max_p[2], 'PCA: %.3f ns\nLC: %.3f ns' % (btt-time2, max_point[3]), size=12)
        print('times at the speed of light: {:.3f} ns, {:.3f} ns'.format(btt-time2, btt+time1))
    
    #Build secondary edges
    for x,y,z,e,i,j in data:
        
        if e / max_en > ENERGY_RATIO_THR:
            # and not covered by any edge
            point = np.array([x, y, z, i])
            min_dist, min_edge, closest_point_edge, closest_endpoint =  find_dist_to_closest_edge_set(point[:3], edges)
            #print(min_dist)
            
            if min_dist < DST_THR:
                # LC covered by some edge already
                ax.scatter(x, y, z, s=e*bubble_size, alpha=0.2, c="green", zorder=-5)
                covered_nodes.add((x, y, z))
                covered_node_idx.append(j)  
                continue
            else:
                ax.scatter(x, y, z, s=e*bubble_size, alpha=0.2, c="black", zorder=-5)
                pass

            if not small_trackster and secondary_edges:
                # find neares higher
                # go through all points, if the point is within the Neighbour_threshold
                distances = dist_matrix[j]
                indices = np.argsort(distances)

                for idx in indices:

                    if idx in covered_node_idx and distances[idx] < NEIGHBOUR_THR and vet[idx] > e and abs(vzt[idx]) < abs(z):
                        # found the nearest higher
                        # create an edge - to projection of the nearest higher to the pca
                        #edges[(idx, j)] = (np.array([vxt[idx], vyt[idx], vzt[idx]]), point)
                        ax.plot([vxt[idx], x], [vyt[idx], y], [vzt[idx], z], c='blue')
                        nearest_higher = np.array([vxt[idx], vyt[idx], vzt[idx]])
                        
                        # TODO: check that it lies on the axis!!!!!!!!!!!!!!!
                        dist, closest_point = project_lc_to_pca(nearest_higher, b_coord, segment_end)

                        if not np.allclose(closest_point, point[:3], atol=0.1):
                            edges.add((tuple(closest_point), tuple(point[:3])))
                            nodes.add(tuple(closest_point))
                            nodes.add(tuple(point[:3]))
                            times.append(point)
                        ax.plot([closest_point[0], x], [closest_point[1], y], [closest_point[2], z], c='blue')

                        break
        else:
            # if low energy
            pass
            ax.scatter(x, y, z, s=e*bubble_size, alpha=0.2, c="blue", zorder=-5)
    
    edges.remove((tuple(b_coord), tuple(b_coord)))
    set_axes_equal(ax)
    plt.show()
    return edges, nodes, covered_node_idx, times