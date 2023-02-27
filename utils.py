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
