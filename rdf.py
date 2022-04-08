import ase
import numpy as np
import numpy.linalg
from scipy.signal import fftconvolve

import math
import pandas as pd

from typing import List, Dict, Tuple
from dataclasses import dataclass
from collections import namedtuple
from functools import reduce

Point = namedtuple("Point", ['spec', 'idx', 'pos'])
Distances : Tuple[Point, pd.DataFrame] \
    = namedtuple("Distances", ['origin', 'dists', 'specs', 'idxs'])
DistancesByCenter : Tuple[float, List[str], List[int], List[Distances]] = \
    namedtuple("DistancesByCenter", ["cutoff_radius", "species", "indices", "distances_by_center"])

RDF_struc = namedtuple("RDF_struc", ["description", "binsize", "data", "origin"])
FP_dict_raw = Dict[str, Dict[str, RDF_struc]]
FP_tuple_flat = Tuple[List[str], np.array]

def downsample_rdf(rdf : RDF_struc, target_bin):
    pass

# weight functions
def w_stanley_density(center, far):
    return far

def w_product(center, far):
    return center*far

def w_square(center, far):
    return far*far

def w_unit(center, far):
    return 1

# there was a class for calculating the RDF structure
# in commit 268a64d71ccd01cf249b42109264f5349e3806c6,
# changed to joblib now for no-brain-caching
# removed joblib after speeding up with numpy https://shocksolution.com/microfluidics-and-biotechnology/calculating-the-pair-correlation-function-in-python/
# http://www.physics.emory.edu/faculty/weeks//idl/gofr2.html

def _get_covering_repetitions(base_epiped : np.array,
                              radius : float,
                              origin=np.array([0,0,0])):
    repetition_dirs = [[True, False, False],
                       [False, True, False],
                       [False, False, True]]

    reps = []
    base_epiped = np.array(base_epiped)
    origin = np.array(origin)

    # get linear factor for shifting the epiped around origin
    linf = np.dot(origin, np.linalg.inv(base_epiped))
    #print(linf)
    
    for selection_vector, vectorstep in zip(repetition_dirs, linf):
        # get the vector along whihc to calculate the epiped repetitions
        rep_vector = base_epiped[selection_vector][0]
        bounding_plane_v1, bounding_plane_v2 =  base_epiped[np.logical_not(selection_vector)]
        #print("\n\nSelected:", rep_vector)
        #print("BP", bounding_plane_v1, bounding_plane_v2)
        plane_normal = np.cross(bounding_plane_v1, bounding_plane_v2)
        #print(plane_normal)
        # get the angle between the plane spanned by the remaining vectors and rep_vector
        cos_plane2rep = np.dot(plane_normal, rep_vector)/(np.linalg.norm(plane_normal)*np.linalg.norm(rep_vector))
        # we want to find the angle of the right triangle defined by
        # A. the center of a sphere of radius `radius`
        # B. the Intersection of the tangential plane (defined by plane_normal) and the sphere
        # C. the point where the line defined by (A) and rep_vector cuts the plane (B)
        # this angle is <90°, thus independent of the exact alignment of vectors
        cos_triangle = abs(cos_plane2rep)
        # thus we get for the length of line segment AC
        rep_tot_length = radius/cos_triangle
        #print("rep_length: ",  rep_tot_length)
        # now calculate, how many repetitions one needs
        # and adjust for sphere origin (basically add more repetitions)
        rep_axis_length = np.linalg.norm(rep_vector, ord=2)
        #print(origin, rep_vector)
        center_proj = np.dot(origin, rep_vector/rep_axis_length)
        # nasty for stuff just about outside the cell
        ext_max = rep_tot_length+vectorstep*rep_axis_length
        ext_min = -rep_tot_length+vectorstep*rep_axis_length
        #print("max/min:", ext_max, ext_min)
        #print(ext_max/rep_axis_length, ext_min/rep_axis_length)
        # FIXING: a strange rounding error, where for pathological combinations the ceil/floor would remove bits
        max_rep = math.ceil(ext_max/rep_axis_length+1e-9)
        min_rep = math.floor(ext_min/rep_axis_length-1e-9)
        reps.append((min_rep, max_rep))
        #print("rep", min_rep, max_rep)
    return reps


def _repetitionranges_to_translations(repetition_ranges : List[Tuple[int,int]],
                                      base_cell : np.array,):
    """ translate the covering repetition ranges 
    to suitable translations with the base cell
    """
    ax1_range, ax2_range, ax3_range = repetition_ranges
    ax1 = base_cell[0]
    ax2 = base_cell[1]
    ax3 = base_cell[2]
    grid_size = len(range(*ax1_range))*len(range(*ax2_range))*len(range(*ax3_range))
    shifts = np.zeros((grid_size, 3), dtype=np.float)
    curr_idx = 0
    for ax1_mult in range(*ax1_range):
        for ax2_mult in range(*ax2_range):
            for ax3_mult in range(*ax3_range):
                cell_shift = ax1_mult*ax1 + ax2_mult*ax2 + ax3_mult*ax3
                shifts[curr_idx] = cell_shift
                curr_idx += 1
    return shifts

def _get_epiped_circumsphere(base_epiped : np.array) -> (float, np.array):
    base_epiped = np.array(base_epiped)
    center = 0.5*(base_epiped[0] + base_epiped[1] + base_epiped[2])
    corner_dists = [np.linalg.norm(x-center) for x in [np.array([0,]*3), base_epiped[0], base_epiped[1], base_epiped[2]]]
    return max(corner_dists), center

def _cull_translations_to_origin(translation_centers : np.array,
                                 radius : float,
                                 origin: np.array):
    return np.linalg.norm(translation_centers-origin, axis=1) <= radius


def _get_pointdistances(struc: ase.Atoms, cutoff_radius: float, override_spec={}) -> DistancesByCenter:
    """
    Get distances of all the other atoms for a crystal within a specific radius
    
    :param struc: thought to be an ase.Atoms, in practice needs a .cell, .positions, .get_chemical_symbols and per element .index-method if you have some other datastruc
    :param cutoff_radius: up to which radius to count
    :param override_spec: hocus-pocus for ase, where I do course-graining by putting random atoms noone needs (Pu and Co.) into the atoms-struc. Allows to pass in the weights in a sane way later on.
    """
    cutoff_radius = cutoff_radius
    override_spec = override_spec

    base_positions = struc.positions
    base_specs = [override_spec.get(s,s) for s in struc.get_chemical_symbols()]
    base_indices = [a.index for a in struc]
    base_cell = struc.cell
    base_size = len(base_indices)

    uc_origins = [Point(s, i, p) for s, i, p in zip(base_specs, base_indices, base_positions)]

    axis_rep_range = [(0,0),]*3

    # find the epiped covering all the RDFs
    # this might not be the best approach for big cells and small radii
    #for center_point in uc_origins:
    #    axis_rep_range_new = _get_covering_repetitions(base_cell, cutoff_radius, center_point.pos)
    #    axis_rep_range = [(min(x[0][0],x[1][0]), max(x[0][1],x[1][1]))
    #                      for x in zip(axis_rep_range, axis_rep_range_new)]

    # zeros > tiling >> subselections > masked arrays
    # -> do everything with extra tiled arrays, memory is of no concern
    cell_radius, cell_center = _get_epiped_circumsphere(base_cell)
    distances_by_center = []
    # TODO: make a function?
    for center_point in uc_origins:
        center_point_pos = center_point.pos
        axis_rep_range = _get_covering_repetitions(base_cell, cutoff_radius, center_point_pos)
        # get the necessary translations and cull all unnecessary ones
        trans_vectors = _repetitionranges_to_translations(axis_rep_range, base_cell)
        #print(trans_vectors.shape)
        cull_vector = _cull_translations_to_origin(trans_vectors+cell_center,
                                                   cell_radius + cutoff_radius,
                                                   center_point_pos)
        trans_vectors = trans_vectors[cull_vector]
        #print(trans_vectors.shape)
        cellgrid_size = len(trans_vectors)
        total_size = cellgrid_size*base_size
        
        # this stores the shifts for every cell, tiling with
        # TODO: TEST THAT TILE BEHAVES LIKE THIS
        shifts = np.tile(trans_vectors, (base_size,))
        shifts.shape = (total_size, 3)
        #print(shifts[::base_size])

        # create all points to consider
        points = np.tile(base_positions, (cellgrid_size, 1))
        points = points + shifts
        specs = np.tile(np.array(base_specs), cellgrid_size)
        idxs = np.tile(np.array(base_indices), cellgrid_size)

        # preallocated distances
        dists = np.zeros(total_size, dtype=np.float)
        dists = np.linalg.norm(points - center_point_pos, axis=1)
        
        # now "finely" cull everything out of scope
        # masked arrays might be faster?
        in_cutoff = dists < cutoff_radius
        dists = dists[in_cutoff]
        specs = specs[in_cutoff]
        idxs = idxs[in_cutoff]

        distances_by_center.append(Distances(origin=center_point, dists=dists, specs=specs, idxs=idxs))

    return DistancesByCenter(cutoff_radius=cutoff_radius,
                             species=set(base_specs),
                             indices=set(base_indices),
                             distances_by_center=distances_by_center)


def _bin_single_dist(single_dists : Distances,
                     cutoff_radius : float, binsize : float,
                     limit_atoms : list, limit_idx : list,
                     gaussian, norm_vol,
                     weights, weight_function) -> np.array:
    """
    Takes in a single distances objects and creates a binned RDF.

    BE CAREFUL: numpy histogram might introduce subtle rounding errors with small weigths

    :param cutoff_radius: the cutoff_radius. Determines the length of the histogram
    :param binsize: binsize
    :param limit_spec: for each distance-center, only calculate the RDF center→limit_spec
    :param limit_spec: for each distance-center, only calculate the RDF center→limit_idx
    :param gaussian: don't count only in one bin, but add a gaussian function with FWHM of about "gaussian"
    :param norm_vol: norm the bins by a property of the sphere-shell, where the points are included. (currently: "3D", "count" or None)
    :param weights: weights for each species, weights for a pair are given to weight_function
    :param weight_function: a weight_function, calculating the RDF contribution between two points. Takes as input the species-specific elements of the weights array. The first argument is the centerpoint of the binned-dist, the second the 
    """
    
    binned = np.zeros(math.ceil(cutoff_radius/binsize))

    specs = single_dists.specs
    idxs = single_dists.idxs
    distances = single_dists.dists

    #bulk_check for the spec/idx-limitation.
    #TODO: speed up? (doubles exec time!)
    if limit_atoms:
        inspec = np.vectorize(lambda e: e in limit_atoms)
        specs_to_count = inspec(specs)
    if limit_idx:
        inidxs = np.vectorize(lambda e: e in limit_idx)
        idxs_to_count = inidxs(idxs)

    if limit_idx or limit_atoms:
        if limit_idx and limit_atoms:
            limiter = np.logical_and(specs_to_count, idxs_to_count)
        elif limit_idx:
            limiter = idxs_to_count
        elif limit_atoms:
            limiter = specs_to_count
        specs = specs[limiter]
        idxs = idxs[limiter]
        distances = distances[limiter]

    # calculate the bin for each distances in distances
    #binidxs = np.array(distances//binsize, dtype=np.int)
    origin = single_dists.origin

    # 2x faster
    binedges = np.arange(0, len(binned)+1)*binsize
    if weights and weight_function:
        vecweigths = np.vectorize(lambda spec: weight_function(weights[origin.spec],
                                                               weights[spec]),
                                  otypes=[float])
        weights = vecweigths(specs)
        binned = numpy.histogram(distances, bins=binedges, weights=weights)[0]
    else:
        #print("no_weights")
        binned = numpy.histogram(distances, bins=binedges)[0]
        binned = binned.astype(float)

    if norm_vol == "count":
        bin_count = numpy.histogram(distances, bins=binedges)[0]
        bin_count[bin_count==0] = 1 #so we don't have /0
        binned = binned/bin_count


    binned_gaussian = None
    binextent = None
    if gaussian:
        # calculate the binned gaussian
        fwhm = gaussian
        extent = 3*fwhm
        binextent = math.ceil(extent/binsize)
        gaussian_range = np.arange(-binextent*binsize, (binextent+1)*binsize, step=binsize)
        binned_gaussian = np.exp(-0.5*(gaussian_range/(fwhm/2.355))**2)
        binned = np.convolve(binned_gaussian, binned, "same")
        #binned = fftconvolve(binned_gaussian, binned, "same")

    if norm_vol == "3D":
        binned_idx = np.array(list(range(len(binned))))
        binned = binned/(4/3*math.pi*(((binned_idx+1)*binsize)**3-(binned_idx*binsize)**3))


    return binned


def _bin_dists(distances : DistancesByCenter, binsize : float,
               limit_spec : List[str] = None, limit_idx : List[int] = None,
               gaussian : int = None, norm_vol="3D",
               weights : Dict[str, float] = None, weight_function=None) -> List[RDF_struc]:
    """
    Takes in distances (an DistancesByCenter object), and subsequently bins them (up to the radius specified in DistancesByCenter).
    This is basically an RDF, but keep in mind, that this function outputs _per center_ in distances, so for a general RDF you have to add all the outputs.
    Currently "i{}/{}_to_{}".format(pdist.origin.idx, pdist.origin.spec, desc) is set to describe the resulting RDF_struc's.
    Wraps _bin_single_dist for actual binning.
    
    :param binsize: binsize
    :param limit_spec: for each distance-center, only calculate the RDF center→limit_spec
    :param limit_spec: for each distance-center, only calculate the RDF center→limit_idx
    :param gaussian: don't count only in one bin, but add a gaussian function with FWHM of about "gaussian"
    :param norm_vol: norm the bins by a volume of the sphere-shell, where the points are included. (currently: "3D" or None)
    :param weights: weights for each species, weights for a pair are given to weight_function
    :param weight_function: a weight_function, calculating the RDF contribution between two points. Takes as input the species-specific elements of the weights array
    """
    name = None
    desc = ""
    cutoff_radius = distances.cutoff_radius
    if not limit_spec:
        limit_spec = None #set(distances.species)
        desc += "allspec"
    else:
        desc += "-".join([str(s) for s in limit_spec])

    if not limit_idx:
        limit_idx = None #set(distances.indices)
        desc += "_allidx"
    else:
        desc += "-".join([str(i) for i in limit_idx])

    bin_dists = []
    for pdist in distances.distances_by_center:
        name = "i{}/{}_to_{}".format(pdist.origin.idx, pdist.origin.spec, desc)
        histo = _bin_single_dist(pdist,
                                 cutoff_radius, binsize,
                                 limit_spec, limit_idx,
                                 gaussian, norm_vol,
                                 weights, weight_function)
        bin_dists.append(RDF_struc(description=name, binsize=binsize, data=histo, origin=pdist.origin))
    return bin_dists


def _rdf_partial(distances : DistancesByCenter, binsize : float,
                 collect_origins=True, gaussian=None, norm_vol="3D") -> List[RDF_struc]:
    """
    Calculate the partial rdf between species, return a list of RDF_strucs.
    Basically a wrapper around _bin_dists.
    Output RDF_struc.description is set to "rdf-speciesA-speciesB" (and VV).
    
    :param distances: a DistancesByCenter-object
    :param binsize: the binsize of the discretized RDF
    :param collect_origins: partial RDF sorted by same species (with multiple species added) when True, otherwise per index
    :param gaussian: gaussian spreading to get less zeros, enter a FWHM-value
    :param norm_vol: how to normalize the RDF for larger-radii 
    """
    partial_dists = []
    unames = distances.species
    for spec in unames:
        limited_dists = _bin_dists(distances, binsize, limit_spec=[spec,], gaussian=gaussian, norm_vol=norm_vol)
        if collect_origins:
            limited_dists_collected = []
            for ospec in unames:
                try:
                    one_species_limited_dist = sum([r.data for r in limited_dists if r.origin.spec == ospec])
                    limited_dists_collected.append(RDF_struc(
                        description="rdf-{}-{}".format(ospec, spec),
                        binsize=binsize,
                        data=one_species_limited_dist,
                        origin="{}".format(ospec)
                    ))
                except ValueError:
                    print([len(r.data) for r in limited_dists if r.origin.spec == ospec])
                    print(ospec)
                    print(unames)
                    raise ValueError
            partial_dists.extend(limited_dists_collected)
        else:
            partial_dists.extend(limited_dists)
    return partial_dists

def calc_partial_fp(struc: ase.Atoms,
                    override_spec={},
                    radius=10,
                    binsize=0.1,
                    gaussian=None,
                    norm_vol="3D",) -> Dict[str, RDF_struc]:
    """ calculate partial fingerprints and output a dictionary.
    for parameter explanation checkout _get_pointdistances and _rdf_partial/_bin_single_dists

    :return: a dictionary, with keys of the form "speciesA-speciesB" for the corresponding RDF_struc
    """
    raw_distances = _get_pointdistances(struc, radius, override_spec)
    #return raw_distances
    fps = _rdf_partial(raw_distances, binsize, gaussian=gaussian, norm_vol=norm_vol)
    fpdict = dict([(fp.description[4:], fp) for fp in fps])
    return fpdict


def calc_property_fp(
        struc: ase.Atoms,
        cumulated_fp=False,
        radius=10,
        binsize=0.1,
        gaussian=None,
        norm_vol="3D",
        override_spec={},
        properties : Dict[str, Dict] = {},
        property_mixer=lambda a, b: a*b) -> Dict[str, RDF_struc]:
    """
    calculate the property-RDF as outlined by Stanley et. al. (2019).
    If no properties are specified, just does a general RDF
    
    :param struc: the input structure
    :param cumulated_fp: cumulate the fingerprint data (e.g. cumsum over the histogram)
    :param radius: the RDF cutoff radius
    :param binsize: the binning of the RDF_struc
    :param gaussian: apply gaussian-FHWM spreading to the discrete location
    :param norm_vol: nromalize the binned rdf to spherical-shell-volumes
    :param override_spec: use in conjunction with a "nice"-properties-array, when the input struc is "coarse-grained" (using Pu,X,... as coarse grains in ase)
    :param properties: properties to use for the property-RDF, input a dict of properties for multiple ones
    :property_mixer: how to weight the RDF-elements

    :return: a dictionary<name of the property (or generic, if properties was empty
    """
    # mangle the properties to the specified format
    formatted_properties = None
    if properties and isinstance(properties, dict):
        property_types = set([type(t) for t in properties.values()])
        if len(property_types) == 1 and (dict in property_types):
            formatted_properties = properties
        else:
            formatted_properties = {"property" : properties}
    elif isinstance(properties, dict):
        formatted_properties = {"unit" : properties}

    if not formatted_properties:
        raise TypeError("something wrong with the input")

    raw_distances = _get_pointdistances(struc, radius, override_spec)

    fpdict = {}
    for p, p_values in formatted_properties.items():
        fps = _bin_dists(
            raw_distances,
            binsize=binsize,
            gaussian=gaussian,
            norm_vol=norm_vol,
            weights=p_values,
            weight_function=property_mixer
        )
        rdf_data = np.zeros(len(fps[0].data))
        for centered_rdf in fps:
            rdf_data += centered_rdf.data
        rdf_data /= len(fps)

        if cumulated_fp:
            rdf_data = np.cumsum(rdf_data)
        
        rdf = RDF_struc(
            description="prdf-{}".format(p),
            binsize=binsize,
            data=rdf_data,
            origin=None
        )
        fpdict[p] = rdf
    return fpdict
