import unittest
import timeit
import cProfile

import numpy as np
import numpy.testing as npt

# stuff for fixtures/mangling
import os
import json

import ase
import ase.build

# the actual tested module
import rdf

def np_compare(test, correct, exact=True):
    try:
        if isinstance(exact, bool) and exact == True:
            npt.assert_array_equal(test, correct)
        elif isinstance(exact, int):
            npt.assert_array_almost_equal(test, correct, decimal=exact)
        else:
            npt.assert_array_almost_equal(test, correct)
        return True
    except AssertionError:
        return False


class TestHelperRoutines(unittest.TestCase):
    def setUp(self):
        # cuboid for testing the covering
        self.cuboid = np.array([[1,0,0],
                                [0,1,0],
                                [0,0,2]])
        self.cuboid_radius_0 = 2.8
        
        self.cuboid_zero_0 = np.array([0,0,0])
        self.cuboid_zero_1 = np.array([-3, -4, -5])
        self.cuboid_zero_2 = np.array([1,1,2])

        self.cuboid_covering_z0_r0 = [(-3,3), (-3,3), (-2,2)]
        self.cuboid_covering_z1_r0 = [(-6,0), (-7,-1), (-4,-1)]
        self.cuboid_covering_z2_r0 = [(-2,4), (-2,4), (-1,3)]

        self.cuboid_radius_1 = 1

        self.cuboid_covering_z0_r1 = [(-2,2), (-2,2), (-1,1)]
        self.cuboid_covering_z1_r1 = [(-5,-1), (-6,-2), (-4,-1)]
        self.cuboid_covering_z2_r1 = [(-1,3), (-1,3), (0, 2)]

        # these were visually confirmed and generated with code from @2020-07-08
        self.epiped = np.array([[1,1,0],[0,2,0],[0,2,3]])
        self.epiped_radius_0 = 8.2
        self.epiped_radius_1 = 0.71

        self.epiped_zero_0 = np.array([0,0,0])
        self.epiped_zero_1 = np.array([-3, -4, -5])
        self.epiped_zero_2 = np.array([1.4,1.1,2.051651])

        self.epiped_covering_z0_r0 = [(-9, 9), (-7, 7), (-3, 3)]
        self.epiped_covering_z1_r0 = [(-12, 6), (-6, 8), (-5, 2)]
        self.epiped_covering_z2_r0 = [(-7, 10), (-8, 6), (-3, 4)]

        self.epiped_covering_z0_r1 = [(-1, 1), (-1, 1), (-1, 1)]
        self.epiped_covering_z1_r1 = [(-4, -2), (0, 2), (-2, -1)]
        self.epiped_covering_z2_r1 = [(0, 3), (-2, 0), (0, 1)]

        # add: test np.tile
        self.cuboid_rep0 = [(-1, 1), (-1, 1), (-1, 1)]
        self.cuboid_repvec0 = np.array([[-1., -1., -2.],
                                        [-1., -1.,  0.],
                                        [-1.,  0., -2.],
                                        [-1.,  0.,  0.],
                                        [ 0., -1., -2.],
                                        [ 0., -1.,  0.],
                                        [ 0.,  0., -2.],
                                        [ 0.,  0.,  0.]])
        self.cuboid_rep1 = [(-4,-3), (10, 13), (-1, 0)]
        self.cuboid_repvec1 = np.array([[-4., 10., -2.],
                                        [-4., 11., -2.],
                                        [-4., 12., -2.]])
        self.epiped_rep0 = [(-1, 1), (-1, 1), (-1, 1)]
        self.epiped_repvec0 = np.array([[-1., -5., -3.],
                                        [-1., -3.,  0.],
                                        [-1., -3., -3.],
                                        [-1., -1.,  0.],
                                        [ 0., -4., -3.],
                                        [ 0., -2.,  0.],
                                        [ 0., -2., -3.],
                                        [ 0.,  0.,  0.]])
            
        self.epiped_rep1 = [(-1, 2), (-1, 1), (10, 12)]
        self.epiped_repvec1 = np.array([[-1., 17., 30.],
                                        [-1., 19., 33.],
                                        [-1., 19., 30.],
                                        [-1., 21., 33.],
                                        [ 0., 18., 30.],
                                        [ 0., 20., 33.],
                                        [ 0., 20., 30.],
                                        [ 0., 22., 33.],
                                        [ 1., 19., 30.],
                                        [ 1., 21., 33.],
                                        [ 1., 21., 30.],
                                        [ 1., 23., 33.]])
        self.rep_cells = [
            (self.cuboid, self.cuboid_rep0, self.cuboid_repvec0),
            (self.cuboid, self.cuboid_rep1, self.cuboid_repvec1),
            (self.epiped, self.epiped_rep0, self.epiped_repvec0),
            (self.epiped, self.epiped_rep1, self.epiped_repvec1),
            ]
        self.cuboid_circumsphere = (np.sqrt(1.5), np.array([0.5,0.5,1]),
                                    self.cuboid)
        self.epiped_circumsphere = (np.sqrt(8.75),
                                    np.array([0.5,2.5,1.5]),
                                    self.epiped)

        self.cull_radius = 1
        self.cull_origin = np.array([0,3,0])
        self.cull_vectors = np.array([
            [0,2,0],[0,51,-131],[100,0,30],[0.1,2.9,-0.7]
            ])
        self.cull_result = np.array([True, False, False, True])

    def test_cuboid_covering(self):
        c_z0_r0 = rdf._get_covering_repetitions(self.cuboid, self.cuboid_radius_0, self.cuboid_zero_0)
        self.assertEqual(c_z0_r0, self.cuboid_covering_z0_r0)
        c_z1_r0 = rdf._get_covering_repetitions(self.cuboid, self.cuboid_radius_0, self.cuboid_zero_1)
        self.assertEqual(c_z1_r0, self.cuboid_covering_z1_r0)
        c_z2_r0 = rdf._get_covering_repetitions(self.cuboid, self.cuboid_radius_0, self.cuboid_zero_2)
        self.assertEqual(c_z2_r0, self.cuboid_covering_z2_r0)
        c_z0_r1 = rdf._get_covering_repetitions(self.cuboid, self.cuboid_radius_1, self.cuboid_zero_0)
        self.assertEqual(c_z0_r1, self.cuboid_covering_z0_r1)
        c_z1_r1 = rdf._get_covering_repetitions(self.cuboid, self.cuboid_radius_1, self.cuboid_zero_1)
        self.assertEqual(c_z1_r1, self.cuboid_covering_z1_r1)
        c_z2_r1 = rdf._get_covering_repetitions(self.cuboid, self.cuboid_radius_1, self.cuboid_zero_2)
        self.assertEqual(c_z2_r1, self.cuboid_covering_z2_r1)

    def test_epiped_covering(self):
        c_z0_r0 = rdf._get_covering_repetitions(self.epiped, self.epiped_radius_0, self.epiped_zero_0)
        self.assertEqual(c_z0_r0, self.epiped_covering_z0_r0)
        c_z1_r0 = rdf._get_covering_repetitions(self.epiped, self.epiped_radius_0, self.epiped_zero_1)
        self.assertEqual(c_z1_r0, self.epiped_covering_z1_r0)
        c_z2_r0 = rdf._get_covering_repetitions(self.epiped, self.epiped_radius_0, self.epiped_zero_2)
        self.assertEqual(c_z2_r0, self.epiped_covering_z2_r0)
        c_z0_r1 = rdf._get_covering_repetitions(self.epiped, self.epiped_radius_1, self.epiped_zero_0)
        self.assertEqual(c_z0_r1, self.epiped_covering_z0_r1)
        c_z1_r1 = rdf._get_covering_repetitions(self.epiped, self.epiped_radius_1, self.epiped_zero_1)
        self.assertEqual(c_z1_r1, self.epiped_covering_z1_r1)
        c_z2_r1 = rdf._get_covering_repetitions(self.epiped, self.epiped_radius_1, self.epiped_zero_2)
        self.assertEqual(c_z2_r1, self.epiped_covering_z2_r1)

    def test_rep_to_translations(self):
        for cell, reps, vecs in self.rep_cells:
            repvecs = rdf._repetitionranges_to_translations(reps, cell)
            self.assertTrue(np_compare(repvecs, vecs))

    def test_epiped_circumsphere(self):
        for r, center, cell in (self.cuboid_circumsphere, self.epiped_circumsphere):
            fr, fcenter = rdf._get_epiped_circumsphere(cell)
            self.assertAlmostEqual(r, fr)
            self.assertTrue(np_compare(fcenter, center, exact=False))

    def test_cull(self):
        cull_res = rdf._cull_translations_to_origin(self.cull_vectors,
                                                    self.cull_radius,
                                                    self.cull_origin)
        self.assertTrue(np_compare(cull_res, self.cull_result))


class AtomMock():
    def __init__(self, index):
        self._index = index

    def __get_index(self):
        return self._index

    index = property(__get_index)

class AtomsMock():
    def __init__(self, pos, spec, cell):
        self._cell = np.array(cell)
        if len(spec) != len(pos):
            raise ValueError
        else:
            self._positions = pos
            self._spec = spec
            self._indices = list(AtomMock(i) for i in range(len(pos)))

    def __get_pos(self):
        return self._positions

    def __get_cell(self):
        return self._cell

    def get_chemical_symbols(self):
        return self._spec

    def __iter__(self):
        return iter(self._indices)

    positions = property(__get_pos)
    cell = property(__get_cell)



class TestPointDistances(unittest.TestCase):
    def setUp(self):
        # a simple, handcalculated case
        self.test_s = AtomsMock([[0, 1, 50],
                               [0, 2, 50],
                               [0, 3, 50]],
                              ["A", "B", "C"],
                              [[20,2,0],
                               [-20,2,0],
                               [0, 0, 100]]
                              )
        self.test_s1_radius = 9.9
        # should then get another point from another "row" included
        self.test_s2_radius = 20.01 
        # the following things are only calculated for point "B"
        self.test_s1B_start = rdf.Point(spec='B',
                                       idx=1,
                                       pos=np.array([0, 2, 50]))
        self.test_s1B_distances = np.array([0, 1, 1, 3, 3, 4, 4,
                                            5, 5, 7, 7, 8, 8, 9, 9])
        self.test_s1B_indices = np.array([1, 0, 2, 2, 0, 1, 1, 0, 2,
                                 2, 0, 1, 1, 0, 2])
        self.test_s1B_species = np.array(["B", "A", "C", "C",
                                 "A", "B", "B", "A", "C",
                                 "C", "A", "B", "B", "A", "C"])
        self.test_s1B_result = rdf.Distances(
            self.test_s1B_start,
            self.test_s1B_distances,
            self.test_s1B_species,
            self.test_s1B_indices)
            
        self.test_s2C_start = rdf.Point(spec='C',
                                       idx=2,
                                       pos=np.array([0, 3, 50]))
        self.test_s2C_distances = np.array([20., 18., 17., 16., 14., 13., 12., 10.,  9.,  8.,  6.,  5.,  4.,
        2.,  1.,  0., 20., 20.,  2.,  3.,  4.,  6.,  7.,  8., 10., 11.,
       12., 14., 15., 16., 18., 19., 20.])
        self.test_s2C_indices = np.array([2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 0, 0, 1, 2, 0,
       1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
        self.test_s2C_species = np.array(['C', 'A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C',
       'A', 'B', 'C', 'A', 'A', 'A', 'B', 'C', 'A', 'B', 'C', 'A', 'B',
       'C', 'A', 'B', 'C', 'A', 'B', 'C'], dtype='<U1')
        self.test_s2C_result = rdf.Distances(
            self.test_s2C_start,
            self.test_s2C_distances,
            self.test_s2C_species,
            self.test_s2C_indices)
        # TODO: add a 2D testcase with handcalculation
        
        # generated with code @2020-07-08
        self.test_crystal = ase.build.bulk("NaCl", 'rocksalt', a=5)
        self.test_crystal_radius = 5.7
        self.test_override = {'Na': 'Sodium', 'Cl': 'Chloride'}

        self.test_crystal_start0 = rdf.Point(spec='Na',
                                       idx=0,
                                       pos=np.array([0, 0, 0]))
        self.test_crystal_start0_repl = rdf.Point(spec='Natrium',
                                                  idx=0,
                                                  pos=np.array([0, 0, 0])) 
        self.test_crystal_start1 = rdf.Point(spec='Cl',
                                       idx=1,
                                       pos=np.array([2.5, 0, 0]))
        self.test_crystal_start1_repl = rdf.Point(spec='Chlorine',
                                                  idx=1,
                                                  pos=np.array([2.5, 0, 0]))

        self.test_crystal_distances0 = np.array(
            [5.59016994, 5.        , 5.59016994, 5.59016994, 3.53553391,
             4.33012702, 3.53553391, 5.59016994, 5.        , 5.59016994,
             3.53553391, 5.59016994, 5.        , 5.59016994, 5.59016994,
             4.33012702, 3.53553391, 2.5       , 3.53553391, 4.33012702,
             5.59016994, 3.53553391, 2.5       , 0.        , 2.5       ,
             3.53553391, 5.59016994, 5.59016994, 3.53553391, 4.33012702,
             3.53553391, 5.59016994, 5.59016994, 4.33012702, 5.59016994,
             5.59016994, 5.        , 2.5       , 3.53553391, 2.5       ,
             5.        , 5.59016994, 4.33012702, 3.53553391, 2.5       ,
             3.53553391, 4.33012702, 5.59016994, 5.        , 5.59016994]
        )
        self.test_crystal_distances1 = np.array(
            [5.59016994, 5.59016994, 5.59016994, 4.33012702, 5.59016994,
             5.59016994, 5.59016994, 5.59016994, 5.        , 5.59016994,
             4.33012702, 3.53553391, 2.5       , 3.53553391, 4.33012702,
             5.59016994, 5.        , 2.5       , 3.53553391, 2.5       ,
             5.        , 5.59016994, 5.59016994, 4.33012702, 5.59016994,
             5.59016994, 3.53553391, 4.33012702, 3.53553391, 5.59016994,
             5.59016994, 3.53553391, 2.5       , 0.        , 2.5       ,
             3.53553391, 5.59016994, 4.33012702, 3.53553391, 2.5       ,
             3.53553391, 4.33012702, 5.59016994, 5.59016994, 5.        ,
             5.59016994, 3.53553391, 5.59016994, 5.        , 5.59016994,
             3.53553391, 4.33012702, 3.53553391, 5.59016994, 5.59016994,
             5.        , 5.59016994]
        )

        self.test_crystal_indices0 = np.array(
            [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0,
             1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0,
             1, 0, 1, 1, 0, 1]
        )
        self.test_crystal_indices1 = np.array(
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0,
             0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0,
             1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0]
        )

        self.test_crystal_species0 = np.array(
            ['Cl', 'Na', 'Cl', 'Cl', 'Na', 'Cl', 'Na', 'Cl', 'Na', 'Cl', 'Na',
             'Cl', 'Na', 'Cl', 'Cl', 'Cl', 'Na', 'Cl', 'Na', 'Cl', 'Cl', 'Na',
             'Cl', 'Na', 'Cl', 'Na', 'Cl', 'Cl', 'Na', 'Cl', 'Na', 'Cl', 'Cl',
             'Cl', 'Cl', 'Cl', 'Na', 'Cl', 'Na', 'Cl', 'Na', 'Cl', 'Cl', 'Na',
             'Cl', 'Na', 'Cl', 'Cl', 'Na', 'Cl'], dtype='<U2'
        )
        self.test_crystal_species1 = np.array(
            ['Na', 'Na', 'Na', 'Na', 'Na', 'Na', 'Na', 'Na', 'Cl', 'Na', 'Na',
             'Cl', 'Na', 'Cl', 'Na', 'Na', 'Cl', 'Na', 'Cl', 'Na', 'Cl', 'Na',
             'Na', 'Na', 'Na', 'Na', 'Cl', 'Na', 'Cl', 'Na', 'Na', 'Cl', 'Na',
             'Cl', 'Na', 'Cl', 'Na', 'Na', 'Cl', 'Na', 'Cl', 'Na', 'Na', 'Na',
             'Cl', 'Na', 'Cl', 'Na', 'Cl', 'Na', 'Cl', 'Na', 'Cl', 'Na', 'Na',
             'Cl', 'Na'], dtype='<U2'
        )

        self.test_crystal_repl0= np.array(
            ['Chlorine', 'Natrium', 'Chlorine', 'Chlorine', 'Natrium', 'Chlorine', 'Natrium', 'Chlorine', 'Natrium', 'Chlorine', 'Natrium',
             'Chlorine', 'Natrium', 'Chlorine', 'Chlorine', 'Chlorine', 'Natrium', 'Chlorine', 'Natrium', 'Chlorine', 'Chlorine', 'Natrium',
             'Chlorine', 'Natrium', 'Chlorine', 'Natrium', 'Chlorine', 'Chlorine', 'Natrium', 'Chlorine', 'Natrium', 'Chlorine', 'Chlorine',
             'Chlorine', 'Chlorine', 'Chlorine', 'Natrium', 'Chlorine', 'Natrium', 'Chlorine', 'Natrium', 'Chlorine', 'Chlorine', 'Natrium',
             'Chlorine', 'Natrium', 'Chlorine', 'Chlorine', 'Natrium', 'Chlorine'], dtype='<U8'
        )
        self.test_crystal_repl1 = np.array(
            ['Natrium', 'Natrium', 'Natrium', 'Natrium', 'Natrium', 'Natrium', 'Natrium', 'Natrium', 'Chlorine', 'Natrium', 'Natrium',
             'Chlorine', 'Natrium', 'Chlorine', 'Natrium', 'Natrium', 'Chlorine', 'Natrium', 'Chlorine', 'Natrium', 'Chlorine', 'Natrium',
             'Natrium', 'Natrium', 'Natrium', 'Natrium', 'Chlorine', 'Natrium', 'Chlorine', 'Natrium', 'Natrium', 'Chlorine', 'Natrium',
             'Chlorine', 'Natrium', 'Chlorine', 'Natrium', 'Natrium', 'Chlorine', 'Natrium', 'Chlorine', 'Natrium', 'Natrium', 'Natrium',
             'Chlorine', 'Natrium', 'Chlorine', 'Natrium', 'Chlorine', 'Natrium', 'Chlorine', 'Natrium', 'Chlorine', 'Natrium', 'Natrium',
             'Chlorine', 'Natrium'], dtype='<U8'
        )

        self.test_crystal0 = rdf.Distances(
            self.test_crystal_start0,
            self.test_crystal_distances0,
            self.test_crystal_species0,
            self.test_crystal_indices0)

        self.test_crystal1 = rdf.Distances(
            self.test_crystal_start1,
            self.test_crystal_distances1,
            self.test_crystal_species1,
            self.test_crystal_indices1)

        self.test_crystal0_repl = rdf.Distances(
            self.test_crystal_start0_repl,
            self.test_crystal_distances0,
            self.test_crystal_repl0,
            self.test_crystal_indices0)

        self.test_crystal1_repl = rdf.Distances(
            self.test_crystal_start1_repl,
            self.test_crystal_distances1,
            self.test_crystal_repl1,
            self.test_crystal_indices1)

    def _check_equal_dists(self, dists_test, dists_ref):
        # first check that the origin point is the same
        o_test = dists_test.origin
        o_ref = dists_ref.origin
        self.assertEqual(o_test.spec, o_ref.spec)
        self.assertEqual(o_test.idx, o_ref.idx)
        self.assertTrue(np_compare(o_test.pos, o_ref.pos))
        # now pack everything into a pandas array to sort!
        

    def test_pathological_case1(self):
        dists = rdf._get_pointdistances(self.test_s, self.test_s1_radius).distances_by_center
        self._check_equal_dists(dists[1], self.test_s1B_result)
        pass

    def test_pathological_case2(self):
        dists = rdf._get_pointdistances(self.test_s, self.test_s2_radius).distances_by_center
        self._check_equal_dists(dists[2], self.test_s2C_result)
        pass

    def test_nacl(self):
        dists = rdf._get_pointdistances(self.test_crystal, self.test_crystal_radius).distances_by_center
        self._check_equal_dists(dists[0], self.test_crystal0)
        self._check_equal_dists(dists[1], self.test_crystal1)

    def test_nacl_repl(self):
        dists = rdf._get_pointdistances(self.test_crystal, self.test_crystal_radius,
                                        override_spec={"Cl" : "Chlorine", "Na" : "Natrium"}).distances_by_center
        self._check_equal_dists(dists[0], self.test_crystal0_repl)
        self._check_equal_dists(dists[1], self.test_crystal1_repl)


class TestSingleDistBin(unittest.TestCase):
    def setUp(self):
        self.data_origin = rdf.Point(spec="O",
                                     idx=0,
                                     pos=None)
        self.data_dists = np.array([0, 1, 1.7, 0.5, 17, 16.3, 19.2,])
        self.data_spec = np.array(["O", "B", "C", "A", "O",  "A", "B",])
        self.data_idxs = np.array([0, 2, 3,  1, 0, 4, 2])
        self.data_struc = rdf.Distances(self.data_origin,
                                        self.data_dists,
                                        self.data_spec,
                                        self.data_idxs)
        self.data_weights = {"O": 10, "A" : 5, "B" : -1, "C" : -0.15, "X" : 0}

        self.data_dists_bad = np.array([0, 1, 1.7, 0.5, 17, 16.3, 19.2, 100])
        self.data_spec_bad = np.array(["O", "B", "C", "A", "O",  "A", "B", "X"])
        self.data_idxs_bad = np.array([0, 2, 3,  1, 0, 4, 2, 5])
        self.data_struc_bad = rdf.Distances(self.data_origin,
                                            self.data_dists_bad,
                                            self.data_spec_bad,
                                            self.data_idxs_bad)
        # right order
        # 0, 0.5, 1, 1.7, 16.3, 17, 19.2
        # O, A, B, C, A, O, B
        # 0, 1, 2, 3, 4, 0, 2

        self.tt_radius = 20
        self.tt_binsize0 = 1
        self.tt_binsize1 = 0.3

        self.tt_limit_spec = set(["A","B"])
        self.tt_limit_idx = set([2, 0])
        self.tt_limit_all_empty = (set(["A",]), set([2,]))
        self.tt_stanley_weight = rdf.w_stanley_density

        self.res_binsize0 = np.zeros(20, dtype=float)
        self.res_binsize0[[0,1,16,17,19]] = [2,2,1,1,1]
        self.res_binsize1 = np.zeros(int(np.ceil(20/0.3)), dtype=float)
        self.res_binsize1[[0,1,3,5,54,56,64]] = [1,1,1,1,1,1,1]

        # all the other things for tt_binsize0 == 1
        self.res_limit_spec = np.zeros(20, dtype=float)
        self.res_limit_spec[[0,1,16,19]] = [1,1,1,1]
        self.res_limit_idx = np.zeros(20, dtype=float)
        self.res_limit_idx[[1,19,0,17,]] = [1,1,1,1]
        
        self.res_limit_idx_spec = np.zeros(20, dtype=float)
        self.res_limit_idx_spec[[1,19]] = [1,1]
        self.res_limit_all_empty = np.zeros(20, dtype=float)

        self.res_binsize1_normvol3d_vols = np.array([
            4/3*np.pi*(((i+1)*self.tt_binsize1)**3-(i*self.tt_binsize1)**3)
            for i in range(len(self.res_binsize1))])
        self.res_binsize1_normvol3d = self.res_binsize1/self.res_binsize1_normvol3d_vols

        # also the complete stanley pddf
        self.res_pddf = np.zeros(20, dtype=float)
        self.res_pddf[[0,1,16,17,19]] = [10+5, -1-0.15, 5, 10, -1]
        

    def test_size_error(self):
        """ The thing only works for distances < cutoff, everything else should be cut off """
        binned0 = rdf._bin_single_dist(self.data_struc_bad,
                                       self.tt_radius, self.tt_binsize0,
                                       limit_atoms=None, limit_idx=None,
                                       gaussian=None, norm_vol=None,
                                       weights=None, weight_function=None)
        self.assertTrue(np_compare(binned0, self.res_binsize0))
        #with self.assertRaises(IndexError):
        #    rdf._bin_single_dist(self.data_struc_bad,
        #                         self.tt_radius, self.tt_binsize0,
        #                         limit_atoms=None, limit_idx=None,
        #                         gaussian=None, norm_vol=None,
        #                         weights=None, weight_function=None)
        
        
    def test_binsizes(self):
        binned0 = rdf._bin_single_dist(self.data_struc,
                                       self.tt_radius, self.tt_binsize0,
                                       limit_atoms=None, limit_idx=None,
                                       gaussian=None, norm_vol=None,
                                       weights=None, weight_function=None)
        binned1 = rdf._bin_single_dist(self.data_struc,
                                       self.tt_radius, self.tt_binsize1,
                                       limit_atoms=None, limit_idx=None,
                                       gaussian=None, norm_vol=None,
                                       weights=None, weight_function=None)
        self.assertTrue(np_compare(binned0, self.res_binsize0))
        self.assertTrue(np_compare(binned1, self.res_binsize1))
        print("\nTODO: test large binsizes on a real system\n")

    def test_limit_spec(self):
        binned = rdf._bin_single_dist(self.data_struc,
                                      self.tt_radius, self.tt_binsize0,
                                      limit_atoms=self.tt_limit_spec, limit_idx=None,
                                      gaussian=None, norm_vol=None,
                                      weights=None, weight_function=None)
        self.assertTrue(np_compare(binned, self.res_limit_spec))

    def test_limit_idx(self):
        binned = rdf._bin_single_dist(self.data_struc,
                                      self.tt_radius, self.tt_binsize0,
                                      limit_atoms=None, limit_idx=self.tt_limit_idx,
                                      gaussian=None, norm_vol=None,
                                      weights=None, weight_function=None)
        self.assertTrue(np_compare(binned, self.res_limit_idx))

    def test_limit_all(self):
        binned_all = rdf._bin_single_dist(self.data_struc,
                                          self.tt_radius, self.tt_binsize0,
                                          limit_atoms=self.tt_limit_spec, limit_idx=self.tt_limit_idx,
                                          gaussian=None, norm_vol=None,
                                          weights=None, weight_function=None)
        binned_all_empty = rdf._bin_single_dist(self.data_struc,
                                                self.tt_radius, self.tt_binsize0,
                                                limit_atoms=self.tt_limit_all_empty[0], limit_idx=self.tt_limit_all_empty[1],
                                                gaussian=None, norm_vol=None,
                                                weights=None, weight_function=None)
        self.assertTrue(np_compare(binned_all, self.res_limit_idx_spec))
        self.assertTrue(np_compare(binned_all_empty, self.res_limit_all_empty))

    def test_norm_vol(self):
        binned1 = rdf._bin_single_dist(self.data_struc,
                                       self.tt_radius, self.tt_binsize1,
                                       limit_atoms=None, limit_idx=None,
                                       gaussian=None, norm_vol="3D",
                                       weights=None, weight_function=None)
        print("\nTODO: test the 'count' norm volume\n")
        self.assertTrue(np_compare(binned1, self.res_binsize1_normvol3d))
        

    def test_stanley_weights(self):
        binned = rdf._bin_single_dist(self.data_struc,
                                       self.tt_radius, self.tt_binsize0,
                                       limit_atoms=None, limit_idx=None,
                                       gaussian=None, norm_vol=None,
                                       weights=self.data_weights, weight_function=self.tt_stanley_weight)
        #print("\n", binned, "\n", self.res_pddf)
        #return binned, self.res_pddf
        # the np.histogram functionality introduces subtle numerical errors
        self.assertTrue(np_compare(binned, self.res_pddf, exact=10))

    def test_gaussian_filter(self):
        print("\nTODO: test the 'gaussian' binning\n")
        pass

    
    
def vis_verify_covering(base_epiped : np.array,
                        radius : float,
                        origin=np.array([0,0,0]),
                        draw_subtiles=False,
                        cull_subtiles=True):
    try:
        import matplotlib as mpl
        mpl.use("tkAgg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except:
        print("Interactive run required")
        return
    rep = rdf._get_covering_repetitions(base_epiped, radius, origin)
    print(rep)

    all_corners = np.array(
        [[0,0,0],
         [1,0,0],
         [1,1,0],
         [0,1,0],
         [0,0,1],
         [1,0,1],
         [1,1,1],
         [0,1,1]])

    
    def plot_epiped(base_point, cell, axes, verts=True,
                    full=True, draft=False):
        Z = all_corners @ cell
        Z += base_point
        if verts:
            ax.scatter(Z[:,0],
                       Z[:,1],
                       Z[:,2],)
          #https://stackoverflow.com/questions/44881885/python-draw-parallelepiped
                    
        faces = [[Z[0],Z[1],Z[2],Z[3]],
                 [Z[4],Z[5],Z[6],Z[7]], 
                 [Z[0],Z[1],Z[5],Z[4]], 
                 [Z[2],Z[3],Z[7],Z[6]], 
                 [Z[1],Z[2],Z[6],Z[5]],
                 [Z[4],Z[7],Z[3],Z[0]]]
        if full:
            ax.add_collection3d(Poly3DCollection(faces, 
                                                 facecolors='cyan',
                                                 linewidths=1,
                                                 edgecolors='r',
                                                 alpha=.04))
        elif draft:
            ax.add_collection3d(Poly3DCollection(faces, 
                                                 facecolors='cyan',
                                                 linewidths=0.5,
                                                 ls = '-',
                                                 edgecolors='b',
                                                 alpha=.0))
        else:
            ax.add_collection3d(Poly3DCollection(faces, 
                                                 facecolors='cyan',
                                                 linewidths=1,
                                                 edgecolors='r',
                                                 alpha=.0))
            

    base_epiped = np.array(base_epiped, dtype=float)
    origin = np.array(origin, dtype=float)
            
    fig = plt.figure()

    for n in range(len(base_epiped)):
        # each view along another plane normal
        selection = [True,]*3
        selection[n] = False
        plane_v1, plane_v2 = base_epiped[selection]
        plane_normal = np.cross(plane_v1, plane_v2)
        plane_normal = base_epiped[n]
        if plane_normal[0]:
            az = np.arctan(plane_normal[1]/plane_normal[0])
        else:
            az = 0
        elev = np.arccos(plane_normal[2]/np.linalg.norm(plane_normal))
        ax = fig.add_subplot(1, 3, n+1,
                             projection="3d",
                             proj_type='ortho',
                             elev=(az/(2*np.pi))*360,
                             azim=(elev/(2*np.pi))*360,
        )

        # plot circle
        u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:40j]
        x = radius*np.cos(u)*np.sin(v)+origin[0]
        y = radius*np.sin(u)*np.sin(v)+origin[1]
        z = radius*np.cos(v)+origin[2]
        ax.plot_surface(x, y, z, color="r", alpha=0.1)


        # plot the basic epiped
        plot_epiped(np.array([0,0,0], dtype=float), base_epiped, ax, full=True)

        # plot the covering epiped
        cover_origin = np.array([0,0,0], dtype=float)
        for d, reps in enumerate(rep):
            cover_origin += reps[0]*base_epiped[d]
        cover_epiped = np.array([(reps[1]-reps[0])*base_epiped[d]
                             for d, reps in enumerate(rep)])
        plot_epiped(cover_origin, cover_epiped, ax,
                    verts=False, full=False,)
        # plot the tiled epipeds if called force
        if draw_subtiles:
            trans_vectors = rdf._repetitionranges_to_translations(
                rep, base_epiped)
            epi_radius, epi_center = rdf._get_epiped_circumsphere(base_epiped)
            if cull_subtiles:
                cull_vector = rdf._cull_translations_to_origin(trans_vectors+epi_center, radius+epi_radius, origin)
                trans_vectors = trans_vectors[cull_vector]
            for tvec in trans_vectors:
                plot_epiped(tvec, base_epiped, ax, verts=False, full=False, draft=True)
        s = max((radius//10)*11, 10)
        ax.set_xlabel('X')
        ax.set_xlim([-s,s])
        ax.set_ylabel('Y')
        ax.set_ylim([-s,s])
        ax.set_zlabel('Z')
        ax.set_zlim([-s,s])
                
        # plot the center
        ax.scatter([0,],[0,],[0,], marker="*", s=400, c='y')
        
    plt.show()
        
def _calculate_sample_pddf(strucs, props, radius=20, gaussian=None):
    rdfs = []
    for s in strucs:
        r = rdf.calc_property_fp(s, radius=radius, binsize=0.1, gaussian=gaussian, norm_vol="3D",
                                 properties=props, property_mixer=rdf.w_stanley_density)
        rdfs.append(r)
    return rdfs


def benchmark_pddf(with_gaussian=None):
    DATA_DIR = "test_data/perovskites/"
    strucfiles = [f for f in os.listdir(DATA_DIR) if f.endswith(".cif")]
    sample_strucs = [ase.io.read(f"{DATA_DIR}/{f}") for f in strucfiles]
    sample_props = json.load(open(f"{DATA_DIR}/propdict.json"))
    #_calculate_sample_pddf(sample_strucs, sample_props)
    prof = cProfile.Profile()
    rdfs = prof.runcall(_calculate_sample_pddf, sample_strucs, sample_props, gaussian=with_gaussian)
    
    return prof, sample_strucs, rdfs

if __name__ == '__main__':
    unittest.main()



#s = benchmark_pddf()
# start@2020-07-08: 52.8s @ Ryzen-VM with OpenBLAS
# base_epiped = np.array([[1,1,0],[0,2,0],[0,2,3]])
# base_point = np.array([0,0,0])
# #base_point = np.array([-3, -4, -5])
# #base_point = np.array([1.4,1.1,2.051651])
# #base_r = 8.2
# base_r = 0.71
# vis_verify_covering(np.array(base_epiped),
#                     base_r,
#                     np.array(base_point),
#                     draw_subtiles=False,
#                     cull_subtiles=True)

# nacl = ase.build.bulk("NaCl", 'rocksalt', a=5)
# x = ase.io.read("test_data/perovskites/simple_cubic_perovskite.cif")
# y = rdf._get_pointdistances(x, 10)
# sample_dists = y.distances_by_center[0]
# sample_bins = rdf._bin_single_dist(sample_dists, 10,
#                                    1, None, None, None, "count", {"C": 1.4, "H": 2, "I": 1, "N": 10, "Ge" : 141}, rdf.w_stanley_density)
# print(sample_bins)
#to = TestSingleDistBin()
#to.setUp()
