# pair_distribution.py
#
# A tool to find the distribution of pairs, depending on dopant concentration.
# David Lister
# June 2023
#

import numpy as np
from numba import jit
import time
import pickle
import matplotlib.pyplot as plt
import datetime
import matplotlib
matplotlib.use("TkAgg")


# Symbolic Constants
Zn = "Zn"
O = "O"
Ga = "Ga"

# Fingerprint indexes
x = 0
y = 1
z = 2
sub_pos = 3


class CrystalStructure:
    def __init__(self, a, b, c, atoms, dopant_probabilities):
        """
        Holding class for the crystal lattice properties
        :param a: Unit cell a vector, units of nm
        :param b: Unit cell b vector, units of nm
        :param c: Unit cell c vector, units of nm
        :param atoms: a dict of atom positions that make up the structure.
        :param dopant_probabilities: dictionary of dopants and their probabilities.
        """
        self.a = a
        self.b = b
        self.c = c
        self.atoms = atoms
        self.dopants = dopant_probabilities


class Crystal:
    def __init__(self, crystal_structure, side_length):
        """
        Hold information about a crystal.
        :param crystal_structure: A CrystalStructure object containing overall structure information
        """
        self.structure = crystal_structure
        self.side_length = side_length
        self.dopant_dict = {}
        self.dopant_positions = []
        self.dopant_coordinates = []
        self.n_dopants = None
        self.volume = None
        self.unit_cell_volume = None
        self.cells_per_side = None
        self.nominal_dopant_concentration = None


def calculate_unit_cell_volume(struct):
    """
    Calculates unit cell volume using a tripple product
    :param struct: Crystal structure object
    :return: volume as a float
    """
    volume = np.linalg.det(np.dstack([struct.a, struct.b, struct.c]))  # Determinant form of triple product
    return volume[0]


def select_element(element_prob_dict):
    value = np.random.random_sample(1)[0]
    element = None
    cumulative_prob = 0
    for ele in sorted(element_prob_dict):
        if element_prob_dict[ele] + cumulative_prob >= value > cumulative_prob:
            element = ele

        cumulative_prob += element_prob_dict[ele]

    if element is None:
        print("Error in probability definitions!")

    return element


def get_xyz(struct, path):
    if len(path) == 3:
        sub_coordinate = [0, 0, 0]
    else:
        sub_coordinate = path[sub_pos]

    row_1 = struct.a * (path[x] + sub_coordinate[x])
    row_2 = struct.b * (path[y] + sub_coordinate[y])
    row_3 = struct.c * (path[z] + sub_coordinate[z])
    total = row_1 + row_2 + row_3
    # print(path, total)
    return np.array((total[x], total[y], total[z]))


def make_ZnO_crystal(dopant_concentration, n_dopants):
    """
    Makes a ZnO crystal object.
    :param dopant_concentration: Dopant concentration in units of cm^-3
    :param n_dopants: Number of dopants to simulate, size will be scaled accordingly.
    :return: A Crystal object representing a doped ZnO sample.
    """
    start = time.time()
    a = 0.32495
    c = 0.52069
    u = 3/8
    ZnO_struct = CrystalStructure(a=np.array((a/2, -np.sqrt(3) * a / 2, 0)), # From https://www.sciencedirect.com/science/article/pii/S2214785320378445 and https://www.atomic-scale-physics.de/lattice/struk/b4.html
                                  b=np.array((a/2, np.sqrt(3) * a / 2, 0)),
                                  c=np.array((0, 0, c)),
                                  atoms={Zn:   (np.array((1/3, 2/3, 0)),
                                                np.array((2/3, 1/3, 1/2))),
                                         O:    (np.array((1/3, 2/3, u)),
                                                np.array((2/3, 1/3, u + 1/2)))
                                                },
                                  dopant_probabilities={Ga: {Zn:0.99999, O:0.00001}})   # Guess for the relative ratios

    unit_cell_volume = calculate_unit_cell_volume(ZnO_struct)
    print(f"Unit cell volume is {unit_cell_volume:0.4}nm^3")

    dopants_per_nm3 = dopant_concentration / 1e21  # ( 10^7 )^3
    target_volume_nm3 = n_dopants / dopants_per_nm3
    n_cells_per_side = int((target_volume_nm3/unit_cell_volume)**(1/3))

    system_volume = n_cells_per_side**3 * unit_cell_volume
    print(f"System volume is {system_volume:0.1f}nm^3 or equivalent to a cube of {system_volume**(1/3):0.1f}nm per side")


    mean_n_dopants = system_volume * dopants_per_nm3
    print(f"The mean number of dopants in the volume is {mean_n_dopants:0.1f}, adding {int(mean_n_dopants)} dopants")

    crystal = Crystal(ZnO_struct, n_cells_per_side)
    report_every = 100000

    i = 0
    dopant = Ga
    overlap = 0
    while i < int(mean_n_dopants):
        # Not efficient, but clear. Could be improved later.
        if i % report_every == 0:
            print(f"Adding dopant number {i}")

        xyz = np.random.randint(0, n_cells_per_side, size=3)
        replacement = select_element(ZnO_struct.dopants[Ga])
        site_index = np.random.randint(0, len(ZnO_struct.atoms[replacement]), size=1)[0]
        fingerprint = tuple(xyz) + (tuple(ZnO_struct.atoms[replacement][site_index]),)

        if fingerprint in crystal.dopant_dict:
            overlap += 1
        else:
            i += 1
            crystal.dopant_dict[fingerprint] = dopant
            crystal.dopant_positions.append(get_xyz(ZnO_struct, fingerprint))
            crystal.dopant_coordinates.append(fingerprint)

    crystal.dopant_positions = np.array(crystal.dopant_positions)
    crystal.n_dopants = int(mean_n_dopants)
    crystal.volume = system_volume  # nm^3
    crystal.unit_cell_volume = unit_cell_volume
    crystal.cells_per_side = n_cells_per_side
    crystal.nominal_dopant_concentration = dopant_concentration

    print(f"Complete! There were {overlap} overlapping cases, corresponding to {100 * overlap / i:.3f}%")
    print(f"Crystal took {time.time() - start:.2f} seconds to build")
    return crystal
    # todo: calculate stdev of n_dopants given the volume and conc


def test_one_bound(test_position, r, plane, bound):
    # print(test_position, r, plane, bound)
    projection = np.abs(np.vdot(test_position, plane)) / np.linalg.norm(plane)
    # print(projection)
    if 0 + r < projection < bound - r:
        return True
    return False

def in_bounds(test_position, r, p100, p010, p001, b100, b010, b001):
    a_test = test_one_bound(test_position, r, p100, b100)
    b_test = test_one_bound(test_position, r, p010, b010)
    c_test = test_one_bound(test_position, r, p001, b001)
    # print(a_test, b_test, c_test)
    return a_test and b_test and c_test


@jit(nopython=True)
def points_within(test_point, points, r):
    mask = np.full(points.T[0].shape, True)
    for i in range(3):
        mask = np.logical_and(mask, points.T[i] >= test_point[i] - r)
        mask = np.logical_and(mask, points.T[i] <= test_point[i] + r)
    return points[mask]


def find_nearest_distance(test_point, points, max_r, return_position=False):
    test_r = max_r/5 # nm
    over = False
    out = None
    pos = None
    while not over:
        if test_r == max_r:
            over = True
            # print("Last attempt")

        subset = points_within(test_point, points, test_r)
        # print(subset.shape)
        if len(subset) < 1:
            test_r = max_r
            # print("Expanding search")

        else:
            subset = subset.T
            for i in range(3):
                subset[i] -= test_point[i]
            subset = subset.T

            distances = np.linalg.norm(subset, axis=1)
            min_index = np.where(distances == np.min(distances))[0][0]
            # print(f"Test r is {test_r}, {len(distances)} pairs found, smallest is {distances[min_index]} nm")

            if distances[min_index] > test_r:
                if test_r != max_r:
                    test_r = min((max_r, test_r * 1.5))
                    # print("Spurious value possible, double checking")

            else:
                over = True
                out = distances[min_index]
                pos = subset[min_index] + test_point
                # print(f"Selected {out}")

    if return_position:
        return pos
    return out


def calculate_pair_distribution(crystal, sample_percent=0.01, max_r=50):
    """
    Calculates standard pair distribution using the simulated crystal.
    :param crystal: Crystal object
    :param sample_percent: Percent of dopants to sample, should probably be no higher than 10%, better if it's on the order of 1%
    :param max_r: Maximum distance between dopants to look for, in nanometers
    :return: List of dopant distances
    """
    start = time.time()
    samples = int(crystal.n_dopants * sample_percent)
    print(f"Taking {samples} samples to calculate pair distribution")

    plane_100 = np.cross(crystal.structure.b, crystal.structure.c)
    plane_010 = np.cross(crystal.structure.a, crystal.structure.c)
    plane_001 = np.cross(crystal.structure.a, crystal.structure.b)

    bounds_100 = np.linalg.norm(crystal.structure.a) * crystal.cells_per_side
    bounds_010 = np.linalg.norm(crystal.structure.b) * crystal.cells_per_side
    bounds_001 = np.linalg.norm(crystal.structure.c) * crystal.cells_per_side

    dopant_positions = crystal.dopant_positions
    blank_mask = np.full(dopant_positions.T[0].shape, True)

    rejected_retest = 0
    rejected_bounds = 0
    i = 0
    tested_indexes = []
    distances = []
    while i < samples:
        if i % 1000 == 0:
            print(f"Finding pair distance number {i}")
        index = np.random.randint(0, high=len(dopant_positions))
        if in_bounds(dopant_positions[index], max_r, plane_100, plane_010, plane_001, bounds_100, bounds_010, bounds_001):
            if index not in tested_indexes:
                i += 1
                tested_indexes.append(index)
                test_mask = blank_mask
                test_mask[index] = False
                r = find_nearest_distance(dopant_positions[index], dopant_positions[test_mask], max_r)
                if r is None:
                    r = max_r

                distances.append(r)

            else:
                rejected_retest += 1
        else:
            rejected_bounds += 1

    print(f"Pair distribution with {samples} took {time.time() - start:.2f} seconds")
    print(f"In the process, {rejected_bounds} test samples were rejected because they were out of bounds")
    print(f"and {rejected_retest} were rejected to prevent redundant sampling.")
    print(f"In total {100 * (rejected_bounds + rejected_retest)/(samples + rejected_retest + rejected_bounds):2f}% of attempted samples were rejected.")
    return distances


def calculate_two_step_pair_distribution(crystal, sample_percent=0.01, max_r=50):
    """
    Calculates two-step pair distribution using the simulated crystal.
    :param crystal: Crystal object
    :param sample_percent: Percent of dopants to sample, should probably be no higher than 10%, better if it's on the order of 1%
    :param max_r: Maximum distance between dopants to look for, in nanometers
    :return: List of dopant distances
    """
    start = time.time()
    samples = int(crystal.n_dopants * sample_percent)
    print(f"Taking {samples} samples to calculate two-step pair distribution")

    plane_100 = np.cross(crystal.structure.b, crystal.structure.c)
    plane_010 = np.cross(crystal.structure.a, crystal.structure.c)
    plane_001 = np.cross(crystal.structure.a, crystal.structure.b)

    bounds_100 = np.linalg.norm(crystal.structure.a) * crystal.cells_per_side
    bounds_010 = np.linalg.norm(crystal.structure.b) * crystal.cells_per_side
    bounds_001 = np.linalg.norm(crystal.structure.c) * crystal.cells_per_side

    dopant_positions = crystal.dopant_positions
    blank_mask = np.full(dopant_positions.T[0].shape, True)

    rejected_bounds = 0
    i = 0
    distances = []
    while i < samples:
        if i % 500 == 0:
            print(f"Finding pair distance number {i}")

        test_vector = np.random.random_sample(3) * crystal.cells_per_side
        test_position = get_xyz(crystal.structure, test_vector)

        if in_bounds(test_position, max_r, plane_100, plane_010, plane_001, bounds_100, bounds_010, bounds_001):
            i += 1
            nearest = find_nearest_distance(test_position, dopant_positions, max_r, return_position=True)
            if nearest is None:
                pass
                ## Update to correct for lost sample

            else:
                index = np.where(nearest == dopant_positions)[0][0]
                test_mask = blank_mask
                test_mask[index] = False
                r = find_nearest_distance(dopant_positions[index], dopant_positions[test_mask], max_r)
                if r is None:
                    r = max_r
                    
                distances.append(r)
    
        else:
            rejected_bounds += 1

    print(f"Pair distribution with {samples} took {time.time() - start:.2f} seconds")
    print(f"In the process, {rejected_bounds} test samples were rejected because they were out of bounds.")
    print(f"In total {100 * (rejected_bounds)/(samples + rejected_bounds):2f}% of attempted samples were rejected.")
    return distances


gaussian = lambda value, center, fwhm: 1/((fwhm/(2 * np.sqrt(2 * np.log(2)))) * np.sqrt(2 *np.pi)) * np.exp( - (value - center)**2 / (2 * (fwhm/(2 * np.sqrt(2 * np.log(2))))**2))


def make_simulated_spectra(E_lst, fwhm, x_axis=None):
    if x_axis is None:
        max_E = np.max(E_lst) + fwhm * 10
        min_E = np.min(E_lst) - fwhm * 10
        resolution = fwhm / 20
        x_axis = np.linspace(min_E, max_E, int((max_E - min_E)/resolution))

    out = np.zeros(x_axis.shape)
    for value in E_lst:
        out = out + gaussian(x_axis, value, fwhm)

    out = out / len(E_lst)
    return x_axis, out


ZnO_DoX_binding_eV = 0.0156
ZnO_DoX_bohr_rad_nm = 1.5

delta_E_from_R_nm = lambda r_nm, bohr_rad_nm, binding_energy_eV: -2 * binding_energy_eV * np.exp(-r_nm / bohr_rad_nm)

R_avg_nm_from_Nd_cm = lambda Nd_cm: 6 ** (1 / 3) / (2 * (np.pi * Nd_cm * 100 ** 3) ** (1 / 3)) * 1e9
R_nm_from_delta_E = lambda delta_E_eV, bohr_rad_nm, binding_energy_eV: -bohr_rad_nm * np.log(-delta_E_eV / (2 * binding_energy_eV))
prob_from_R_nm = lambda R_nm, average_R_nm: ((3 * R_nm ** 2) / (average_R_nm ** 3)) * np.exp((-R_nm / average_R_nm) ** 3)

pair_model = lambda delta_E_eV, Nd_cm, bohr_rad_nm, binding_energy_eV: ( (3 * R_nm_from_delta_E(delta_E_eV, bohr_rad_nm, binding_energy_eV) ** 2) / (R_avg_nm_from_Nd_cm(Nd_cm)) ) * \
                                                                         np.exp( (R_nm_from_delta_E(delta_E_eV, bohr_rad_nm, binding_energy_eV)) / (bohr_rad_nm) - (R_nm_from_delta_E(delta_E_eV, bohr_rad_nm, binding_energy_eV) / R_avg_nm_from_Nd_cm(Nd_cm))**3)


if __name__ == "__main__":
    start_time = time.time()
    nd_lst = [1e16, 2.5e17, 5e16, 7.5e16,
              1e17, 2.5e17, 5e17, 7.5e17,
              1e18, 2.5e18, 5e18, 7.5e18,
              1e19, 2.5e19, 5e19, 7.5e19,
              1e20, 2.5e20, 5e20, 7.5e20,
              1e21]
    min_samples = 100000
    min_iterations = 10
    results = {"pair": {},
               "step": {}}

    cycle = 0
    while True:  # run forever
        cycle += 1
        for nd_test in nd_lst:
            r_0 = R_avg_nm_from_Nd_cm(nd_test)
            distances_step = []
            distances_pair = []

            over = False
            iteration = 0
            while not over:
                iteration += 1
                print(f"\n\n\n\nIteration {iteration + 1} at Nd = {nd_test:.2E}. {len(distances_step)}  samples collected so far:\n")
                ZnO = make_ZnO_crystal(nd_test, 250000)
                out = calculate_two_step_pair_distribution(ZnO, max_r=5*r_0)
                for val in out:
                    distances_step.append(val)

                out = calculate_pair_distribution(ZnO, max_r=5*r_0)
                for val in out:
                    distances_pair.append(val)

                if iteration >= min_iterations and len(distances_step) >= min_samples:
                    over = True

                elif iteration <= min_iterations:
                    print(f"Need more iterations! {100 * iteration / min_iterations}% complete")

                else:
                    print(f"Need more samples! {100 * len(distances_step) / min_samples}% complete")

            results["pair"][nd_test] = np.array(distances_pair)
            results["step"][nd_test] = np.array(distances_step)

            fname = "Data_Capture_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".data"
            with open(fname, 'wb') as f:
                pickle.dump(results)

            print(f"Time taken: {(time.time() - start_time) / 3600} hours, finished cycle {cycle}.")

    # Plotting results not presently used.
    for nd_test in nd_lst:
        r_0 = R_avg_nm_from_Nd_cm(nd_test)
        plt.hist(results["pair"][nd_test], bins=100, alpha=0.5, density=True, label="Pair Experimental")
        hst = plt.hist(results["step"][nd_test], bins=100, alpha=0.5, density=True, label="Two Step")
        plt.plot(hst[1], prob_from_R_nm(hst[1], r_0), label="Pair analytic")

        plt.title(f"Pair vs Two-step model for Nd = {nd_test:.1} cm^-3")
        plt.xlabel("r (nm)")
        plt.ylabel("prob")
        plt.legend()
        plt.show()

        simulated_fwhm = 1e-4
        delta_E_pair_eV = delta_E_from_R_nm(results["pair"][nd_test], ZnO_DoX_bohr_rad_nm, ZnO_DoX_binding_eV)
        energy_axis, pair_E_spectra = make_simulated_spectra(delta_E_pair_eV, simulated_fwhm)
        delta_E_step_eV = delta_E_from_R_nm(results["step"][nd_test], ZnO_DoX_bohr_rad_nm, ZnO_DoX_binding_eV)
        skip, step_E_spectra = make_simulated_spectra(delta_E_step_eV, simulated_fwhm, x_axis=energy_axis)
        pair_analytic = pair_model(energy_axis, nd_test, ZnO_DoX_bohr_rad_nm, ZnO_DoX_binding_eV)

        plt.semilogy(energy_axis, pair_E_spectra, label="Pair experimental")
        plt.semilogy(energy_axis, step_E_spectra, label="Step experimental")
        plt.semilogy(energy_axis, pair_analytic, label="Pair analytic")

        plt.title(f"Energy spectra Pair vs Two-step model for Nd = {nd_test:.1} cm^-3")
        plt.xlabel("Delta E (eV)")
        plt.ylabel("prob")
        plt.legend()
        plt.show()

