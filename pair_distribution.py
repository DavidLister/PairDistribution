# pair_distribution.py
#
# A tool to find the distribution of pairs, depending on dopant concentration.
# David Lister
# June 2023
#

import numpy as np
import matplotlib.pyplot as plt

# Symbolic Constants
Zn = "Zn"
O = "O"

class CrystalStructure:
    def __init__(self, a, b, c, atoms):
        """
        Holding class for the crystal lattice properties
        :param a: Unit cell a vector, units of nm
        :param b: Unit cell b vector, units of nm
        :param c: Unit cell c vector, units of nm
        :param atoms: a list of Atom classes that make up the structure.
        """
        self.a = a
        self.b = b
        self.c = c
        self.atoms = atoms


class Atom:
    def __init__(self, position, element):
        """
        Holds information about the atoms that make up the lattice
        :param position:
        :param element: String denoting the atom
        """
        self.position = position
        self.element = element


class Crystal:
    def __init__(self, crystal_structure):
        """
        Hold information about a crystal.
        :param crystal_structure: A CrystalStructure object containing overall structure information
        """
        self.structure = crystal_structure
        self.dopant_positions = []


def calculate_unit_cell_volume(struct):
    """
    Calculates unit cell volume using a tripple product
    :param struct: Crystal structure object
    :return: volume as a float
    """
    volume = np.linalg.det(np.dstack([struct.a, struct.b, struct.c]))  # Determinant form of tripple product
    return volume[0]

def make_ZnO_crystal(dopant_concentration, n_cells_per_side):
    """
    Makes a ZnO crystal object.
    :param dopant_concentration: Dopant concentration in units of cm^-3
    :param n_cells_per_side: Number of unit cells per side of the crystal
    :return: A Crystal object representing a doped ZnO sample.
    """
    a = 0.32495
    c = 0.52069
    u = 3/8
    ZnO_struct = CrystalStructure(a=np.array((a/2, -np.sqrt(3) * a / 2, 0)), # From https://www.sciencedirect.com/science/article/pii/S2214785320378445 and https://www.atomic-scale-physics.de/lattice/struk/b4.html
                                  b=np.array((a/2, np.sqrt(3) * a / 2, 0)),
                                  c=np.array((0, 0, c)),
                                  atoms=[(Zn, np.array((1/3, 2/3, 0))),
                                           (Zn, np.array((2/3, 1/3, 1/2))),
                                           (O, np.array((1/3, 2/3, u))),
                                           (O, np.array((2/3, 1/3, u + 1/2)))])

    unit_cell_volume = calculate_unit_cell_volume(ZnO_struct)
    print(f"Unit cell volume is {unit_cell_volume:0.4}nm^3")

    system_volume = n_cells_per_side**3 * unit_cell_volume
    print(f"System volume is {system_volume:0.1f}nm^3 or equivalent to a cube of {system_volume**(1/3):0.1f}nm per side")

    dopants_per_nm3 = dopant_concentration / 1e21  # ( 10^7 )^3
    mean_n_dopants = system_volume * dopants_per_nm3
    print(f"The mean number of dopants in the volume is {mean_n_dopants:0.1f}")

    # todo: calculate stdev of n_dopants given the volume and conc

if __name__ == "__main__":
    make_ZnO_crystal(10e18, 1000)