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
        :param a: Unit cell a vector
        :param b: Unit cell b vector
        :param c: Unit cell c vector
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
    def __init__(self):


def make_ZnO_crystal(dopant_concentration, )