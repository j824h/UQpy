import numpy as np
import scipy.stats as stats

from UQpy.sample_methods.strata.Strata import Strata


class DelaunayStrata(Strata):
    """
        Define a geometric decomposition of the n-dimensional unit hypercube into disjoint and space-filling
        Delaunay strata of n-dimensional simplexes.

        ``DelaunayStrata`` is a child class of the ``strata`` class.

        **Inputs:**

        * **seeds** (`ndarray`):
            An array of dimension `N x n` specifying the seeds of all strata. The seeds of the strata are the
            coordinates of the vertices of the Delaunay cells.

            The user must provide `seeds` or `nseeds` and `dimension`

            Note that, if `seeds` does not include all corners of the unit hypercube, they are added.

        * **nseeds** (`int`):
            The number of seeds to randomly generate. Seeds are generated by random sampling on the unit hypercube. In
            addition, the class also adds seed points at all corners of the unit hypercube.

            The user must provide `seeds` or `nseeds` and `dimension`

        * **dimension** (`ndarray`):
            The dimension of the unit hypercube in which to generate random seeds. Used only if `nseeds` is provided.

            The user must provide `seeds` or `nseeds` and `dimension`

        * **random_state** (None or `int` or ``numpy.random.RandomState`` object):
            Random seed used to initialize the pseudo-random number generator. Default is None.

            If an integer is provided, this sets the seed for an object of ``numpy.random.RandomState``. Otherwise, the
            object itself can be passed directly.

        * **verbose** (`Boolean`):
            A boolean declaring whether to write text to the terminal.


        **Attributes:**

        * **seeds** (`ndarray`):
            An array of dimension `N x n` containing the seeds of all strata. The seeds of the strata are the
            coordinates of the vertices of the Delaunay cells.

        * **centroids** (`ndarray`)
            A list of the vertices for each Voronoi stratum on the unit hypercube.

        * **delaunay** (`object` of ``scipy.spatial.Delaunay``)
            Defines a Delaunay decomposition of the set of seed points and all corner points.

        * **volume** (`ndarray`):
            An array of dimension `(nstrata, )` containing the volume of each Delaunay stratum in the unit hypercube.

        **Methods:**
        """

    def __init__(self, seeds=None, seeds_number=None, dimension=None, random_state=None, verbose=False):
        super().__init__(random_state=random_state, seeds=seeds, verbose=verbose)

        self.seeds_number = seeds_number
        self.dimension = dimension
        self.delaunay = None
        self.centroids = []

        if self.seeds is not None:
            if self.seeds_number is not None or self.dimension is not None:
                print("UQpy: Ignoring 'nseeds' and 'dimension' attributes because 'seeds' are provided")
            self.seeds_number, self.dimension = self.seeds.shape[0], self.seeds.shape[1]

        self.stratify()

    def stratify(self):
        import itertools
        from scipy.spatial import Delaunay

        if self.verbose:
            print('UQpy: Creating Delaaunay stratification ...')

        initial_seeds = self.seeds
        if self.seeds is None:
            initial_seeds = stats.uniform.rvs(size=[self.seeds_number, self.dimension], random_state=self.random_state)

        # Modify seeds to include corner points of (0,1) space
        corners = list(itertools.product(*zip([0]*self.dimension, [1]*self.dimension)))
        initial_seeds = np.vstack([initial_seeds, corners])
        initial_seeds = np.unique([tuple(row) for row in initial_seeds], axis=0)

        self.delaunay = Delaunay(initial_seeds)
        self.centroids = np.zeros([0, self.dimension])
        self.volume = np.zeros([0])
        count = 0
        for sim in self.delaunay.simplices:  # extract simplices from Delaunay triangulation
            # pylint: disable=E1136
            cent, vol = self.compute_delaunay_centroid_volume(self.delaunay.points[sim])
            self.centroids = np.vstack([self.centroids, cent])
            self.volume = np.hstack([self.volume, np.array([vol])])
            count = count + 1

        if self.verbose:
            print('UQpy: Delaunay stratification created.')

    @staticmethod
    def compute_delaunay_centroid_volume(vertices):
        """
        This function computes the centroid and volume of a Delaunay simplex from its vertices.

        **Inputs:**

        * **vertices** (`ndarray`):
            Coordinates of the vertices of the simplex.

        **Output/Returns:**

        * **centroid** (`numpy.ndarray`):
            Centroid of the Delaunay simplex.

        * **volume** (`numpy.ndarray`):
            Volume of the Delaunay simplex.
        """

        from scipy.spatial import ConvexHull

        ch = ConvexHull(vertices)
        volume = ch.volume
        # ch.volume: float = ch.volume
        centroid = np.mean(vertices, axis=0)

        return centroid, volume
