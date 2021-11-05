import itertools

import numpy as np
from beartype import beartype

from UQpy.utilities.ValidationTypes import Numpy2DFloatArray
from UQpy.dimension_reduction.kernels.baseclass.Kernel import Kernel
from UQpy.dimension_reduction.grassmann_manifold.optimization.baseclass.OptimizationMethod \
    import OptimizationMethod
from UQpy.dimension_reduction.distances.grassmann.baseclass.RiemannianDistance import RiemannianDistance
from UQpy.dimension_reduction.grassmann_manifold.projection.baseclass.ManifoldProjection import (
    ManifoldProjection,
)


class Grassmann:
    def __init__(self, manifold_projected_points: ManifoldProjection):
        """

        :param ManifoldProjection manifold_projected_points: Points on Grassmann manifold given as an object of type
         ManifoldProjection.
        """
        self.manifold_projected_points = manifold_projected_points

    def evaluate_kernel_matrix(self, kernel):
        kernel_matrix = self.manifold_projected_points.evaluate_matrix(kernel)
        return kernel_matrix

    @staticmethod
    @beartype
    def log_map(manifold_points: list[Numpy2DFloatArray], reference_point: Numpy2DFloatArray)\
            -> list[Numpy2DFloatArray]:
        """
        :param manifold_points: Point(s) on the Grassmann manifold.
        :param reference_point: Origin of the tangent space.
        :return: Point(s) on the tangent space.
        """
        number_of_points = Kernel.check_data(manifold_points)

        for i in range(number_of_points):
            RiemannianDistance.check_points(reference_point, manifold_points[i])
            if reference_point.shape[1] != manifold_points[i].shape[1]:
                raise ValueError("UQpy: Point {0} is on G({1},{2}) - Reference is on"
                                 " G({1},{2})".format(i, manifold_points[i].shape[1], manifold_points[i].shape[0],
                                                      reference_point.shape[1])
                                 )

        # Multiply ref by its transpose.
        reference_point_transpose = reference_point.T
        m_ = np.dot(reference_point, reference_point_transpose)

        tangent_points = []
        for i in range(number_of_points):
            u_trunc = manifold_points[i]
            # compute: M = ((I - psi0*psi0')*psi1)*inv(psi0'*psi1)
            m_inv = np.linalg.inv(np.dot(reference_point_transpose, u_trunc))
            m = np.dot(u_trunc - np.dot(m_, u_trunc), m_inv)
            ui, si, vi = np.linalg.svd(m, full_matrices=False)
            tangent_points.append(np.dot(np.dot(ui, np.diag(np.arctan(si))), vi))

        return tangent_points

    @staticmethod
    @beartype
    def exp_map(tangent_points: list[Numpy2DFloatArray], reference_point: Numpy2DFloatArray) \
            -> list[Numpy2DFloatArray]:
        """
        :param tangent_points: Tangent vector(s).
        :param reference_point: Origin of the tangent space.
        :return: Point(s) on the Grassmann manifold.
        """

        number_of_points = len(tangent_points)

        for i in range(number_of_points):
            if reference_point.shape[1] != tangent_points[i].shape[1]:
                raise ValueError("UQpy: Point {0} is on G({1},{2}) - Reference is on"
                                 " G({1},{2})".format(i, tangent_points[i].shape[1], tangent_points[i].shape[0],
                                                      reference_point.shape[1])
                                 )

        # Map the each point back to the manifold.
        manifold_points = list()
        for i in range(number_of_points):
            u_trunc = tangent_points[i]
            ui, si, vi = np.linalg.svd(u_trunc, full_matrices=False)

            x0 = np.dot(
                np.dot(np.dot(reference_point, vi.T), np.diag(np.cos(si)))
                + np.dot(ui, np.diag(np.sin(si))),
                vi,
            )

            if not np.allclose(x0.T @ x0, np.eye(u_trunc.shape[1])):
                x0, _ = np.linalg.qr(x0)

            manifold_points.append(x0)

        return manifold_points

    @staticmethod
    @beartype
    def frechet_variance(manifold_points: list[Numpy2DFloatArray], reference_point: Numpy2DFloatArray,
                         distance: RiemannianDistance) -> float:
        """
        :param manifold_points: Point(s) on the Grassmann manifold
        :param reference_point: Reference point
        :param distance: Distance metric to be used for the optimization.
        """
        p_dim = []
        for i in range(len(manifold_points)):
            p_dim.append(min(np.shape(np.array(manifold_points[i]))))

        points_number = len(manifold_points)

        if points_number < 2:
            raise ValueError("UQpy: At least two input matrices must be provided.")

        variance_nominator = 0
        for i in range(points_number):
            distances = Grassmann.__estimate_distance(
                [reference_point, manifold_points[i]], p_dim, distance
            )
            variance_nominator += distances[0] ** 2

        frechet_variance = variance_nominator / points_number
        return frechet_variance

    @staticmethod
    def __estimate_distance(points, p_dim, distance):

        # Check points for type and shape consistency.
        # -----------------------------------------------------------
        if not isinstance(points, list) and not isinstance(points, np.ndarray):
            raise TypeError(
                "UQpy: The input matrices must be either list or numpy.ndarray."
            )

        nargs = len(points)

        if nargs < 2:
            raise ValueError("UQpy: At least two matrices must be provided.")

        # ------------------------------------------------------------

        # Define the pairs of points to compute the grassmann_manifold distance.
        indices = range(nargs)
        pairs = list(itertools.combinations(indices, 2))

        # Compute the pairwise distances.
        distance_list = []
        for id_pair in range(np.shape(pairs)[0]):
            ii = pairs[id_pair][0]  # Point i
            jj = pairs[id_pair][1]  # Point j

            p0 = int(p_dim[ii])
            p1 = int(p_dim[jj])

            x0 = np.asarray(points[ii])[:, :p0]
            x1 = np.asarray(points[jj])[:, :p1]

            # Call the functions where the distance metric is implemented.
            distance_value = distance.compute_distance(x0, x1)

            distance_list.append(distance_value)

        return distance_list

    @staticmethod
    @beartype
    def karcher_mean(manifold_points: list[Numpy2DFloatArray], optimization_method: OptimizationMethod,
                     distance: RiemannianDistance) -> Numpy2DFloatArray:
        """
        :param manifold_points: Point(s) on the Grassmann manifold.
        :param optimization_method: The optimization method.
        :param distance: Distance metric to be used for the optimization.
        :return:
        """
        # Test the input data for type consistency.
        if not isinstance(manifold_points, list) and not isinstance(
            manifold_points, np.ndarray
        ):
            raise TypeError(
                "UQpy: `points_grassmann` must be either list or numpy.ndarray."
            )

        # Compute and test the number of input matrices necessary to compute the Karcher mean.
        nargs = len(manifold_points)
        if nargs < 2:
            raise ValueError("UQpy: At least two matrices must be provided.")

        kr_mean = optimization_method.optimize(manifold_points, distance)

        return kr_mean

    @staticmethod
    def calculate_pairwise_distances(distance_method, points_grassmann):
        if isinstance(points_grassmann, np.ndarray):
            points_grassmann = points_grassmann.tolist()

        n_size = max(np.shape(points_grassmann[0]))
        for i in range(len(points_grassmann)):
            if n_size != max(np.shape(points_grassmann[i])):
                raise TypeError(
                    "UQpy: The shape of the input matrices must be the same."
                )

        # if manifold_points is provided, use the shape of the input matrices to define
        # the dimension of the p-planes defining the manifold of each individual input matrix.
        p_dim = []
        for i in range(len(points_grassmann)):
            p_dim.append(min(np.shape(np.array(points_grassmann[i]))))

        # Compute the pairwise distances.
        points_distance = Grassmann.__estimate_distance(
            points_grassmann, p_dim, distance_method
        )

        # Return the pairwise distances.
        return points_distance
