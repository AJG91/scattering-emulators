"""
File used to build the momentum mesh.
"""
from numpy import append, tan, cos, pi
from numpy.polynomial.legendre import leggauss
from numpy.typing import ArrayLike
from typing import Union, Optional

class BuildMesh():
    """
    Generates a mesh and its corresponding intergration measure.

    Parameters
    ----------
    mesh_pts : list
        This is a list composed of the start and end points of the
        mesh. If building a compound mesh, this list should contain
        the start point, end point, and the intermediate start points of 
        each region of the internal meshes.
    pts_per : list
        This is a list composed of the total number of points in the mesh.
        If building a compound mesh, this list contains the total number
        of points in each region.
    inf_map : boolean (default=True)
        If True, uses an infinite mapping for the mesh endpoint.
        If False, uses a hard cutoff.
    """
    def __init__(
        self,
        mesh_pts: list,
        pts_per: Union[list, int],
        inf_map: bool = True
    ) -> tuple[ArrayLike, ArrayLike]:  
  
        nodes, weights = [], []
        if (isinstance(pts_per, list) == False):
            pts_per = [pts_per]
            
        check = False
        num_regions = len(pts_per)
        for i in range(num_regions):
            if (i == num_regions - 1):
                check = inf_map
            ps, ws = self.gauleg(mesh_pts[i], mesh_pts[i+1], pts_per[i], check)
            nodes, weights = append(nodes, ps), append(weights, ws)
            
        self.nodes, self.weights = nodes, weights
        
    def gauleg(
        self,
        a: float,
        b: float,  
        N: int, 
        inf_map: bool,
        opt: int = 3
    ) -> tuple[ArrayLike, ArrayLike]:
        """
        Generates a mesh and calculates an integral using 
        Gauss-Legendre quadrature (default interval is [-1,1]).
        Returns the chosen points and their corresponding weights.
        Can be used for integrals with upper limit of infinity.

        Parameters
        ----------
        a : float
            The initial point (lower limit of integral)
        b : float
            The end point (upper limit of integral)
        N : int
            The number of points used to evaluate integral
        inf_map : boolean
            If True, calculates integral with upper limit of infinity.
            If False, calculates integral with upper limit of b 
        opt : int (default = 3)
            Used to pick which expression for infinite mapping the user wants.
            Optio (opt) 3 and 4 are an infinite mapping that does not rely on the end point.

        Returns
        -------
        ps : array
            Mesh points.
        ws : array
            Mesh weights.
        """
        x, w = leggauss(N)

        if inf_map:
            if opt == 1:
                ps = 0.5 * (x + 1) * (b - a) + a
                ws = (b - a) / 2 * w

            elif opt == 2:
                ps = 0.5 * (b - a) * (1.0 + x) / (1.0 - x) + a
                ws = (b - a) * w / (1 - x)**2

            elif opt == 3:
                ps = tan((pi / 4) * (1 + x)) + a
                ws = (pi / 4) * (w / cos((pi / 4) * (1 + x))**2)
                
            elif opt == 4:
                ps = a + x / (1 - x)
                ws = w / (1 - x**2)
                
            else:
                raise ValueError('Wrong choice of mapping!')
        else:
            ps = 0.5 * ((b - a) * x + (b + a))
            ws = 0.5 * (b - a) * w
        return ps, ws



