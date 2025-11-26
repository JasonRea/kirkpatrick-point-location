import primitives as prim
import polygon as poly
import planar_graph as pg

X = 0
Y = 1
DIM = 2

def SegSegInt(a: prim.Point, b: prim.Point, c: prim.Point, d: prim.Point, p: prim.Point):
    '''
    given line segments ab, cd, and intersection point p
    s and t are parameters of parametric equations
    num and denom are the numerator and denominator of equations
    returns: 
        'e' -> segments collinearly overlap sharing a point, 'e' stands for edge
        'v' -> an endpoint of one segment is on the other segment but e doesn't hold, 'v' stands for vertex
        '1' -> segments properly intersect, '1' stands for True
        '0' -> segments do not intersect, '0' stands for False
    Section 7.2 of O'Rourke contains further details
    '''
    code = '?'
    denom = a[X] * (d[Y] - c[Y]) + \
            b[X] * (c[Y] - d[Y]) + \
            d[X] * (b[Y] - a[Y]) + \
            c[X] * (a[Y] - b[Y])
    
    if (denom == 0): # Segments are parallel
        return ParallelInt(a, b, c, d, p)
    
    num = a[X] * (d[Y] - c[Y]) + \
          c[X] * (a[Y] - d[Y]) + \
          d[X] * (c[Y] - a[Y])
    if (num == 0 or num == denom):
        code = 'v'
    s = num / denom

    num = -(
        a[X] * (c[Y] - b[Y]) + \
        b[X] * (a[Y] - c[Y]) + \
        c[X] * (b[Y] - a[Y])
    )
    if (num == 0 or num == denom):
        code = 'v'
    t = num / denom

    if ((0 < s < 1) and (0 < t < 1)):
        code = '1'
    elif (0 > s or s > 1 or 0 > t or t > 1):
        code = '0'

    p[X] = a[X] + s * (b[X] - a[X])
    p[Y] = a[Y] + s * (b[Y] - a[Y])

    return code

def ParallelInt(a: prim.Point, b: prim.Point, c: prim.Point, d: prim.Point, p: prim.Point):
    if not prim.Collinear(a, b, c):
        return 0
    if prim.Between(a, b, c):
        prim.Assigndi(p, c)
        return 'e'
    if prim.Between(a, b, d):
        prim.Assigndi(p, d)
        return 'e'
    if prim.Between(c, d, a):
        prim.Assigndi(p, a)
        return 'e'
    if prim.Between(c, d, b):
        prim.Assigndi(p, b)
        return 'e'
    return '0'

def inTri2D(tp: list, q: prim.Point):
    """
    Test if point q is inside triangle tp[0], tp[1], tp[2].
    Returns:
        'v' -> q is a vertex of triangle
        'e' -> q is on an edge of triangle  
        'f' -> q is in the interior (face) of triangle
        'o' -> q is outside triangle
    """
    area0 = prim.AreaSign(q, tp[0], tp[1])
    area1 = prim.AreaSign(q, tp[1], tp[2])
    area2 = prim.AreaSign(q, tp[2], tp[0])

    if (
        (area0 == 0 and area1 > 0 and area2 > 0) or
        (area1 == 0 and area0 > 0 and area2 > 0) or
        (area2 == 0 and area0 > 0 and area1 > 0)
        ):
        return 'e'
    
    if (
        (area0 == 0 and area1 < 0 and area2 > 0) or
        (area1 == 0 and area0 < 0 and area2 > 0) or
        (area2 == 0 and area0 < 0 and area1 > 0)
    ):
        return 'e'
    
    if (
        (area0 > 0 and area1 > 0 and area2 > 0) or
        (area0 < 0 and area1 < 0 and area2 < 0)
    ):
        return 'f'
    
    if (
        (area0 == 0 and area1 == 0 and area2 == 0)
    ):
        raise RuntimeError('Error in InTri2D')
    
    if (
        (area0 == 0 and area1 == 0) or
        (area0 == 0 and area2 == 0) or
        (area1 == 0 and area2 == 0)
    ):
        return 'v'

def InPoly1(q: prim.Point, P: poly.Polygon, n: int):
    """
    Test if point q is inside polygon P using ray casting algorithm.
    Does NOT modify the original polygon points.
    Returns:
        'v' -> q is a vertex of P
        'e' -> q is on an edge of P (or rays have same parity)
        'i' -> q is inside P
        'o' -> q is outside P
    """
    Rcross = 0  # Crossings to the right
    Lcross = 0  # Crossings to the left

    # Create shifted copies of polygon points (do not modify original)
    shifted_points = []
    for i in range(n):
        shifted_point = prim.Point(
            P.points[i][X] - q[X],
            P.points[i][Y] - q[Y]
        )
        shifted_points.append(shifted_point)

    for i in range(n):
        if (shifted_points[i][X] == 0 and shifted_points[i][Y] == 0):
            return 'v'  # q is a vertex
        
        i1 = (i + n - 1) % n

        Rstrad = (shifted_points[i][Y] > 0) != (shifted_points[i1][Y] > 0)
        Lstrad = (shifted_points[i][Y] < 0) != (shifted_points[i1][Y] < 0)

        if(Rstrad or Lstrad):
            # Compute x-coordinate of edge/ray intersection
            x = (shifted_points[i][X] * shifted_points[i1][Y] - shifted_points[i1][X] * shifted_points[i][Y]) / (shifted_points[i1][Y] - shifted_points[i][Y])
            if (Rstrad and x > 0):
                Rcross += 1
            if (Lstrad and x < 0):
                Lcross += 1

    # Different parity means on boundary
    if ((Rcross % 2) != (Lcross % 2)):
        return 'e'
    
    # Odd number of crossings means inside
    if((Rcross % 2) == 1):
        return 'i'
    
    # Even crossings means outside
    else:
        return 'o'
    
def ConstructIndependentSet(G: pg.PlanarGraph):
    I = set()
    marked_verteces = set()
    
    for vertex in G.vertices:
        if vertex.degree >= 9:
            marked_verteces.add(vertex)

    while len(marked_verteces) != len(G.vertices):
        for vertex in G.vertices:
            if vertex not in marked_verteces:
                marked_verteces.add(vertex)
                for edge in vertex.get_incident_edges():
                    marked_verteces.add(edge.destination())
                I.add(vertex)

    return I
        