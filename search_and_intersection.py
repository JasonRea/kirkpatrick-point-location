import primitives as prim
import polygon as poly

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
        Assigndi(p, c)
        return 'e'
    if prim.Between(a, b, d):
        Assigndi(p, d)
        return 'e'
    if prim.Between(c, d, a):
        Assigndi(p, a)
        return 'e'
    if prim.Between(c, d, b):
        Assigndi(p, b)
        return 'e'
    return '0'

def Assigndi(p: prim.Point, a: prim.Point):
    for i in range(DIM):
        p[i] = a[i]

def Dot(a: prim.Point, b: prim.Point):
    sum = 0
    
    for i in range(DIM):
        sum += a[i] * b[i]

    return sum

def InPoly1(q: prim.Point, P: poly.Polygon, n: int):
    Rcross = 0
    Lcross = 0

    # Shift q to the origin
    for i in range(n):
        for d in range(DIM):
            P.points[i][d] = P.points[i][d] - q[d]

    for i in range(n):
        if (P.points[i][X] == 0 and P.points[i][Y] == 0):
            return 'v'
        
        i1 = (i + n - 1) % n

        Rstrad = (P.points[i][Y] > 0) != (P.points[i1][Y] > 0)
        Lstrad = (P.points[i][Y] < 0) != (P.points[i1][Y] < 0)

        if(Rstrad or Lstrad):
            x = (P[i][X] * P[i1][Y] - P[i1][X] * P[i][Y]) / (P[i1][Y] - P[i][Y])
            if (Rstrad and x > 0):
                Rcross += 1
            if (Lstrad and x < 0):
                Lcross += 1

    if ((Rcross % 2) != (Lcross % 2)):
        return 'e'
    
    if((Rcross % 2) == 1):
        return 'i'
    
    else:
        return 'o'