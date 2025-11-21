import primitives as prim
import polygon as poly

X = 0
Y = 1
DIM = 2

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