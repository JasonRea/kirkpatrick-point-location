from typing import List, Optional
from enum import Enum

# Constants
X = 0
Y = 1

# Boolean enum
class Bool(Enum):
    FALSE = 0
    TRUE = 1

# Dimension constant
DIM = 2

# Point type: array of 2 integers [x, y]
class Point:
    def __init__(self, x: int = 0, y: int = 0):
        self.coords = [x, y]
    
    def __getitem__(self, index):
        return self.coords[index]
    
    def __setitem__(self, index, value):
        self.coords[index] = value
    
    def __repr__(self):
        return f"Point({self.coords[X]}, {self.coords[Y]})"

# Vertex structure
class Vertex:
    def __init__(self, vnum: int, v: Point, ear: bool = False):
        self.vnum = vnum          # Index
        self.v = v                 # Coordinates
        self.ear = ear             # TRUE iff an ear
        self.next: Optional[Vertex] = None
        self.prev: Optional[Vertex] = None
    
    def __repr__(self):
        return f"Vertex(vnum={self.vnum}, v={self.v}, ear={self.ear})"
    

# Vertex list
vertices: Optional[Vertex] = None

def link_vertices_circular(vertex_list: List[Vertex]):
    global vertices
    
    if not vertex_list:
        raise ValueError("Cannot create circular list from empty list")
    
    n = len(vertex_list)
    
    # Link each vertex to its next and previous
    for i in range(n):
        vertex_list[i].next = vertex_list[(i + 1) % n]  # Wrap around at end
        vertex_list[i].prev = vertex_list[(i - 1) % n]  # Wrap around at start
    
    vertices = vertex_list[0]  # Assign to global vertices

def Area2(a: Point, b: Point, c: Point):
    return (b[X] - a[X]) * (c[Y] - a[Y]) - (c[X] - a[X]) * (b[Y] - a[Y])

def AreaPoly2():
    sum = 0

    global vertices
    p = vertices
    a = p.next

    while True:
        sum += Area2(p.v, a.v, a.next.v)
        a = a.next
        if a.next == vertices:
            break

    return sum

def Left(a: Point, b: Point, c: Point):
    return Area2(a, b, c) > 0

def LeftOn(a: Point, b: Point, c: Point):
    return Area2(a, b, c) >= 0

def Collinear(a: Point, b: Point, c: Point):
    return Area2(a, b, c) == 0

def IntersectProp(a: Point, b: Point, c: Point, d: Point):
    #Eliminate collinear degeneracies
    if(
        Collinear(a, b, c) or
        Collinear(a, b, d) or
        Collinear(c, d, a) or
        Collinear(c, d, b)
    ):
        return False
    
    return(
        (Left(a, b, c) ^ Left(a, b, d)) and
        (Left(c, d, a) ^ Left(c, d, b))
    )

def Between(a: Point, b: Point, c: Point):
    if not Collinear(a, b, c):
        return False
    
    if(a[X] != b[X]):
        return (a[X] <= c[X] <= b[X]) or (a[X] >= c[X] >= b[X])
    else:
        return (a[Y] <= c[Y] <= b[Y]) or (a[Y] >= c[Y] >= b[Y])
    
def Intersect(a: Point, b: Point, c: Point, d: Point):
    if IntersectProp(a, b, c, d): 
        return True
    elif(
        Between(a, b, c) or
        Between(a, b, d) or
        Between(c, d, a) or
        Between(c, d, b)
    ):
        return True
    else:
        return False

def DiagonalIE(a: Vertex, b:Vertex):
    global vertices
    c = vertices

    while True:
        c1 = c.next
        if(
            (c != a) and 
            (c1 != a) and
            (c != b) and 
            (c1 != b) and
            Intersect(a.v, b.v, c.v, c1.v)
        ):
            return False
        c = c.next
        if c == vertices:
            break
    return True

def InCone(a: Vertex, b: Vertex):
    a1 = a.next
    a0 = a.prev

    if(LeftOn(a.v, a1.v, a0.v)):
        return (
            Left(a.v, b.v, a0.v) and
            Left(b.v, a.v, a1.v)
        )
    
    return not (
        LeftOn(a.v, b.v, a1.v) and
        LeftOn(b.v, a.v, a0.v)
    )

def Diagonal(a: Vertex, b: Vertex):
    return InCone(a, b) and InCone(b,a) and DiagonalIE(a, b)

def EarInit():
    global vertices
    v1 = vertices

    while True:
        v2 = v1.next
        v0 = v1.prev
        v1.ear = Diagonal(v0, v2)
        v1 = v1.next

        if(v1 == vertices):
            break

if __name__ == '__main__':
    # Create a simple triangle
    v1 = Vertex(0, Point(0, 0))
    v2 = Vertex(1, Point(4, 0))
    v3 = Vertex(2, Point(2, 3))

    # Link them in a circular list
    v1.next = v2
    v2.next = v3
    v3.next = v1

    v1.prev = v3
    v2.prev = v1
    v3.prev = v2

    vertices = v1

    # Calculate twice the area
    twice_area = AreaPoly2()
    print(f"Twice the area: {twice_area}")
    print(f"Actual area: {twice_area / 2}")