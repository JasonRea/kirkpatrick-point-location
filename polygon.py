from typing import List
import primitives as prim

X = 0
Y = 1
DIM = 2
PMAX = 10000

class Polygon:
    def __init__(self, points: List[prim.Point] = None):
        if points is None:
            self.points = []
        else:
            self.points = points[:PMAX]  # Limit to PMAX points
        self.n = len(self.points)  # Number of points
    
    def __getitem__(self, index) -> prim.Point:
        return self.points[index]
    
    def __setitem__(self, index, value: prim.Point):
        self.points[index] = value
    
    def __len__(self) -> int:
        return self.n
    
    def __repr__(self):
        return f"Polygon(n={self.n}, points={self.points})"

# Triangulation by ear-clipping
def Triangulate():
    n = prim.vertices.prev.vnum + 1

    prim.EarInit()

    while (n > 3):
        v2 = prim.vertices
        
        while True:
            if (v2.ear):
                v3 = v2.next
                v4 = v3.next
                v1 = v2.prev
                v0 = v1.prev
                
                print(f"({v1.vnum, v3.vnum}) is a diagonal, vertex {v2.vnum} is an ear")

                # Update earity of diagonal
                v1.ear = prim.Diagonal(v0 ,v3)
                v3.ear = prim.Diagonal(v1, v4)

                # Cut off ear v2
                v1.next = v3
                v3.prev = v1
                prim.vertices = v3 # Incase head was v2
                n = n - 1
                break

            v2 = v2.next

            if(v2 == prim.vertices):
                print("No ear found")
                return


def create_polygon(points: List[tuple[int, int]]):
    vertex_list = [
        prim.Vertex(i, prim.Point(x, y)) 
        for i, (x, y) in enumerate(points)
    ]

    prim.link_vertices_circular(vertex_list)
    return prim.vertices