import planar_graph as pg
import polygon as poly
from search_and_intersection import *

class DAGNode():
    def __init__(self, value: tuple[int, int, int]):
        self.value = value
        self.children = []

    def add_child(self, child):
        self.children.append(child)

    def polygonalize(self, P: pg.PlanarGraph):
        points = []

        for vertex in self.value:
            points.append(P.get_coords_by_vertex_id(vertex))

        return poly.Polygon(points)

def BoundTriangle(P: pg.PlanarGraph):
    # add bounding triangle
    min_x, min_y = min(vertex.point[X] for vertex in P.vertices), min(vertex.point[Y] for vertex in P.vertices)
    max_x, max_y = max(vertex.point[X] for vertex in P.vertices), max(vertex.point[Y] for vertex in P.vertices)

    width = max_x - min_x
    height = max_y - min_y
    padding = max(width, height) * 0.1

    points = [prim.Point(min_x - padding, min_y - padding), 
              prim.Point(max_x + padding, min_y - padding), 
              prim.Point((min_x + max_x) / 2, max_y + padding * 2)]
    
    for point in points: P.add_vertex(point[X], point[Y]) 

def BuildDAG(P: pg.PlanarGraph):
    # what needs happen when we build the dag?

    # add bounding triangle
    # construct nested polytope hierarchy (TODO: review and fix)
    # i think the below steps should be in polytope hierarchy construction
    # for each PSLG in the hierarchy
        # initialize dag nodes for each triangle
        # link nodes
            
    BoundTriangle(P)
    P.triangulate()

    hierarchy = ConstructNestedPolytopeHierarchy(P)

    dag_nodes: list[DAGNode] = []

    for pslg in hierarchy:
        triangles = pslg.triangulate()
        
        for triangle in triangles:
            #initialize new nodes in current level of hierarchy
            new_node = DAGNode(triangle)
            dag_nodes.append(new_node)

            #link nodes
            for node in dag_nodes:

                #turn these tuples into polygons
                tri_one = node.polygonalize(hierarchy[0]) # use P_0 for coordinate lookup
                tri_two = new_node.polygonalize(hierarchy[0])

                if TriTriInt(tri_one, tri_two):
                    new_node.add_child(node)

    return dag_nodes[-1] # return head of DAG