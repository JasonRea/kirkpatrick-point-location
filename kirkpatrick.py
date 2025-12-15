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
    """
    Add a large bounding triangle that contains the entire graph.

    This modifies the graph P in-place by:
    1. Adding 3 bounding vertices forming a large triangle
    2. Connecting the original polygon to the bounding triangle with diagonals
    3. Rebuilding faces from the edge structure

    Note: The graph P should already be a simple polygon before calling this.
    """
    original_vertices = list(P.vertices)

    min_x = min(vertex.point[X] for vertex in original_vertices)
    min_y = min(vertex.point[Y] for vertex in original_vertices)
    max_x = max(vertex.point[X] for vertex in original_vertices)
    max_y = max(vertex.point[Y] for vertex in original_vertices)

    width = max_x - min_x
    height = max_y - min_y
    padding = max(width, height) * 1  # Use larger padding to ensure containment

    bounding_points = [
        prim.Point(min_x - padding, min_y - padding),
        prim.Point(max_x + padding, min_y - padding),
        prim.Point((min_x + max_x) / 2, max_y + padding * 2)
    ]
    print(bounding_points)

    convex_hull = P.compute_convex_hull()

    for edge in P.edges:
        if edge.edge_type == pg.EdgeType.BOUNDARY:
            edge.edge_type = pg.EdgeType.INTERNAL

    for i in range(len(convex_hull)):
        next_i = (i + 1) % len(convex_hull)
        # Check if edge already exists
        v1, v2 = convex_hull[i], convex_hull[next_i]
        edge_exists = False
        for edge in P.edges:
            if ((edge.origin == v1 and edge.destination == v2) or
                (edge.origin == v2 and edge.destination == v1)):
                edge_exists = True
                break

        if not edge_exists:
            P.add_edge(v1, v2, pg.EdgeType.INTERNAL)

    bounding_vertices = []
    for point in bounding_points:
        bounding_vertices.append(P.add_vertex(point[X], point[Y]))

    P.add_edge(bounding_vertices[0], bounding_vertices[1], pg.EdgeType.BOUNDARY)
    P.add_edge(bounding_vertices[1], bounding_vertices[2], pg.EdgeType.BOUNDARY)
    P.add_edge(bounding_vertices[2], bounding_vertices[0], pg.EdgeType.BOUNDARY)

    leftmost = min(convex_hull, key=lambda v: v.point[X])
    rightmost = max(convex_hull, key=lambda v: v.point[X])
    topmost = max(convex_hull, key=lambda v: v.point[Y])
    chull_vertices = [leftmost, rightmost, topmost]

    for bv, cv in zip(bounding_vertices, chull_vertices):
        P.add_edge(bv, cv, pg.EdgeType.INTERNAL)

    P.rebuild_dcel()

def BuildDAG(P: pg.PlanarGraph):
    #TODO evaluate and fix ConstructNestedPolytopeHierarchy, BuildDag 
    #There is an issue with how we construct the nested polytope hierarchy
            
    BoundTriangle(P)
    P.triangulate()

    hierarchy = ConstructNestedPolytopeHierarchy(P)

    dag_nodes: list[DAGNode] = []

    for pslg in hierarchy:
        triangles = pslg.triangulate()
        print(f"Triangles: {triangles}")
        
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