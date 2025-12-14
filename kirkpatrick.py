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
    
'''
NOTE

Currently having a problem with face subdivisions.
To bound the PSLG by a triangle we currently do the following
    - Find the convex hull
    - Find the bounding triangle
    - Add new edges of the convex hull to the PSLG
    - Add the bounding triangle edges to the PSLG
    - Link the following vertices creating new edges in the PSLG
        - Bottom-left triangle vertex & left most chull vertex
        - Bottom-right triangle vertex & right most chull vertex
        - Top triangle vertex & top most chull vertex
The issue arises when faces have to be restructured.
I am getting a correct number of vertices and edges but incorrect number of faces.
    - Square test case is working properly
    - Triangle test case is not
'''


def BoundTriangle(P: pg.PlanarGraph):
    """
    Add a large bounding triangle that contains the entire graph.

    This modifies the graph P in-place by:
    1. Adding 3 bounding vertices forming a large triangle
    2. Connecting the original polygon to the bounding triangle with diagonals
    3. Rebuilding faces from the edge structure

    Note: The graph P should already be a simple polygon before calling this.
    """
    # Store original vertices (before adding bounding triangle)
    original_vertices = list(P.vertices)

    # Compute bounding box with padding
    min_x = min(vertex.point[X] for vertex in original_vertices)
    min_y = min(vertex.point[Y] for vertex in original_vertices)
    max_x = max(vertex.point[X] for vertex in original_vertices)
    max_y = max(vertex.point[Y] for vertex in original_vertices)

    width = max_x - min_x
    height = max_y - min_y
    padding = max(width, height) * 1  # Use larger padding to ensure containment

    # Create 3 bounding vertices forming a large triangle
    # Bottom-left, bottom-right, top-center
    bounding_points = [
        prim.Point(min_x - padding, min_y - padding),
        prim.Point(max_x + padding, min_y - padding),
        prim.Point((min_x + max_x) / 2, max_y + padding * 2)
    ]
    print(bounding_points)

    # Compute the convex hull before adding bounding triangle
    convex_hull = P.compute_convex_hull()

    # Change all existing boundary edges to internal edges
    # because the bounding triangle will become the new boundary
    for edge in P.edges:
        if edge.edge_type == pg.EdgeType.BOUNDARY:
            edge.edge_type = pg.EdgeType.INTERNAL

    # Add convex hull edges to form the hole boundary (only if they don't exist)
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

        # Only add if edge doesn't exist
        if not edge_exists:
            P.add_edge(v1, v2, pg.EdgeType.INTERNAL)

    # Add the three bounding vertices
    bounding_vertices = []
    for point in bounding_points:
        bounding_vertices.append(P.add_vertex(point[X], point[Y]))

    # Connect the bounding vertices to form the outer triangle (new boundary)
    P.add_edge(bounding_vertices[0], bounding_vertices[1], pg.EdgeType.BOUNDARY)
    P.add_edge(bounding_vertices[1], bounding_vertices[2], pg.EdgeType.BOUNDARY)
    P.add_edge(bounding_vertices[2], bounding_vertices[0], pg.EdgeType.BOUNDARY)

    # The bounding triangle forms the outer boundary, convex hull forms the hole
    # Connect the topmost bounding triangle vertex to the top chull vertex
    # Connect the rightmost chull vertex to the bottom right tri vertex
    # connect the leftmost chull vertex to the bottom left tri vertex

    # Store the leftmost, rightmost, and top most convex hull vertex
    leftmost = min(convex_hull, key=lambda v: v.point[X])
    rightmost = max(convex_hull, key=lambda v: v.point[X])
    topmost = max(convex_hull, key=lambda v: v.point[Y])
    chull_vertices = [leftmost, rightmost, topmost]

    for bv, cv in zip(bounding_vertices, chull_vertices):
        P.add_edge(bv, cv, pg.EdgeType.INTERNAL)

    # Rebuild DCEL and faces after adding all edges
    P.rebuild_dcel()
    print(f"After rebuild_dcel: {P}, Faces: {len(P.faces)}")

    # Hold off on triangluation after rebuilding faces.
    #P.triangulate()

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

# TODO fix kirk to start with a fully triangulated graph