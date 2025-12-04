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
    
#
# IMPORTANT NOTE!!!!!
#
#
# The main issue arises from after trying to bound the triangle around the current PSLG
# What happens is we have two disconnected graph components
# We need to connect them
# This is how
# Take the convex hull of the current PSLG
# Now we have the bounding triangle and a 'hole'
# Triangulate this polygon with a hole [O(n log n) I believe?] and internal faces
# Add the new edges and faces into the winged-edge
# bada bing bada boom
#

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
    padding = max(width, height) * 0.5  # Use larger padding to ensure containment

    # Create 3 bounding vertices forming a large triangle
    # Bottom-left, bottom-right, top-center
    bounding_points = [
        prim.Point(min_x - padding, min_y - padding),
        prim.Point(max_x + padding, min_y - padding),
        prim.Point((min_x + max_x) / 2, max_y + padding * 2)
    ]
    print(bounding_points)

    # Add the three bounding vertices
    bounding_vertices = []
    for point in bounding_points:
        bounding_vertices.append(P.add_vertex(point[X], point[Y]))

    # Connect each original vertex to all 3 bounding vertices
    # This ensures the graph is connected and can be triangulated
    for orig_v in original_vertices:
        for bound_v in bounding_vertices:
            P.add_edge(orig_v, bound_v, pg.EdgeType.INTERNAL)

    # Connect the bounding vertices to form the outer triangle
    P.add_edge(bounding_vertices[0], bounding_vertices[1], pg.EdgeType.BOUNDARY)
    P.add_edge(bounding_vertices[1], bounding_vertices[2], pg.EdgeType.BOUNDARY)
    P.add_edge(bounding_vertices[2], bounding_vertices[0], pg.EdgeType.BOUNDARY)

    # Rebuild all faces from the edge structure
    P.rebuild_faces()
    print(f"After rebuild_faces: {P}, Faces: {len(P.faces)}") 

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