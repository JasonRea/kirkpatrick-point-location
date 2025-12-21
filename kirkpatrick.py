import planar_graph as pg
import polygon as poly
from search_and_intersection import *

class DAGNode():
    """
    A node in the DAG representing a triangle in the hierarchical triangulation.

    The DAG is built bottom-up:
    - Level 0 (bottom): Original triangulation with all vertices
    - Level k (top): Final bounding triangle only

    Each node points to children in the level below that it overlaps with.
    """
    def __init__(self, triangle: tuple[int, int, int], level: int):
        self.triangle = triangle  # Tuple of 3 vertex IDs (sorted)
        self.level = level        # Which level in the hierarchy (0 = bottom)
        self.children = []        # Nodes in the level below that overlap with this triangle

    def add_child(self, child: 'DAGNode'):
        """Add a child node (from the level below) that overlaps with this triangle"""
        if child not in self.children:
            self.children.append(child)

    def to_polygon(self, P: pg.PlanarGraph) -> poly.Polygon:
        """Convert this triangle to a Polygon object using coordinates from graph P"""
        points = []
        for vertex_id in self.triangle:
            coords = P.get_coords_by_vertex_id(vertex_id)
            if coords is None:
                raise ValueError(f"Vertex ID {vertex_id} not found in graph")
            points.append(prim.Point(coords[0], coords[1]))
        return poly.Polygon(points)

    def __repr__(self):
        return f"DAGNode(tri={self.triangle}, level={self.level}, children={len(self.children)})"

    def __hash__(self):
        return hash(self.triangle)

    def __eq__(self, other):
        if not isinstance(other, DAGNode):
            return False
        return self.triangle == other.triangle

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
    # Tiebreaker: if multiple vertices have same max Y, choose the one with greatest X
    topmost = max(convex_hull, key=lambda v: (v.point[Y], v.point[X]))
    chull_vertices = [leftmost, rightmost, topmost]

    for bv, cv in zip(bounding_vertices, chull_vertices):
        P.add_edge(bv, cv, pg.EdgeType.INTERNAL)

    P.rebuild_dcel()

def BuildDAG(P: pg.PlanarGraph) -> DAGNode:
    """
    Build a Directed Acyclic Graph (DAG) for point location queries.

    The DAG is constructed bottom-up from a hierarchy of triangulations.
    Each node at level i points to overlapping nodes at level i-1.
    """
    # Add bounding triangle and triangulate
    BoundTriangle(P)
    P.triangulate()

    # Build hierarchy of nested triangulations
    hierarchy = ConstructNestedPolytopeHierarchy(P)

    if not hierarchy:
        raise ValueError("Empty hierarchy - cannot build DAG")

    # Build DAG bottom-up, level by level
    # Each unique triangle should only appear at ONE level (the first level it appears in)
    seen_triangles: set[tuple[int, int, int]] = set()

    # Track nodes at each level for linking purposes
    levels: list[list[DAGNode]] = []

    # Process each level in the hierarchy
    for level_idx, pslg in enumerate(hierarchy):
        triangles = pslg.triangulate()
        level_nodes = []

        for triangle in triangles:
            triangle_normalized = tuple(sorted(triangle))

            if triangle_normalized not in seen_triangles:
                node = DAGNode(triangle_normalized, level_idx)
                level_nodes.append(node)
                seen_triangles.add(triangle_normalized)

        # Link current level nodes to previous level nodes
        if level_idx > 0:
            previous_level = levels[level_idx - 1]
            previous_pslg = hierarchy[level_idx - 1]
            current_pslg = pslg

            for current_node in level_nodes:
                try:
                    current_polygon = current_node.to_polygon(current_pslg)
                except ValueError:
                    continue

                # Check which nodes from the previous level overlap
                for prev_node in previous_level:
                    try:
                        # Use previous level's PSLG for previous node
                        prev_polygon = prev_node.to_polygon(previous_pslg)
                    except ValueError:
                        continue

                    # Link if triangles intersect
                    if TriTriInt(current_polygon, prev_polygon):
                        current_node.add_child(prev_node)

        levels.append(level_nodes)

    # Return the root
    if not levels:
        raise ValueError("No levels created - DAG construction failed")

    top_level = levels[-1]
    if len(top_level) != 1:
        print(f"Warning: Expected 1 node at top level, found {len(top_level)}")

    root = top_level[0] if top_level else None

    # Store metadata on the root for debugging/analysis
    if root:
        root.all_levels = levels
        # Collect all unique nodes from all levels
        all_nodes_list = []
        for level in levels:
            all_nodes_list.extend(level)
        root.all_nodes = all_nodes_list

    return root