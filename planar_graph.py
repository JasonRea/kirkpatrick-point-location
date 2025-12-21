"""
Planar Graph with DCEL (Doubly Connected Edge List) Data Structure

This module implements a planar graph using the pydcel library, which provides
a robust DCEL implementation. The DCEL structure efficiently represents planar
subdivisions and supports topological queries and modifications.

DCEL Structure:
--------------
- Vertices: Points in 2D space
- Half-Edges: Directed edges with twin, next, prev pointers
- Faces: Regions bounded by half-edge cycles

This wrapper provides compatibility with the existing codebase while using
pydcel's efficient DCEL implementation underneath.
"""

from typing import Optional, List, Tuple
from enum import Enum
import primitives as prim
from dcel import Dcel, Vertex, HalfEdge, Face as DcelFace


class EdgeType(Enum):
    """Classification of edges in the planar graph"""
    BOUNDARY = "boundary"
    DIAGONAL = "diagonal"
    INTERNAL = "internal"
    EXTERNAL = "external"


class FaceType(Enum):
    """Classification of faces"""
    INTERIOR = "interior"
    EXTERIOR = "exterior"
    UNBOUNDED = "unbounded"


class GraphVertex:
    """
    Wrapper for pydcel Vertex with additional metadata.
    """
    def __init__(self, dcel_vertex: Vertex, vertex_id: int, dcel_index: int, graph=None):
        self.dcel_vertex = dcel_vertex
        self.id = vertex_id
        self.dcel_index = dcel_index  # Index in DCEL vertices list
        self.point = prim.Point(dcel_vertex.x, dcel_vertex.y)
        self._graph = graph  # Reference to PlanarGraph for edge lookups

    @property
    def degree(self) -> int:
        """Count incident edges"""
        # pydcel has degree attribute
        return self.dcel_vertex.degree

    @property
    def incident_edge(self):
        """Get incident half-edge"""
        hedges = self.dcel_vertex.hedgelist
        return hedges[0] if hedges else None

    # NOTE we are not retrieving incident edges correctly
    def get_incident_edges(self) -> List['Edge']:
        """Get all edges incident to this vertex"""
        # pydcel has hedgelist attribute - but we need to return Edge wrappers
        incident_edges = []
        for hedge in self.dcel_vertex.hedgelist:
            # Look up the Edge wrapper for this half-edge
            # We need to access the graph's edge list
            # For now, we'll need to search through edges
            # This is not ideal but necessary without storing back-references
            if hasattr(self, '_graph') and self._graph:
                for edge in self._graph.edges:
                    if edge.half_edge == hedge or edge.half_edge.twin == hedge:
                        if edge not in incident_edges:
                            incident_edges.append(edge)
                        break
        return incident_edges

    def get_neighbors(self) -> List['GraphVertex']:
        """Get all neighboring vertices"""
        neighbors = []
        hedges = self.dcel_vertex.hedgelist

        for hedge in hedges:
            # Each half-edge goes from this vertex to a neighbor
            neighbor_vertex = hedge.twin.origin
            neighbors.append(neighbor_vertex)

        return neighbors

    def __repr__(self):
        return f"V{self.id}({self.point[prim.X]:.2f}, {self.point[prim.Y]:.2f})"

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, GraphVertex):
            return False
        return self.id == other.id


class Edge:
    """
    Wrapper for pydcel HalfEdge with edge type metadata.
    Represents an undirected edge (pair of half-edges).
    """
    def __init__(self, half_edge: HalfEdge, edge_id: int, edge_type: EdgeType = EdgeType.INTERNAL, graph=None):
        self.half_edge = half_edge  # Primary half-edge
        self.id = edge_id
        self.edge_type = edge_type
        self._graph = graph  # Reference to PlanarGraph for vertex mapping

    @property
    def origin(self):
        """Origin vertex of primary half-edge"""
        dcel_vertex = self.half_edge.origin
        if self._graph and dcel_vertex in self._graph._vertex_map:
            return self._graph._vertex_map[dcel_vertex]
        return dcel_vertex

    @property
    def destination(self):
        """Destination vertex of primary half-edge"""
        dcel_vertex = self.half_edge.twin.origin
        if self._graph and dcel_vertex in self._graph._vertex_map:
            return self._graph._vertex_map[dcel_vertex]
        return dcel_vertex

    @property
    def left_face(self):
        """Face on the left when traversing origin->dest"""
        return self.half_edge.incidentFace

    @property
    def right_face(self):
        """Face on the right when traversing origin->dest"""
        return self.half_edge.twin.incidentFace

    @left_face.setter
    def left_face(self, face):
        self.half_edge.incidentFace = face

    @right_face.setter
    def right_face(self, face):
        self.half_edge.twin.incidentFace = face

    @property
    def next_at_origin(self):
        """Next half-edge CCW around origin"""
        return self.half_edge.twin.next

    @property
    def prev_at_origin(self):
        """Previous half-edge CCW around origin"""
        return self.half_edge.prev.twin

    @property
    def next_at_dest(self):
        """Next half-edge CCW around destination"""
        return self.half_edge.next

    @property
    def prev_at_dest(self):
        """Previous half-edge CCW around destination"""
        return self.half_edge.twin.prev

    def __repr__(self):
        if hasattr(self.origin, 'id'):
            origin_id = self.origin.id
            dest_id = self.destination.id
        else:
            origin_id = id(self.origin)
            dest_id = id(self.destination)
        return f"E{self.id}({origin_id}→{dest_id})"

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, Edge):
            return False
        return self.id == other.id


class Face:
    """
    Wrapper for pydcel Face with type metadata.
    """
    def __init__(self, dcel_face: DcelFace, face_id: int, face_type: FaceType = FaceType.INTERIOR):
        self.dcel_face = dcel_face
        self.id = face_id
        self.face_type = face_type

    @property
    def boundary_edge(self):
        """Get a boundary half-edge"""
        return self.dcel_face.wedge if self.dcel_face else None

    def get_boundary_vertices(self) -> List:
        """Get vertices bounding this face in order"""
        if not self.dcel_face:
            return []
        # pydcel Face has vertices() method
        return list(self.dcel_face.vertices())

    def get_boundary_edges(self) -> List:
        """Get half-edges bounding this face"""
        if not self.dcel_face:
            return []
        # pydcel Face has edges() method
        return list(self.dcel_face.edges())

    def area(self) -> float:
        """Compute signed area of face"""
        vertices = self.get_boundary_vertices()
        if len(vertices) < 3:
            return 0.0

        area = 0.0
        n = len(vertices)
        for i in range(n):
            j = (i + 1) % n
            if hasattr(vertices[i], 'x'):
                x1, y1 = vertices[i].x, vertices[i].y
                x2, y2 = vertices[j].x, vertices[j].y
            else:
                x1, y1 = vertices[i].point[prim.X], vertices[i].point[prim.Y]
                x2, y2 = vertices[j].point[prim.X], vertices[j].point[prim.Y]
            area += x1 * y2 - x2 * y1
        return area / 2.0

    def __repr__(self):
        return f"Face{self.id}({self.face_type.value})"

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, Face):
            return False
        return self.id == other.id


class PlanarGraph:
    """
    Planar graph using DCEL (via pydcel) for efficient topological operations.
    """

    def __init__(self):
        self.dcel = Dcel()
        self.vertices: List[GraphVertex] = []
        self.edges: List[Edge] = []
        self.faces: List[Face] = []
        self.unbounded_face: Optional[Face] = None

        self._next_vertex_id = 0
        self._next_edge_id = 0
        self._next_face_id = 0

        # Mappings from dcel objects to wrappers
        self._vertex_map = {}  # dcel.Vertex -> GraphVertex
        self._edge_map = {}    # dcel.HalfEdge -> Edge
        self._face_map = {}    # dcel.Face -> Face

    def add_vertex(self, x: float, y: float) -> GraphVertex:
        """Add a vertex at the given coordinates"""
        # Check if vertex already exists at this location
        for v in self.vertices:
            if abs(v.point[prim.X] - x) < 1e-9 and abs(v.point[prim.Y] - y) < 1e-9:
                return v

        dcel_vertex = self.dcel.add_vertex(x, y)
        dcel_index = len(self.dcel.vertices) - 1  # Index of the vertex just added
        graph_vertex = GraphVertex(dcel_vertex, self._next_vertex_id, dcel_index, graph=self)
        self._next_vertex_id += 1

        self.vertices.append(graph_vertex)
        self._vertex_map[dcel_vertex] = graph_vertex

        return graph_vertex

    def add_edge(self, v1: GraphVertex, v2: GraphVertex, edge_type: EdgeType = EdgeType.INTERNAL) -> Edge:
        """Add an edge between two vertices"""
        # Add edge to DCEL using vertex indices
        half_edge_pair = self.dcel.add_edge(v1.dcel_index, v2.dcel_index)
        half_edge = half_edge_pair[0]  # Get first half-edge from the pair

        # Create Edge wrapper with graph reference
        edge = Edge(half_edge, self._next_edge_id, edge_type, graph=self)
        self._next_edge_id += 1

        self.edges.append(edge)
        self._edge_map[half_edge] = edge
        self._edge_map[half_edge.twin] = edge  # Both half-edges map to same Edge

        return edge

    def remove_vertex(self, vertex: GraphVertex) -> None:
        """
        Remove a vertex from the graph.

        This removes the vertex and all edges incident to it, then rebuilds
        the DCEL to update the face structure.

        Args:
            vertex: The GraphVertex to remove (can be passed by vertex object or vertex ID)
        """
        # Handle both GraphVertex and vertex ID
        if isinstance(vertex, int):
            # Find vertex by ID
            vertex_to_remove = None
            for v in self.vertices:
                if v.id == vertex:
                    vertex_to_remove = v
                    break
            if vertex_to_remove is None:
                raise ValueError(f"Vertex with ID {vertex} not found")
            vertex = vertex_to_remove

        # Remove all edges incident to this vertex
        edges_to_remove = []
        for edge in self.edges:
            if edge.origin == vertex or edge.destination == vertex:
                edges_to_remove.append(edge)

        for edge in edges_to_remove:
            self.edges.remove(edge)
            # Remove from edge map
            if edge.half_edge in self._edge_map:
                del self._edge_map[edge.half_edge]
            if edge.half_edge.twin in self._edge_map:
                del self._edge_map[edge.half_edge.twin]

        # Remove vertex from vertex list
        self.vertices.remove(vertex)

        # Remove from vertex map
        if vertex.dcel_vertex in self._vertex_map:
            del self._vertex_map[vertex.dcel_vertex]

        # Rebuild DCEL to update face structure
        self.rebuild_dcel()

    def from_polygon(self, points: List[Tuple[float, float]]) -> 'PlanarGraph':
        """
        Create a planar graph from a polygon (list of points).
        """
        if len(points) < 3:
            raise ValueError("Polygon must have at least 3 points")

        # Build edges list (indices into points list)
        edges = []
        for i in range(len(points)):
            j = (i + 1) % len(points)
            edges.append((i, j))

        # Build DCEL with all data at once
        self.dcel.build_dcel(points, edges)

        # Create GraphVertex wrappers for all DCEL vertices
        for idx, dcel_vertex in enumerate(self.dcel.vertices):
            graph_vertex = GraphVertex(dcel_vertex, self._next_vertex_id, idx, graph=self)
            self._next_vertex_id += 1
            self.vertices.append(graph_vertex)
            self._vertex_map[dcel_vertex] = graph_vertex

        # Create Edge wrappers for all DCEL half-edges
        seen_edges = set()
        for hedge in self.dcel.hedges:
            # Only create one Edge per pair of half-edges
            hedge_id = id(hedge)
            twin_id = id(hedge.twin)
            if twin_id not in seen_edges:
                edge = Edge(hedge, self._next_edge_id, EdgeType.BOUNDARY, graph=self)
                self._next_edge_id += 1
                self.edges.append(edge)
                self._edge_map[hedge] = edge
                self._edge_map[hedge.twin] = edge
                seen_edges.add(hedge_id)
                seen_edges.add(twin_id)

        # Rebuild faces after DCEL is built
        self.rebuild_faces()

        return self

    def rebuild_dcel(self):
        """
        Rebuild the entire DCEL structure from current vertices and edges.
        This is needed after adding edges incrementally.
        """
        # Extract vertex coordinates
        vertex_coords = []
        vertex_map_old_to_new = {}

        for i, v in enumerate(self.vertices):
            vertex_coords.append((v.point[prim.X], v.point[prim.Y]))
            vertex_map_old_to_new[v.dcel_index] = i

        # Extract edges as index pairs
        edge_pairs = []
        seen_pairs = set()
        for edge in self.edges:
            # Get indices - find position in current vertices list
            v1_idx = self.vertices.index(edge.origin)
            v2_idx = self.vertices.index(edge.destination)

            # Add edge in canonical form (smaller index first)
            pair = tuple(sorted([v1_idx, v2_idx]))
            if pair not in seen_pairs:
                edge_pairs.append((v1_idx, v2_idx))
                seen_pairs.add(pair)

        # Rebuild DCEL
        self.dcel = Dcel()
        self.dcel.build_dcel(vertex_coords, edge_pairs)

        # Rebuild vertex mappings
        self._vertex_map.clear()
        for i, v in enumerate(self.vertices):
            v.dcel_vertex = self.dcel.vertices[i]
            v.dcel_index = i
            v._graph = self  # Ensure graph reference is maintained
            self._vertex_map[v.dcel_vertex] = v

        # Rebuild edge mappings
        self._edge_map.clear()
        self.edges.clear()
        self._next_edge_id = 0

        seen_edges = set()
        for hedge in self.dcel.hedges:
            hedge_id = id(hedge)
            twin_id = id(hedge.twin)
            if twin_id not in seen_edges:
                edge = Edge(hedge, self._next_edge_id, EdgeType.INTERNAL, graph=self)
                self._next_edge_id += 1
                self.edges.append(edge)
                self._edge_map[hedge] = edge
                self._edge_map[hedge.twin] = edge
                seen_edges.add(hedge_id)
                seen_edges.add(twin_id)

        # Rebuild faces
        self.rebuild_faces()

    def rebuild_faces(self):
        """
        Rebuild face list from DCEL.
        """
        self.faces = []
        self._face_map = {}
        self._next_face_id = 0

        # Iterate over DCEL faces
        for dcel_face in self.dcel.faces:
            face = Face(dcel_face, self._next_face_id, FaceType.INTERIOR)
            self._next_face_id += 1
            self.faces.append(face)
            self._face_map[dcel_face] = face

        # Determine unbounded face (largest area)
        if self.faces:
            self.unbounded_face = max(self.faces, key=lambda f: abs(f.area()))
            self.unbounded_face.face_type = FaceType.UNBOUNDED

    def triangulate(self) -> List[Tuple[int, int, int]]:
        """
        Triangulate all faces using ear clipping algorithm.
        Adds diagonal edges to subdivide non-triangular faces.
        Returns list of triangles as (v0_id, v1_id, v2_id) tuples.
        """
        triangles = []
        all_diagonals = []  # Collect all diagonals to add at once

        # Collect faces to triangulate (make a copy since we'll modify the graph)
        faces_to_process = []
        for face in self.faces:
            if face == self.unbounded_face:
                continue

            vertices = face.get_boundary_vertices()
            if len(vertices) < 3:
                continue  # Skip degenerate faces

            faces_to_process.append((face, vertices))

        # Process each face
        for face, vertices in faces_to_process:
            if len(vertices) == 3:
                # Already a triangle
                v_ids = [self._vertex_map[v].id if v in self._vertex_map else v.id for v in vertices]
                triangles.append(tuple(sorted(v_ids)))
                continue

            # Ear clipping for polygons with 4+ vertices
            face_triangles, diagonals = self._triangulate_face(vertices)
            triangles.extend(face_triangles)
            all_diagonals.extend(diagonals)

        # Add all diagonal edges at once and rebuild DCEL once
        if all_diagonals:
            for gv0, gv2 in all_diagonals:
                self.add_edge(gv0, gv2, EdgeType.INTERNAL)

            # Rebuild DCEL to update face structure
            self.rebuild_dcel()

        return triangles

    def _triangulate_face(self, vertices: List) -> tuple[List[Tuple[int, int, int]], List[tuple]]:
        """
        Triangulate a single face using ear clipping.
        Returns (triangles, diagonals) where:
        - triangles: list of (v0_id, v1_id, v2_id) tuples
        - diagonals: list of (GraphVertex, GraphVertex) tuples for edges to add
        """
        triangles = []
        diagonals_to_add = []  # Store diagonals to add after ear clipping

        # Convert to list of vertex indices
        vert_list = list(vertices)
        n = len(vert_list)

        if n < 3:
            return triangles, diagonals_to_add

        # Helper function to get GraphVertex from a vertex object
        def get_graph_vertex(v):
            if v in self._vertex_map:
                return self._vertex_map[v]
            elif hasattr(v, 'id'):
                # It's already a GraphVertex
                return v
            return None

        # Ear clipping algorithm
        attempts = 0
        max_attempts = n * n

        while len(vert_list) > 3 and attempts < max_attempts:
            attempts += 1
            ear_found = False

            for i in range(len(vert_list)):
                if self._is_ear(vert_list, i):
                    # Found an ear, clip it
                    prev_idx = (i - 1) % len(vert_list)
                    next_idx = (i + 1) % len(vert_list)

                    v0 = vert_list[prev_idx]
                    v1 = vert_list[i]
                    v2 = vert_list[next_idx]

                    # Get vertex IDs
                    if v0 in self._vertex_map:
                        v0_id = self._vertex_map[v0].id
                    elif hasattr(v0, 'id'):
                        v0_id = v0.id
                    else:
                        v0_id = id(v0)

                    if v1 in self._vertex_map:
                        v1_id = self._vertex_map[v1].id
                    elif hasattr(v1, 'id'):
                        v1_id = v1.id
                    else:
                        v1_id = id(v1)

                    if v2 in self._vertex_map:
                        v2_id = self._vertex_map[v2].id
                    elif hasattr(v2, 'id'):
                        v2_id = v2.id
                    else:
                        v2_id = id(v2)

                    triangles.append(tuple(sorted([v0_id, v1_id, v2_id])))

                    # Add diagonal edge from v0 to v2 (skipping the ear vertex v1)
                    # This edge subdivides the face
                    gv0 = get_graph_vertex(v0)
                    gv2 = get_graph_vertex(v2)

                    if gv0 and gv2 and prev_idx != next_idx:
                        # Check if edge doesn't already exist
                        edge_exists = False
                        for edge in self.edges:
                            if ((edge.origin == gv0 and edge.destination == gv2) or
                                (edge.origin == gv2 and edge.destination == gv0)):
                                edge_exists = True
                                break

                        if not edge_exists:
                            diagonals_to_add.append((gv0, gv2))

                    # Remove the ear vertex
                    vert_list.pop(i)
                    ear_found = True
                    break

            if not ear_found:
                break

        # Add final triangle
        if len(vert_list) == 3:
            v_ids = []
            for v in vert_list:
                if v in self._vertex_map:
                    v_ids.append(self._vertex_map[v].id)
                elif hasattr(v, 'id'):
                    v_ids.append(v.id)
                else:
                    v_ids.append(id(v))
            triangles.append(tuple(sorted(v_ids)))

        return triangles, diagonals_to_add

    def _is_ear(self, vertices: List, i: int) -> bool:
        """Check if vertex i is an ear of the polygon"""
        n = len(vertices)
        prev_idx = (i - 1) % n
        next_idx = (i + 1) % n

        v_prev = vertices[prev_idx]
        v_curr = vertices[i]
        v_next = vertices[next_idx]

        # Get coordinates
        def get_coords(v):
            if hasattr(v, 'x'):
                return v.x, v.y
            elif hasattr(v, 'point'):
                return v.point[prim.X], v.point[prim.Y]
            else:
                return v[0], v[1]

        x0, y0 = get_coords(v_prev)
        x1, y1 = get_coords(v_curr)
        x2, y2 = get_coords(v_next)

        # Check if triangle is oriented correctly (CCW)
        cross = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
        if cross <= 0:
            return False

        # Check if any other vertex is inside this triangle
        for j in range(n):
            if j == prev_idx or j == i or j == next_idx:
                continue

            v_test = vertices[j]
            x, y = get_coords(v_test)

            if self._point_in_triangle(x, y, x0, y0, x1, y1, x2, y2):
                return False

        return True

    def _point_in_triangle(self, px: float, py: float,
                           x0: float, y0: float,
                           x1: float, y1: float,
                           x2: float, y2: float) -> bool:
        """Check if point (px, py) is inside triangle (x0,y0), (x1,y1), (x2,y2)"""
        def sign(p1x, p1y, p2x, p2y, p3x, p3y):
            return (p1x - p3x) * (p2y - p3y) - (p2x - p3x) * (p1y - p3y)

        d1 = sign(px, py, x0, y0, x1, y1)
        d2 = sign(px, py, x1, y1, x2, y2)
        d3 = sign(px, py, x2, y2, x0, y0)

        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

        return not (has_neg and has_pos)

    def get_triangles(self) -> List[Tuple[int, int, int]]:
        """
        Get all existing triangular faces (faces with exactly 3 vertices).
        """
        triangles = []
        for face in self.faces:
            if face == self.unbounded_face:
                continue

            vertices = face.get_boundary_vertices()
            if len(vertices) == 3:
                v_ids = []
                for v in vertices:
                    if v in self._vertex_map:
                        v_ids.append(self._vertex_map[v].id)
                    elif hasattr(v, 'id'):
                        v_ids.append(v.id)
                    else:
                        v_ids.append(id(v))
                triangles.append(tuple(v_ids))

        return triangles

    def get_vertex_by_id(self, vertex_id: int) -> Optional[GraphVertex]:
        """Get vertex by its ID"""
        for v in self.vertices:
            if v.id == vertex_id:
                return v
        return None

    def get_coords_by_vertex_id(self, vertex_id: int) -> Optional[Tuple[float, float]]:
        """Get coordinates of a vertex by its ID"""
        v = self.get_vertex_by_id(vertex_id)
        if v:
            return (v.point[prim.X], v.point[prim.Y])
        return None

    def euler_characteristic(self) -> int:
        """Compute Euler characteristic: V - E + F"""
        return len(self.vertices) - len(self.edges) + len(self.faces)

    def compute_convex_hull(self) -> List[GraphVertex]:
        """
        Compute convex hull of all vertices using Gift Wrapping algorithm.
        Returns vertices in CCW order.
        """
        if len(self.vertices) < 3:
            return list(self.vertices)

        # Find leftmost point
        leftmost = min(self.vertices, key=lambda v: (v.point[prim.X], v.point[prim.Y]))

        hull = []
        current = leftmost

        while True:
            hull.append(current)

            # Find the most counter-clockwise point from current
            next_vertex = self.vertices[0]

            for candidate in self.vertices:
                if candidate == current:
                    continue

                # Cross product to determine orientation
                x1 = next_vertex.point[prim.X] - current.point[prim.X]
                y1 = next_vertex.point[prim.Y] - current.point[prim.Y]
                x2 = candidate.point[prim.X] - current.point[prim.X]
                y2 = candidate.point[prim.Y] - current.point[prim.Y]

                cross = x1 * y2 - y1 * x2

                if next_vertex == current or cross < 0:
                    next_vertex = candidate

            current = next_vertex

            # Stop when we return to the start
            if current == leftmost:
                break

        return hull

    def print_structure(self):
        """Print the graph structure for debugging"""
        print(self)
        print(f"Euler characteristic: {self.euler_characteristic()}")
        print()

        print(f"Vertices ({len(self.vertices)}):")
        for v in self.vertices:
            neighbors = v.get_neighbors()
            neighbor_ids = [self._vertex_map[n].id if n in self._vertex_map else id(n) for n in neighbors]
            print(f"  {v} degree={v.degree} neighbors={neighbor_ids}")

        print(f"\nFaces ({len(self.faces)}):")
        for f in self.faces:
            verts = f.get_boundary_vertices()
            vert_ids = [self._vertex_map[v].id if v in self._vertex_map else id(v) for v in verts]
            print(f"  {f} vertices={vert_ids} area={f.area():.2f}")

    def clone(self) -> 'PlanarGraph':
        """Create a deep copy of this planar graph"""
        new_graph = PlanarGraph()

        # Extract all vertices and edges
        vertex_coords = [(v.point[prim.X], v.point[prim.Y]) for v in self.vertices]

        edge_pairs = []
        seen_pairs = set()
        for edge in self.edges:
            v1_idx = self.vertices.index(edge.origin)
            v2_idx = self.vertices.index(edge.destination)

            pair = tuple(sorted([v1_idx, v2_idx]))
            if pair not in seen_pairs:
                edge_pairs.append((v1_idx, v2_idx))
                seen_pairs.add(pair)

        # Build new DCEL
        new_graph.dcel.build_dcel(vertex_coords, edge_pairs)

        # Create vertex wrappers, preserving original vertex IDs
        for idx, dcel_vertex in enumerate(new_graph.dcel.vertices):
            # Use the original vertex ID to maintain consistency
            original_vertex = self.vertices[idx]
            graph_vertex = GraphVertex(dcel_vertex, original_vertex.id, idx, graph=new_graph)
            # Update _next_vertex_id to be higher than any existing ID
            if original_vertex.id >= new_graph._next_vertex_id:
                new_graph._next_vertex_id = original_vertex.id + 1
            new_graph.vertices.append(graph_vertex)
            new_graph._vertex_map[dcel_vertex] = graph_vertex

        # Create edge wrappers
        seen_edges = set()
        for hedge in new_graph.dcel.hedges:
            hedge_id = id(hedge)
            twin_id = id(hedge.twin)
            if twin_id not in seen_edges:
                # Preserve edge type from original
                orig_idx = new_graph._next_edge_id
                edge_type = self.edges[orig_idx].edge_type if orig_idx < len(self.edges) else EdgeType.INTERNAL

                edge = Edge(hedge, new_graph._next_edge_id, edge_type, graph=new_graph)
                new_graph._next_edge_id += 1
                new_graph.edges.append(edge)
                new_graph._edge_map[hedge] = edge
                new_graph._edge_map[hedge.twin] = edge
                seen_edges.add(hedge_id)
                seen_edges.add(twin_id)

        # Rebuild faces
        new_graph.rebuild_faces()

        return new_graph

    def __repr__(self):
        return f"PlanarGraph(V={len(self.vertices)}, E={len(self.edges)}, F={len(self.faces)})"
