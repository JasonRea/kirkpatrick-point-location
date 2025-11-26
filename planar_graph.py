from typing import List, Optional, Set, Dict, Tuple
from dataclasses import dataclass, field
from enum import Enum
import primitives as prim

class EdgeType(Enum):
    """Classification of edges in the planar graph"""
    BOUNDARY = "boundary"      # Part of original polygon boundary
    DIAGONAL = "diagonal"      # Triangulation diagonal
    INTERNAL = "internal"      # Internal edge in subdivision
    EXTERNAL = "external"      # External edge

class FaceType(Enum):
    """Classification of faces"""
    INTERIOR = "interior"
    EXTERIOR = "exterior"
    UNBOUNDED = "unbounded"

@dataclass
class HalfEdge:
    """
    Half-edge representation - fundamental building block.
    Each geometric edge is represented by two half-edges going in opposite directions.
    """
    id: int
    origin: 'GraphVertex'           # Starting vertex
    twin: Optional['HalfEdge']      # Opposite half-edge
    next: Optional['HalfEdge']      # Next edge in face boundary (CCW)
    prev: Optional['HalfEdge']      # Previous edge in face boundary
    face: Optional['Face']          # Face to the left of this edge
    edge_type: EdgeType = EdgeType.INTERNAL
    
    def __repr__(self):
        origin_id = self.origin.id if self.origin else None
        dest_id = self.twin.origin.id if self.twin and self.twin.origin else None
        return f"HE{self.id}({origin_id}→{dest_id})"
    
    def destination(self) -> 'GraphVertex':
        """Get the destination vertex (origin of twin)"""
        if self.twin is None:
            raise ValueError("Half-edge has no twin")
        return self.twin.origin
    
    def is_boundary(self) -> bool:
        """Check if this edge is on the boundary"""
        return self.edge_type == EdgeType.BOUNDARY


@dataclass
class GraphVertex:
    """
    Vertex in the planar graph.
    Stores geometric position and connectivity information.
    """
    id: int
    point: prim.Point
    incident_edge: Optional[HalfEdge] = None  
    degree: int = 0
    _incident_edges: List[HalfEdge] = field(default_factory=list)  # All edges from this vertex
    
    def __repr__(self):
        return f"V{self.id}({self.point.coords[0]}, {self.point.coords[1]})"
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        if not isinstance(other, GraphVertex):
            return False
        return self.id == other.id
    
    def get_incident_edges(self) -> List[HalfEdge]:
        """Get all half-edges originating from this vertex"""
        # Return the comprehensive list of incident edges
        return list(self._incident_edges)
    
    def get_neighbors(self) -> List['GraphVertex']:
        """Get all neighboring vertices (bidirectional)"""
        neighbors = []
        for edge in self.get_incident_edges():
            neighbors.append(edge.destination())
        return neighbors
    
    def _add_incident_edge(self, edge: HalfEdge):
        """Internal method to register an incident edge"""
        if edge not in self._incident_edges:
            self._incident_edges.append(edge)
            if self.incident_edge is None:
                self.incident_edge = edge


@dataclass
class Face:
    id: int
    outer_component: Optional[HalfEdge] = None  # One edge on outer boundary
    inner_components: List[HalfEdge] = field(default_factory=list)  # Holes
    face_type: FaceType = FaceType.INTERIOR
    
    def __repr__(self):
        return f"Face{self.id}({self.face_type.value})"
    
    def get_boundary_vertices(self) -> List[GraphVertex]:
        """Get vertices on the outer boundary in order"""
        if self.outer_component is None:
            return []
        
        vertices = []
        current = self.outer_component
        
        while True:
            vertices.append(current.origin)
            current = current.next
            if current == self.outer_component:
                break
        
        return vertices
    
    def get_boundary_edges(self) -> List[HalfEdge]:
        """Get all edges bounding this face"""
        if self.outer_component is None:
            return []
        
        edges = []
        current = self.outer_component
        
        while True:
            edges.append(current)
            current = current.next
            if current == self.outer_component:
                break
        
        return edges
    
    def area(self) -> float:
        """Compute the signed area of this face"""
        vertices = self.get_boundary_vertices()
        if len(vertices) < 3:
            return 0.0
        
        area = 0.0
        n = len(vertices)
        for i in range(n):
            j = (i + 1) % n
            area += vertices[i].point[prim.X] * vertices[j].point[prim.Y]
            area -= vertices[j].point[prim.X] * vertices[i].point[prim.Y]
        
        return area / 2.0


class PlanarGraph:
    # Doubly-Connected Edge List (DCEL) representation of a planar graph.
    
    def __init__(self):
        self.vertices: List[GraphVertex] = []
        self.edges: List[HalfEdge] = []
        self.faces: List[Face] = []
        
        self._vertex_map: Dict[Tuple[float, float], GraphVertex] = {}
        self._next_vertex_id = 0
        self._next_edge_id = 0
        self._next_face_id = 0
        
        # Create unbounded face
        self.unbounded_face = Face(
            id=self._next_face_id,
            face_type=FaceType.UNBOUNDED
        )
        self.faces.append(self.unbounded_face)
        self._next_face_id += 1
    
    def clone(self) -> 'PlanarGraph':
        """
        Create a deep clone of this planar graph.
        All vertices, edges, faces, and their relationships are duplicated.
        
        Returns:
            A new PlanarGraph instance that is independent from the original.
        """
        # Create a new graph
        cloned_graph = PlanarGraph()
        cloned_graph._next_vertex_id = self._next_vertex_id
        cloned_graph._next_edge_id = self._next_edge_id
        cloned_graph._next_face_id = self._next_face_id
        
        # Step 1: Clone all vertices
        vertex_map: Dict[int, GraphVertex] = {}  # Maps old vertex id to new vertex
        for old_vertex in self.vertices:
            new_vertex = GraphVertex(
                id=old_vertex.id,
                point=prim.Point(old_vertex.point[prim.X], old_vertex.point[prim.Y]),
                incident_edge=None,
                degree=old_vertex.degree,
                _incident_edges=[]  # Will be populated later
            )
            cloned_graph.vertices.append(new_vertex)
            vertex_map[old_vertex.id] = new_vertex
            cloned_graph._vertex_map[(old_vertex.point[prim.X], old_vertex.point[prim.Y])] = new_vertex
        
        # Step 2: Clone all half-edges
        edge_map: Dict[int, HalfEdge] = {}  # Maps old edge id to new edge
        for old_edge in self.edges:
            new_edge = HalfEdge(
                id=old_edge.id,
                origin=vertex_map[old_edge.origin.id],
                twin=None,  # Will be linked later
                next=None,  # Will be linked later
                prev=None,  # Will be linked later
                face=None,  # Will be linked later
                edge_type=old_edge.edge_type
            )
            cloned_graph.edges.append(new_edge)
            edge_map[old_edge.id] = new_edge
        
        # Step 3: Link twin relationships
        for old_edge in self.edges:
            new_edge = edge_map[old_edge.id]
            if old_edge.twin is not None:
                new_edge.twin = edge_map[old_edge.twin.id]
        
        # Step 4: Link next/prev pointers
        for old_edge in self.edges:
            new_edge = edge_map[old_edge.id]
            if old_edge.next is not None:
                new_edge.next = edge_map[old_edge.next.id]
            if old_edge.prev is not None:
                new_edge.prev = edge_map[old_edge.prev.id]
        
        # Step 5: Populate incident edges for vertices
        for old_vertex in self.vertices:
            new_vertex = vertex_map[old_vertex.id]
            for old_incident_edge in old_vertex._incident_edges:
                new_vertex._incident_edges.append(edge_map[old_incident_edge.id])
            if old_vertex.incident_edge is not None:
                new_vertex.incident_edge = edge_map[old_vertex.incident_edge.id]
        
        # Step 6: Clone faces and link edges
        face_map: Dict[int, Face] = {}  # Maps old face id to new face
        
        # Clone unbounded face first
        new_unbounded_face = Face(
            id=self.unbounded_face.id,
            outer_component=None,  # Will be linked later
            inner_components=[],
            face_type=self.unbounded_face.face_type
        )
        cloned_graph.faces[0] = new_unbounded_face  # Replace the default unbounded face
        face_map[self.unbounded_face.id] = new_unbounded_face
        
        # Clone other faces
        for old_face in self.faces:
            if old_face.id == self.unbounded_face.id:
                continue  # Already cloned
            
            new_face = Face(
                id=old_face.id,
                outer_component=None,  # Will be linked later
                inner_components=[],
                face_type=old_face.face_type
            )
            cloned_graph.faces.append(new_face)
            face_map[old_face.id] = new_face
        
        # Step 7: Link outer components and inner components
        for old_face in self.faces:
            new_face = face_map[old_face.id]
            if old_face.outer_component is not None:
                new_face.outer_component = edge_map[old_face.outer_component.id]
            for old_inner_edge in old_face.inner_components:
                new_face.inner_components.append(edge_map[old_inner_edge.id])
        
        # Step 8: Link faces to edges
        for old_edge in self.edges:
            new_edge = edge_map[old_edge.id]
            if old_edge.face is not None:
                new_edge.face = face_map[old_edge.face.id]
        
        return cloned_graph
    
    def add_vertex(self, x: float, y: float) -> GraphVertex:
        # Add a vertex at the given coordinates
        # Check if vertex already exists
        key = (x, y)
        if key in self._vertex_map:
            return self._vertex_map[key]
        
        vertex = GraphVertex(
            id=self._next_vertex_id,
            point=prim.Point(x, y)
        )
        self._next_vertex_id += 1
        
        self.vertices.append(vertex)
        self._vertex_map[key] = vertex
        
        return vertex
    
    def add_edge(self, v1: GraphVertex, v2: GraphVertex, 
                 edge_type: EdgeType = EdgeType.INTERNAL) -> Tuple[HalfEdge, HalfEdge]:
        """
        Add an edge between two vertices.
        Returns a tuple of (half_edge_1_to_2, half_edge_2_to_1)
        """
        # Create two half-edges
        he1 = HalfEdge(
            id=self._next_edge_id,
            origin=v1,
            twin=None,
            next=None,
            prev=None,
            face=None,
            edge_type=edge_type
        )
        self._next_edge_id += 1
        
        he2 = HalfEdge(
            id=self._next_edge_id,
            origin=v2,
            twin=None,
            next=None,
            prev=None,
            face=None,
            edge_type=edge_type
        )
        self._next_edge_id += 1
        
        # Link twins
        he1.twin = he2
        he2.twin = he1
        
        # Register edges with BOTH vertices (bidirectional)
        v1._add_incident_edge(he1)
        v2._add_incident_edge(he2)
        
        # Update vertex degrees
        v1.degree += 1
        v2.degree += 1
        
        self.edges.extend([he1, he2])
        
        return he1, he2
    
    def create_face(self, boundary_edge: HalfEdge, 
                   face_type: FaceType = FaceType.INTERIOR) -> Face:
        """Create a face with the given boundary edge"""
        face = Face(
            id=self._next_face_id,
            outer_component=boundary_edge,
            face_type=face_type
        )
        self._next_face_id += 1
        
        # Assign face to all boundary edges
        current = boundary_edge
        while True:
            current.face = face
            current = current.next
            if current == boundary_edge:
                break
        
        self.faces.append(face)
        return face
    
    def from_polygon(self, points: List[Tuple[float, float]]) -> 'PlanarGraph':
        """
        Build planar graph from a simple polygon.
        Creates vertices and edges in order, forming one face.
        """
        if len(points) < 3:
            raise ValueError("Polygon must have at least 3 vertices")
        
        # Create vertices
        vertices = [self.add_vertex(x, y) for x, y in points]
        n = len(vertices)
        
        # Create edges and link them
        half_edges = []
        for i in range(n):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % n]
            he1, he2 = self.add_edge(v1, v2, EdgeType.BOUNDARY)
            half_edges.append(he1)
            
            # Link to unbounded face (twin edges)
            he2.face = self.unbounded_face
        
        # Link next/prev pointers for inner face
        for i in range(n):
            half_edges[i].next = half_edges[(i + 1) % n]
            half_edges[i].prev = half_edges[(i - 1) % n]
        
        # Link next/prev for outer (unbounded) face
        for i in range(n):
            half_edges[i].twin.next = half_edges[(i - 1) % n].twin
            half_edges[i].twin.prev = half_edges[(i + 1) % n].twin
        
        # Create interior face
        interior_face = self.create_face(half_edges[0], FaceType.INTERIOR)
        
        # Update unbounded face component
        if self.unbounded_face.outer_component is None:
            self.unbounded_face.outer_component = half_edges[0].twin
        
        return self
    
    def add_diagonal(self, v1: GraphVertex, v2: GraphVertex) -> Tuple[HalfEdge, HalfEdge]:
        """
        Add a diagonal between two vertices, splitting a face.
        Used in triangulation.
        """
        he1, he2 = self.add_edge(v1, v2, EdgeType.DIAGONAL)
        
        # TODO: Update face information
        # This requires finding which face is being split
        # and creating two new faces
        
        return he1, he2
    
    def triangulate(self) -> List[Tuple[int, int, int]]:
        """
        Triangulate the planar graph using ear clipping.
        This modifies the graph by adding diagonals and splitting faces accordingly.
        Maintains proper Euler characteristic: V - E + F = 2
        
        Returns:
            List of triangles, where each triangle is a tuple of three vertex IDs sorted in increasing order.
            Example: [(0, 1, 2), (0, 2, 3), (1, 2, 4)]
        """
        # Find the interior face to triangulate
        interior_faces = [f for f in self.faces 
                         if f.face_type == FaceType.INTERIOR]
        
        if not interior_faces:
            return []
        
        face = interior_faces[0]
        vertices_list = face.get_boundary_vertices()
        n = len(vertices_list)
        
        # Create a mapping from vertex position to graph vertex
        vertex_map = {i: gv for i, gv in enumerate(vertices_list)}
        
        # Convert to old vertex format for triangulation
        old_vertices = []
        for i, gv in enumerate(vertices_list):
            v = prim.Vertex(i, gv.point)
            old_vertices.append(v)
        
        # Link them circularly
        prim.link_vertices_circular(old_vertices)
        
        # Run ear clipping (returns list of diagonals)
        from polygon import Triangulate
        diagonals = Triangulate()
        
        # Track the original half-edges of the face boundary
        original_edges = face.get_boundary_edges()
        original_edges_set = {id(e) for e in original_edges}
        
        # Add diagonals to graph structure
        added_diagonals = []
        if diagonals:
            print(f"\nAdding {len(diagonals)} diagonals to graph structure:")
            for v1_idx, v2_idx in diagonals:
                if v1_idx < len(vertices_list) and v2_idx < len(vertices_list):
                    gv1 = vertex_map[v1_idx]
                    gv2 = vertex_map[v2_idx]
                    
                    # Add the diagonal edge to the graph
                    he1, he2 = self.add_diagonal(gv1, gv2)
                    added_diagonals.append((he1, he2, gv1, gv2))
                    print(f"  Added diagonal: {gv1.id} -- {gv2.id}")
            
            # Now perform face splitting
            if added_diagonals:
                print(f"\nPerforming face splitting...")
                triangles = self._split_faces_for_triangulation(face, added_diagonals, vertex_map)
                return triangles
        else:
            print("No diagonals added (already triangulated)")
            return []
    
    def _split_faces_for_triangulation(self, original_face: Face, 
                                       diagonals: List[Tuple[HalfEdge, HalfEdge, GraphVertex, GraphVertex]],
                                       vertex_map: Dict[int, GraphVertex]) -> List[Tuple[int, int, int]]:
        """
        Split faces along the added diagonals.
        Each diagonal splits one face into two faces.
        For a triangulation of an n-gon: V - E + F = 2 should hold.
        After triangulation: n vertices, 2n-3 edges, n-1 interior faces + 1 unbounded = n faces total.
        
        Returns:
            List of triangles, where each triangle is a tuple of three vertex IDs sorted in increasing order.
        """
        # Remove the original face from the face list
        self.faces.remove(original_face)
        
        boundary_vertices = original_face.get_boundary_vertices()
        n = len(boundary_vertices)
        
        # Expected statistics after triangulation:
        # - Vertices: n (unchanged)
        # - Edges: n (boundary) + (n-3) diagonals = 2n - 3
        # - Faces: (n-2) interior triangles + 1 unbounded = n - 1 total
        
        # For a proper triangulation, we should have n-2 triangular faces
        num_triangles_expected = n - 2
        triangles = []
        
        # Create n-2 triangular faces
        # We'll use a simplified fan triangulation from the first vertex
        if n >= 3:
            base_vertex = boundary_vertices[0]
            
            # Get all edges in the triangulated face (boundary + diagonals)
            boundary_edges = original_face.get_boundary_edges()
            
            # Create a mapping of consecutive vertices to their boundary edge
            edge_map = {}
            for edge in boundary_edges:
                v_from = edge.origin
                v_to = edge.destination()
                edge_map[(v_from, v_to)] = edge
            
            # Also include diagonal edges
            for he1, he2, gv1, gv2 in diagonals:
                edge_map[(gv1, gv2)] = he1
                edge_map[(gv2, gv1)] = he2
            
            # Create triangular faces
            triangles_created = 0
            for i in range(1, n - 1):
                v1 = boundary_vertices[i]
                v2 = boundary_vertices[i + 1]
                
                # Triangle: base_vertex -> v1 -> v2 -> base_vertex
                # Find or create the edges for this triangle
                edge_list = []
                
                # Edge 1: base_vertex -> v1
                if (base_vertex, v1) in edge_map:
                    edge_list.append(edge_map[(base_vertex, v1)])
                elif (v1, base_vertex) in edge_map:
                    edge_list.append(edge_map[(v1, base_vertex)].twin)
                
                # Edge 2: v1 -> v2 (boundary edge, should exist)
                if (v1, v2) in edge_map:
                    edge_list.append(edge_map[(v1, v2)])
                elif (v2, v1) in edge_map:
                    edge_list.append(edge_map[(v2, v1)].twin)
                
                # Edge 3: v2 -> base_vertex
                if (v2, base_vertex) in edge_map:
                    edge_list.append(edge_map[(v2, base_vertex)])
                elif (base_vertex, v2) in edge_map:
                    edge_list.append(edge_map[(base_vertex, v2)].twin)
                
                # Create triangle face if we have a starting edge
                if edge_list:
                    tri_face = Face(
                        id=self._next_face_id,
                        outer_component=edge_list[0],
                        face_type=FaceType.INTERIOR
                    )
                    self._next_face_id += 1
                    
                    # Assign face to all triangle edges
                    current = edge_list[0]
                    count = 0
                    while count < 3:
                        current.face = tri_face
                        # Try to move to next edge
                        if current.next and current.next != edge_list[0]:
                            current = current.next
                        else:
                            break
                        count += 1
                    
                    self.faces.append(tri_face)
                    triangles_created += 1
                    
                    # Create triangle tuple with sorted vertex IDs
                    tri_indices = tuple(sorted([base_vertex.id, v1.id, v2.id]))
                    triangles.append(tri_indices)
                    
                    print(f"  Created triangle {triangles_created}: {base_vertex.id} -- {v1.id} -- {v2.id}")
            
            print(f"  Total triangles created: {triangles_created} (expected: {num_triangles_expected})")
        
        return triangles
    
    def get_vertex_by_coords(self, x: float, y: float) -> Optional[GraphVertex]:
        """Find vertex at given coordinates"""
        return self._vertex_map.get((x, y))
    
    def get_incident_edges_for_vertex(self, vertex: GraphVertex) -> List[HalfEdge]:
        """
        Get all half-edges originating from a vertex.
        This directly searches the edge list (works even for unlinked edges).
        """
        incident_edges = []
        for edge in self.edges:
            if edge.origin == vertex:
                incident_edges.append(edge)
        return incident_edges
    
    def find_face_containing_point(self, x: float, y: float) -> Optional[Face]:
        """
        Point location: find which face contains the given point.
        Uses the point-in-polygon test from Chapter 7.
        """
        query_point = prim.Point(x, y)
        
        for face in self.faces:
            if face.face_type == FaceType.UNBOUNDED:
                continue
            
            vertices = face.get_boundary_vertices()
            if len(vertices) < 3:
                continue
            
            # Use ray casting algorithm
            if self._point_in_face(query_point, face):
                return face
        
        return self.unbounded_face
    
    def _point_in_face(self, point: prim.Point, face: Face) -> bool:
        """Check if point is inside the given face using ray casting"""
        vertices = face.get_boundary_vertices()
        n = len(vertices)
        
        if n < 3:
            return False
        
        rcross = 0  # Right crossings
        lcross = 0  # Left crossings
        
        for i in range(n):
            i1 = (i - 1) % n
            v_i = vertices[i].point
            v_i1 = vertices[i1].point
            
            # Check if edge straddles horizontal ray through point
            if ((v_i[prim.Y] > point[prim.Y]) != (v_i1[prim.Y] > point[prim.Y])):
                # Compute x intersection
                x = (v_i[prim.X] - v_i1[prim.X]) * (point[prim.Y] - v_i1[prim.Y]) / \
                    (v_i[prim.Y] - v_i1[prim.Y]) + v_i1[prim.X]
                
                if x > point[prim.X]:
                    rcross += 1
                if x < point[prim.X]:
                    lcross += 1
        
        # Inside if odd number of crossings
        return (rcross % 2) == 1
    
    def euler_characteristic(self) -> int:
        """
        Compute Euler characteristic: V - E + F
        For a connected planar graph, this should equal 2.
        """
        V = len(self.vertices)
        E = len(self.edges) // 2  # Each edge has two half-edges
        F = len(self.faces)
        
        return V - E + F
    
    def validate(self) -> bool:
        """
        Validate the planar graph structure.
        Checks:
        - Twin relationships
        - Next/prev cycles
        - Face assignments
        """
        # Check twin relationships
        for he in self.edges:
            if he.twin is None:
                print(f"Error: {he} has no twin")
                return False
            if he.twin.twin != he:
                print(f"Error: Twin relationship broken for {he}")
                return False
        
        # Check next/prev cycles
        for he in self.edges:
            if he.next is not None:
                if he.next.prev != he:
                    print(f"Error: Next/prev mismatch for {he}")
                    return False
        
        return True
    
    def __repr__(self):
        return (f"PlanarGraph(V={len(self.vertices)}, "
                f"E={len(self.edges)//2}, F={len(self.faces)})")
    
    def print_structure(self):
        """Print detailed structure information"""
        print(self)
        print(f"Euler characteristic: {self.euler_characteristic()}")
        print(f"\nVertices ({len(self.vertices)}):")
        for v in self.vertices:
            neighbors = [n.id for n in v.get_neighbors()]
            print(f"  {v} degree={v.degree} neighbors={neighbors}")
        
        print(f"\nFaces ({len(self.faces)}):")
        for f in self.faces:
            vertices = [v.id for v in f.get_boundary_vertices()]
            area = f.area()
            print(f"  {f} vertices={vertices} area={area:.2f}")