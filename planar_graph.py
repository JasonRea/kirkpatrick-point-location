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
    
    def __repr__(self):
        return f"V{self.id}({self.point.coords[0]}, {self.point.coords[1]})"
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        if not isinstance(other, GraphVertex):
            return False
        return self.id == other.id
    
    def get_incident_edges(self) -> List[HalfEdge]:
        """Get all half-edges originating from this vertex (CCW order)"""
        if self.incident_edge is None:
            return []
        
        edges = []
        current = self.incident_edge
        
        while True:
            edges.append(current)
            # Move to next outgoing edge around vertex
            if current.twin and current.twin.next:
                current = current.twin.next
            else:
                break
            
            if current == self.incident_edge:
                break
        
        return edges
    
    def get_neighbors(self) -> List['GraphVertex']:
        edges = self.get_incident_edges()
        return [edge.destination() for edge in edges]


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
        
        # Update vertex incident edges if not set
        if v1.incident_edge is None:
            v1.incident_edge = he1
        if v2.incident_edge is None:
            v2.incident_edge = he2
        
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
    
    def triangulate(self):
        """
        Triangulate the planar graph using ear clipping.
        This modifies the graph by adding diagonals.
        """
        # Find the interior face to triangulate
        interior_faces = [f for f in self.faces 
                         if f.face_type == FaceType.INTERIOR]
        
        if not interior_faces:
            return
        
        face = interior_faces[0]
        vertices_list = face.get_boundary_vertices()
        
        # Convert to old vertex format for triangulation
        old_vertices = []
        for i, gv in enumerate(vertices_list):
            v = prim.Vertex(i, gv.point)
            old_vertices.append(v)
        
        # Link them circularly
        prim.link_vertices_circular(old_vertices)
        
        # Run ear clipping (this will modify prim.vertices)
        from polygon import Triangulate
        Triangulate()
        
        # TODO: Update graph structure with new diagonals
        # This requires tracking which diagonals were added
    
    def get_vertex_by_coords(self, x: float, y: float) -> Optional[GraphVertex]:
        """Find vertex at given coordinates"""
        return self._vertex_map.get((x, y))
    
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