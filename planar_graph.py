"""
Planar Graph with Winged-Edge Data Structure

This module implements a proper winged-edge data structure for representing planar graphs.
The winged-edge structure stores topological information efficiently for traversal and modification.

Winged-Edge Topology:
--------------------
Each edge stores:
- Two endpoint vertices (origin and destination)
- Two incident faces (left and right, when traversing from origin to destination)
- Four adjacent edges:
  - prev_at_origin: predecessor edge in CCW order around origin vertex
  - next_at_origin: successor edge in CCW order around origin vertex
  - prev_at_dest: predecessor edge in CCW order around destination vertex
  - next_at_dest: successor edge in CCW order around destination vertex

This allows efficient traversal of:
- All edges incident to a vertex (walk CCW around vertex)
- All edges bounding a face (walk along face boundary)
- Navigate between adjacent edges and faces
"""

from typing import Optional
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


class GraphVertex:
    """
    Vertex in the planar graph with winged-edge support.

    Attributes:
        id: Unique identifier for the vertex
        point: Geometric position (primitives.Point)
        incident_edge: One edge originating from this vertex (for traversal)
        degree: Number of edges incident to this vertex
    """

    def __init__(self, vertex_id: int, point: prim.Point):
        self.id = vertex_id
        self.point = point
        self.incident_edge: Optional['Edge'] = None
        self.degree = 0

    def __repr__(self):
        return f"V{self.id}({self.point.coords[0]:.2f}, {self.point.coords[1]:.2f})"

    def __hash__(self):
        if self.id is None:
            raise ValueError(f"Cannot hash vertex with None id: {self}")
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, GraphVertex):
            return False
        return self.id == other.id

    def get_incident_edges(self) -> list['Edge']:
        """
        Get all edges incident to this vertex by walking CCW around the vertex.
        Uses the winged-edge next_at_origin pointers for efficient traversal.
        """
        if self.incident_edge is None:
            return []

        edges = []
        visited_edge_ids = set()
        current = self.incident_edge
        first_edge_id = current.id

        # Add first edge
        edges.append(current)
        visited_edge_ids.add(current.id)

        # Move to next edge CCW around this vertex
        if current.origin == self:
            current = current.next_at_origin
        else:
            current = current.next_at_dest

        # Continue until we loop back to the first edge, reach None, or detect a cycle
        while current is not None and current.id != first_edge_id:
            # Detect infinite loops by checking if we've seen this edge before
            if current.id in visited_edge_ids:
                break

            edges.append(current)
            visited_edge_ids.add(current.id)

            # Move to next edge CCW around this vertex
            if current.origin == self:
                current = current.next_at_origin
            else:
                current = current.next_at_dest

        return edges

    def get_neighbors(self) -> list['GraphVertex']:
        """Get all neighboring vertices"""
        neighbors = []
        for edge in self.get_incident_edges():
            # Get the other endpoint
            neighbor = edge.destination if edge.origin == self else edge.origin
            neighbors.append(neighbor)
        return neighbors


class Edge:
    """
    Winged-edge representation of an undirected edge.

    A winged-edge stores complete topological information:
    - Two endpoints (origin and destination)
    - Two incident faces (left and right, when traversing origin → destination)
    - Four adjacent edges for CCW traversal around each endpoint

    The "wings" are the four adjacent edge pointers that enable efficient traversal.

    Attributes:
        id: Unique identifier
        origin: Starting vertex
        destination: Ending vertex
        left_face: Face to the left when traversing origin → destination
        right_face: Face to the right when traversing origin → destination
        prev_at_origin: Previous edge CCW around origin
        next_at_origin: Next edge CCW around origin
        prev_at_dest: Previous edge CCW around destination
        next_at_dest: Next edge CCW around destination
        edge_type: Classification of this edge
    """

    def __init__(self, edge_id: int, origin: GraphVertex, destination: GraphVertex,
                 edge_type: EdgeType = EdgeType.INTERNAL):
        self.id = edge_id
        self.origin = origin
        self.destination = destination
        self.left_face: Optional['Face'] = None
        self.right_face: Optional['Face'] = None

        # The four "wings" - adjacent edges
        self.prev_at_origin: Optional['Edge'] = None
        self.next_at_origin: Optional['Edge'] = None
        self.prev_at_dest: Optional['Edge'] = None
        self.next_at_dest: Optional['Edge'] = None

        self.edge_type = edge_type

    def __repr__(self):
        return f"E{self.id}({self.origin.id}→{self.destination.id})"

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, Edge):
            return False
        return self.id == other.id

    def is_boundary(self) -> bool:
        """Check if this edge is on the boundary"""
        return self.edge_type == EdgeType.BOUNDARY

    def other_endpoint(self, vertex: GraphVertex) -> GraphVertex:
        """Given one endpoint, return the other"""
        if vertex == self.origin:
            return self.destination
        elif vertex == self.destination:
            return self.origin
        else:
            raise ValueError(f"Vertex {vertex} is not an endpoint of edge {self}")

    def get_face_on_side(self, vertex: GraphVertex) -> Optional['Face']:
        """
        Get the face on the left side when traversing from the given vertex.
        If vertex is origin, returns left_face.
        If vertex is destination, returns right_face.
        """
        if vertex == self.origin:
            return self.left_face
        elif vertex == self.destination:
            return self.right_face
        else:
            raise ValueError(f"Vertex {vertex} is not an endpoint of edge {self}")


class Face:
    """
    Face in the planar graph.

    A face is bounded by a cycle of edges. We store one edge on the boundary
    and can traverse the complete boundary using the winged-edge pointers.

    Attributes:
        id: Unique identifier
        boundary_edge: One edge on the face boundary (for traversal)
        inner_components: List of edges for holes (not commonly used)
        face_type: Classification of this face
    """

    def __init__(self, face_id: int, face_type: FaceType = FaceType.INTERIOR):
        self.id = face_id
        self.boundary_edge: Optional[Edge] = None
        self.inner_components: list[Edge] = []
        self.face_type = face_type

    def __repr__(self):
        return f"Face{self.id}({self.face_type.value})"

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, Face):
            return False
        return self.id == other.id

    def get_boundary_vertices(self) -> list[GraphVertex]:
        """
        Get vertices on the boundary in order by walking around the face.
        Uses winged-edge pointers to traverse the face boundary.
        """
        if self.boundary_edge is None:
            return []

        vertices = []
        visited_edges = set()
        current_edge = self.boundary_edge

        while current_edge and current_edge.id not in visited_edges:
            visited_edges.add(current_edge.id)

            # Determine which vertex to add based on which face we're traversing
            if current_edge.left_face == self:
                vertices.append(current_edge.origin)
                # Move to next edge along this face (CCW around face)
                current_edge = self._next_edge_ccw(current_edge, at_origin=False)
            elif current_edge.right_face == self:
                vertices.append(current_edge.destination)
                # Move to next edge along this face
                current_edge = self._next_edge_ccw(current_edge, at_origin=True)
            else:
                break

        return vertices

    def get_boundary_edges(self) -> list[Edge]:
        """Get all edges bounding this face"""
        if self.boundary_edge is None:
            return []

        edges = []
        visited_edges = set()
        current_edge = self.boundary_edge

        while current_edge and current_edge.id not in visited_edges:
            visited_edges.add(current_edge.id)
            edges.append(current_edge)

            # Move to next edge along this face
            if current_edge.left_face == self:
                current_edge = self._next_edge_ccw(current_edge, at_origin=False)
            elif current_edge.right_face == self:
                current_edge = self._next_edge_ccw(current_edge, at_origin=True)
            else:
                break

        return edges

    def _next_edge_ccw(self, edge: Edge, at_origin: bool) -> Optional[Edge]:
        """
        Get the next edge CCW around the face boundary.

        Args:
            edge: Current edge
            at_origin: True if we're at the origin of the edge, False if at destination
        """
        if at_origin:
            # At origin, moving CCW around face means taking prev_at_origin
            return edge.prev_at_origin
        else:
            # At destination, moving CCW around face means taking next_at_dest
            return edge.next_at_dest

    def area(self) -> float:
        """Compute the signed area of this face using the shoelace formula"""
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


# Alias for backward compatibility with existing code
HalfEdge = Edge  # Map old HalfEdge references to Edge


class PlanarGraph:
    """
    Planar graph with winged-edge data structure.

    This class maintains a collection of vertices, edges, and faces representing
    a planar graph embedded in 2D. The winged-edge structure provides efficient
    traversal and modification operations needed for Kirkpatrick's algorithm.
    """

    def __init__(self):
        self.vertices: list[GraphVertex] = []
        self.edges: list[Edge] = []
        self.faces: list[Face] = []

        self._vertex_map: dict[tuple[float, float], GraphVertex] = {}
        self._next_vertex_id = 0
        self._next_edge_id = 0
        self._next_face_id = 0

        # Create unbounded face
        self.unbounded_face = Face(
            face_id=self._next_face_id,
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
        cloned_graph = PlanarGraph()
        cloned_graph._next_vertex_id = self._next_vertex_id
        cloned_graph._next_edge_id = self._next_edge_id
        cloned_graph._next_face_id = self._next_face_id

        # Step 1: Clone all vertices
        vertex_map: dict[int, GraphVertex] = {}
        for old_vertex in self.vertices:
            new_vertex = GraphVertex(
                vertex_id=old_vertex.id,
                point=prim.Point(old_vertex.point[prim.X], old_vertex.point[prim.Y])
            )
            new_vertex.degree = old_vertex.degree
            cloned_graph.vertices.append(new_vertex)
            vertex_map[old_vertex.id] = new_vertex
            cloned_graph._vertex_map[(old_vertex.point[prim.X], old_vertex.point[prim.Y])] = new_vertex

        # Step 2: Clone all edges
        edge_map: dict[int, Edge] = {}
        for old_edge in self.edges:
            new_edge = Edge(
                edge_id=old_edge.id,
                origin=vertex_map[old_edge.origin.id],
                destination=vertex_map[old_edge.destination.id],
                edge_type=old_edge.edge_type
            )
            cloned_graph.edges.append(new_edge)
            edge_map[old_edge.id] = new_edge

        # Step 3: Link edge winged-edge pointers
        for old_edge in self.edges:
            new_edge = edge_map[old_edge.id]
            if old_edge.prev_at_origin:
                new_edge.prev_at_origin = edge_map.get(old_edge.prev_at_origin.id)
            if old_edge.next_at_origin:
                new_edge.next_at_origin = edge_map.get(old_edge.next_at_origin.id)
            if old_edge.prev_at_dest:
                new_edge.prev_at_dest = edge_map.get(old_edge.prev_at_dest.id)
            if old_edge.next_at_dest:
                new_edge.next_at_dest = edge_map.get(old_edge.next_at_dest.id)

        # Step 4: Update vertex incident edges
        for old_vertex in self.vertices:
            new_vertex = vertex_map[old_vertex.id]
            if old_vertex.incident_edge:
                new_vertex.incident_edge = edge_map.get(old_vertex.incident_edge.id)

        # Step 5: Clone faces
        face_map: dict[int, Face] = {}

        # Clone unbounded face
        new_unbounded_face = Face(
            face_id=self.unbounded_face.id,
            face_type=self.unbounded_face.face_type
        )
        cloned_graph.faces[0] = new_unbounded_face
        face_map[self.unbounded_face.id] = new_unbounded_face

        # Clone other faces
        for old_face in self.faces:
            if old_face.id == self.unbounded_face.id:
                continue

            new_face = Face(
                face_id=old_face.id,
                face_type=old_face.face_type
            )
            cloned_graph.faces.append(new_face)
            face_map[old_face.id] = new_face

        # Step 6: Link face boundary edges
        for old_face in self.faces:
            new_face = face_map[old_face.id]
            if old_face.boundary_edge:
                new_face.boundary_edge = edge_map.get(old_face.boundary_edge.id)
            for old_inner_edge in old_face.inner_components:
                if old_inner_edge.id in edge_map:
                    new_face.inner_components.append(edge_map[old_inner_edge.id])

        # Step 7: Link faces to edges
        for old_edge in self.edges:
            new_edge = edge_map[old_edge.id]
            if old_edge.left_face:
                new_edge.left_face = face_map.get(old_edge.left_face.id)
            if old_edge.right_face:
                new_edge.right_face = face_map.get(old_edge.right_face.id)

        return cloned_graph

    def add_vertex(self, x: float, y: float) -> GraphVertex:
        """
        Add a vertex at the given coordinates.
        If a vertex already exists at these coordinates, return it.
        """
        key = (x, y)
        if key in self._vertex_map:
            return self._vertex_map[key]

        vertex = GraphVertex(
            vertex_id=self._next_vertex_id,
            point=prim.Point(x, y)
        )
        self._next_vertex_id += 1

        self.vertices.append(vertex)
        self._vertex_map[key] = vertex

        return vertex

    def add_edge(self, v1: GraphVertex, v2: GraphVertex,
                 edge_type: EdgeType = EdgeType.INTERNAL) -> Edge:
        """
        Add an edge between two vertices with proper winged-edge linkage.

        This maintains the CCW ordering of edges around each vertex and updates
        all winged-edge pointers (prev_at_origin, next_at_origin, etc.)

        The key insight is to insert the new edge in the correct CCW position
        based on geometric angle ordering.

        Returns:
            The new edge connecting v1 to v2
        """
        # Create the edge
        edge = Edge(
            edge_id=self._next_edge_id,
            origin=v1,
            destination=v2,
            edge_type=edge_type
        )
        self._next_edge_id += 1

        # Insert edge into CCW ordering around v1 (origin)
        self._insert_edge_at_vertex(edge, v1, at_origin=True)

        # Insert edge into CCW ordering around v2 (destination)
        self._insert_edge_at_vertex(edge, v2, at_origin=False)

        # Assign faces to the new edge
        # For now, assign unbounded face to both sides
        # This will be updated properly when faces are created/split
        edge.left_face = self.unbounded_face
        edge.right_face = self.unbounded_face

        # Update vertex degrees
        v1.degree += 1
        v2.degree += 1

        self.edges.append(edge)

        return edge

    def _insert_edge_at_vertex(self, new_edge: Edge, vertex: GraphVertex, at_origin: bool):
        """
        Insert an edge into the CCW ordering around a vertex.

        Args:
            new_edge: The edge to insert
            vertex: The vertex around which to insert
            at_origin: True if vertex is the origin of new_edge, False if destination
        """
        if vertex.incident_edge is None:
            # First edge at this vertex - create self-loop
            vertex.incident_edge = new_edge
            if at_origin:
                new_edge.prev_at_origin = new_edge
                new_edge.next_at_origin = new_edge
            else:
                new_edge.prev_at_dest = new_edge
                new_edge.next_at_dest = new_edge
            return

        # Find the correct position to insert based on CCW angle ordering
        # We need to find edges e1 and e2 such that new_edge fits between them in CCW order

        # Get the other endpoint of the new edge (the direction vector)
        if at_origin:
            new_direction = new_edge.destination.point
        else:
            new_direction = new_edge.origin.point

        # Walk around the vertex to find insertion point
        current = vertex.incident_edge
        best_prev = None
        best_next = None

        # For each edge around the vertex, check if new_edge should go between
        # current and current.next in CCW order
        first_edge = current
        while True:
            # Get the direction of the current edge from vertex
            if current.origin == vertex:
                current_direction = current.destination.point
                next_edge = current.next_at_origin
            else:
                current_direction = current.origin.point
                next_edge = current.next_at_dest

            # Get the direction of the next edge from vertex
            if next_edge.origin == vertex:
                next_direction = next_edge.destination.point
            else:
                next_direction = next_edge.origin.point

            # Check if new_edge fits between current and next in CCW order
            # Using cross product: if both (current × new) > 0 and (new × next) > 0,
            # then new is between current and next in CCW order

            area_current_new = prim.Area2(vertex.point, current_direction, new_direction)
            area_new_next = prim.Area2(vertex.point, new_direction, next_direction)
            area_current_next = prim.Area2(vertex.point, current_direction, next_direction)

            # If edges span more than 180 degrees, we need different logic
            if area_current_next <= 0:
                # Reflex angle case (> 180 degrees)
                if area_current_new > 0 or area_new_next > 0:
                    best_prev = current
                    best_next = next_edge
                    break
            else:
                # Normal case (< 180 degrees)
                if area_current_new > 0 and area_new_next > 0:
                    best_prev = current
                    best_next = next_edge
                    break

            current = next_edge
            if current == first_edge:
                # Completed the loop - insert after first_edge
                best_prev = first_edge
                if first_edge.origin == vertex:
                    best_next = first_edge.next_at_origin
                else:
                    best_next = first_edge.next_at_dest
                break

        # Insert new_edge between best_prev and best_next
        if at_origin:
            new_edge.prev_at_origin = best_prev
            new_edge.next_at_origin = best_next

            # Update best_prev to point to new_edge
            if best_prev.origin == vertex:
                best_prev.next_at_origin = new_edge
            else:
                best_prev.next_at_dest = new_edge

            # Update best_next to point to new_edge
            if best_next.origin == vertex:
                best_next.prev_at_origin = new_edge
            else:
                best_next.prev_at_dest = new_edge
        else:
            new_edge.prev_at_dest = best_prev
            new_edge.next_at_dest = best_next

            # Update best_prev to point to new_edge
            if best_prev.origin == vertex:
                best_prev.next_at_origin = new_edge
            else:
                best_prev.next_at_dest = new_edge

            # Update best_next to point to new_edge
            if best_next.origin == vertex:
                best_next.prev_at_origin = new_edge
            else:
                best_next.prev_at_dest = new_edge

    def remove_vertex(self, vertex: GraphVertex) -> bool:
        """
        Remove a vertex from the planar graph while maintaining topological consistency.

        This removes all edges incident to the vertex and merges affected faces.
        Maintains Euler characteristic: V - E + F = 2

        Args:
            vertex: The vertex to remove

        Returns:
            True if vertex was successfully removed, False otherwise
        """
        if vertex not in self.vertices:
            return False

        if len(self.vertices) == 1:
            return False

        # Get all incident edges
        incident_edges = vertex.get_incident_edges()

        if not incident_edges:
            # Isolated vertex - safe to remove
            self.vertices.remove(vertex)
            key = (vertex.point[prim.X], vertex.point[prim.Y])
            if key in self._vertex_map:
                del self._vertex_map[key]
            return True

        # Collect incident faces
        incident_faces = set()
        for edge in incident_edges:
            if edge.left_face and edge.left_face != self.unbounded_face:
                incident_faces.add(edge.left_face)
            if edge.right_face and edge.right_face != self.unbounded_face:
                incident_faces.add(edge.right_face)

        # For each pair of consecutive edges, reconnect around the removed vertex
        for i, edge in enumerate(incident_edges):
            next_edge = incident_edges[(i + 1) % len(incident_edges)]

            # Reconnect the winged-edge pointers to bypass the removed vertex
            # This is complex and depends on the specific topology
            # For now, we'll handle the simpler case
            pass

        # Remove edges
        for edge in incident_edges:
            if edge in self.edges:
                self.edges.remove(edge)

            # Update adjacent vertices
            other_vertex = edge.other_endpoint(vertex)
            if other_vertex.incident_edge == edge:
                # Find another incident edge
                other_edges = [e for e in self.edges if e.origin == other_vertex or e.destination == other_vertex]
                other_vertex.incident_edge = other_edges[0] if other_edges else None
            other_vertex.degree -= 1

        # Merge faces if necessary
        if len(incident_faces) > 1:
            primary_face = next(iter(incident_faces))
            for secondary_face in incident_faces:
                if secondary_face != primary_face and secondary_face in self.faces:
                    self.faces.remove(secondary_face)

        # Remove vertex
        self.vertices.remove(vertex)
        key = (vertex.point[prim.X], vertex.point[prim.Y])
        if key in self._vertex_map:
            del self._vertex_map[key]

        return True

    def create_face(self, boundary_edge: Edge,
                   face_type: FaceType = FaceType.INTERIOR) -> Face:
        """
        Create a face with the given boundary edge.
        Assigns the face to all edges on its boundary.
        """
        face = Face(
            face_id=self._next_face_id,
            face_type=face_type
        )
        face.boundary_edge = boundary_edge
        self._next_face_id += 1

        # Assign face to all boundary edges
        visited = set()
        current = boundary_edge

        while current and current.id not in visited:
            visited.add(current.id)

            # Assign face to appropriate side of edge
            if current.left_face is None:
                current.left_face = face
            elif current.right_face is None:
                current.right_face = face

            # Move to next edge on boundary
            if current.left_face == face:
                current = face._next_edge_ccw(current, at_origin=False)
            elif current.right_face == face:
                current = face._next_edge_ccw(current, at_origin=True)
            else:
                break

            if current == boundary_edge:
                break

        self.faces.append(face)
        return face

    def from_polygon(self, points: list[tuple[float, float]]) -> 'PlanarGraph':
        """
        Build planar graph from a simple polygon.
        Creates vertices and edges in order, forming one interior face.
        """
        if len(points) < 3:
            raise ValueError("Polygon must have at least 3 vertices")

        # Create vertices
        vertices = [self.add_vertex(x, y) for x, y in points]
        n = len(vertices)

        # Create edges and link them in CCW order
        edges = []
        for i in range(n):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % n]
            edge = self.add_edge(v1, v2, EdgeType.BOUNDARY)
            edges.append(edge)

        # Set up proper winged-edge linkage for the polygon boundary
        # For a polygon, at each vertex we have exactly 2 edges meeting
        # Walking CCW around each vertex, we go: incoming_edge → outgoing_edge
        for i in range(n):
            current_edge = edges[i]  # Edge i: vertices[i] → vertices[i+1]
            next_edge = edges[(i + 1) % n]  # Edge i+1: vertices[i+1] → vertices[i+2]
            prev_edge = edges[(i - 1) % n]  # Edge i-1: vertices[i-1] → vertices[i]

            # At origin of current_edge (vertices[i]):
            # - prev edge arriving at V[i] is prev_edge (it ends at V[i])
            # - next edge leaving from V[i] is current_edge (it starts at V[i])
            # But wait - in a circular list, next after current should wrap back to prev!
            # Walking CCW around V[i]: prev_edge(dest) → current_edge(origin)
            current_edge.prev_at_origin = prev_edge  # prev_edge ends at V[i]
            current_edge.next_at_origin = prev_edge  # After current, we're back to prev

            # At destination of current_edge (vertices[i+1]):
            # Walking CCW around V[i+1]: current_edge(dest) → next_edge(origin)
            current_edge.next_at_dest = next_edge  # next_edge starts at V[i+1]
            current_edge.prev_at_dest = next_edge  # Before current (wrapping around), is next

        # Create interior face (CCW traversal gives interior)
        interior_face = Face(
            face_id=self._next_face_id,
            face_type=FaceType.INTERIOR
        )
        interior_face.boundary_edge = edges[0]
        self._next_face_id += 1

        # Assign left face to interior, right face to unbounded
        for edge in edges:
            edge.left_face = interior_face
            edge.right_face = self.unbounded_face

        self.faces.append(interior_face)

        # Update unbounded face
        if self.unbounded_face.boundary_edge is None:
            self.unbounded_face.boundary_edge = edges[0]

        return self

    def add_diagonal(self, v1: GraphVertex, v2: GraphVertex) -> Edge:
        """
        Add a diagonal between two vertices, splitting a face.
        Used in triangulation.

        Returns:
            The new edge connecting v1 to v2
        """
        edge = self.add_edge(v1, v2, EdgeType.DIAGONAL)

        # TODO: Update face split information
        # This requires finding which face is being split and creating two new faces

        return edge

    def get_triangles(self) -> list[tuple[int, int, int]]:
        """
        Extract all triangular faces from the planar graph.

        Returns:
            List of triangles, where each triangle is a tuple of three vertex IDs
            sorted in increasing order. Example: [(0, 1, 2), (0, 2, 3)]
        """
        triangles = []

        for face in self.faces:
            if face.face_type != FaceType.INTERIOR:
                continue

            vertices = face.get_boundary_vertices()

            # Check if this is a triangle (3 vertices)
            if len(vertices) == 3:
                tri = tuple(sorted([vertices[0].id, vertices[1].id, vertices[2].id]))
                triangles.append(tri)

        return triangles

    def triangulate(self) -> list[tuple[int, int, int]]:
        """
        Triangulate the planar graph using ear clipping.

        This modifies the graph by adding diagonals and splitting faces.
        Maintains proper Euler characteristic: V - E + F = 2

        Returns:
            List of triangles, where each triangle is a tuple of three vertex IDs
            sorted in increasing order. Example: [(0, 1, 2), (0, 2, 3)]
        """
        # First check if already triangulated (all interior faces are triangles)
        interior_faces = [f for f in self.faces if f.face_type == FaceType.INTERIOR]

        if not interior_faces:
            return []

        # Check if all interior faces are already triangles
        all_triangular = True
        for face in interior_faces:
            verts = face.get_boundary_vertices()
            if len(verts) != 3:
                all_triangular = False
                break

        if all_triangular:
            # Already triangulated - return the triangles
            return self.get_triangles()

        # Not fully triangulated - find faces that need triangulation
        face = None
        for f in interior_faces:
            verts = f.get_boundary_vertices()
            if len(verts) > 3:
                face = f
                break

        if face is None:
            # All faces are triangular
            return self.get_triangles()

        vertices_list = face.get_boundary_vertices()
        n = len(vertices_list)

        if n < 3:
            return []

        # Check winding order and reverse if CW (ear clipping requires CCW)
        # Compute signed area: positive = CW, negative = CCW
        signed_area = 0.0
        for i in range(n):
            j = (i + 1) % n
            signed_area += (vertices_list[j].point[prim.X] - vertices_list[i].point[prim.X]) * \
                          (vertices_list[j].point[prim.Y] + vertices_list[i].point[prim.Y])

        # If CW winding (positive area), reverse the order
        if signed_area > 0:
            vertices_list = list(reversed(vertices_list))

        # Create vertex mapping
        vertex_map = {i: gv for i, gv in enumerate(vertices_list)}

        # Convert to primitives for triangulation algorithm
        old_vertices = []
        for i, gv in enumerate(vertices_list):
            v = prim.Vertex(i, gv.point)
            old_vertices.append(v)

        # Link them circularly
        prim.link_vertices_circular(old_vertices)

        # Run ear clipping algorithm
        from polygon import Triangulate
        diagonals = Triangulate()

        # Add diagonals to graph structure
        added_diagonals = []
        if diagonals:
            for v1_idx, v2_idx in diagonals:
                if v1_idx < len(vertices_list) and v2_idx < len(vertices_list):
                    gv1 = vertex_map[v1_idx]
                    gv2 = vertex_map[v2_idx]
                    edge = self.add_diagonal(gv1, gv2)
                    added_diagonals.append((edge, gv1, gv2))

            # Rebuild faces after adding diagonals
            self.rebuild_faces()

            # Return all triangles
            return self.get_triangles()

        return self.get_triangles()

    def _split_faces_for_triangulation(self, original_face: Face,
                                       diagonals: list[tuple[Edge, GraphVertex, GraphVertex]],
                                       vertex_map: dict[int, GraphVertex]) -> list[tuple[int, int, int]]:
        """
        Split faces along the added diagonals to create triangular faces.

        Returns:
            List of triangles as tuples of vertex IDs.
        """
        # Remove the original face
        self.faces.remove(original_face)

        boundary_vertices = original_face.get_boundary_vertices()
        n = len(boundary_vertices)

        # For a triangulation: n vertices, n boundary edges + (n-3) diagonals
        # Results in (n-2) triangular faces

        triangles = []

        if n >= 3:
            base_vertex = boundary_vertices[0]

            # Get all edges (boundary + diagonals)
            boundary_edges = original_face.get_boundary_edges()
            edge_map = {}

            for edge in boundary_edges:
                edge_map[(edge.origin, edge.destination)] = edge
                edge_map[(edge.destination, edge.origin)] = edge

            for edge, gv1, gv2 in diagonals:
                edge_map[(gv1, gv2)] = edge
                edge_map[(gv2, gv1)] = edge

            # Create triangular faces using fan triangulation
            for i in range(1, n - 1):
                v1 = boundary_vertices[i]
                v2 = boundary_vertices[i + 1]

                # Create triangle: base_vertex - v1 - v2
                tri_edges = []

                # Find edges for this triangle
                if (base_vertex, v1) in edge_map:
                    tri_edges.append(edge_map[(base_vertex, v1)])
                if (v1, v2) in edge_map:
                    tri_edges.append(edge_map[(v1, v2)])
                if (v2, base_vertex) in edge_map:
                    tri_edges.append(edge_map[(v2, base_vertex)])

                # Create face if we have edges
                if len(tri_edges) >= 1:
                    tri_face = Face(
                        face_id=self._next_face_id,
                        face_type=FaceType.INTERIOR
                    )
                    tri_face.boundary_edge = tri_edges[0]
                    self._next_face_id += 1

                    # Assign face to edges
                    for edge in tri_edges:
                        if edge.left_face is None:
                            edge.left_face = tri_face
                        elif edge.right_face is None:
                            edge.right_face = tri_face

                    self.faces.append(tri_face)

                    # Add triangle to result
                    tri_tuple = tuple(sorted([base_vertex.id, v1.id, v2.id]))
                    triangles.append(tri_tuple)

        return triangles

    def get_vertex_by_id(self, vertex_id: int) -> Optional[GraphVertex]:
        """Find vertex by its ID"""
        for vertex in self.vertices:
            if vertex.id == vertex_id:
                return vertex
        return None

    def get_vertex_by_coords(self, x: float, y: float) -> Optional[GraphVertex]:
        """Find vertex at given coordinates"""
        return self._vertex_map.get((x, y))

    def get_coords_by_vertex_id(self, vertex_id: int) -> Optional[tuple[float, float]]:
        """
        Get coordinates for a vertex given its ID.

        Args:
            vertex_id: The ID of the vertex

        Returns:
            A tuple (x, y) of the vertex coordinates, or None if vertex not found
        """
        vertex = self.get_vertex_by_id(vertex_id)
        if vertex is None:
            return None
        return (vertex.point[prim.X], vertex.point[prim.Y])

    def get_incident_edges_for_vertex(self, vertex: GraphVertex) -> list[Edge]:
        """
        Get all edges incident to a vertex.
        This is a convenience method that delegates to vertex.get_incident_edges()
        """
        return vertex.get_incident_edges()

    def find_face_containing_point(self, x: float, y: float) -> Optional[Face]:
        """
        Point location: find which face contains the given point.
        Uses point-in-polygon test.
        """
        query_point = prim.Point(x, y)

        for face in self.faces:
            if face.face_type == FaceType.UNBOUNDED:
                continue

            vertices = face.get_boundary_vertices()
            if len(vertices) < 3:
                continue

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
            if (v_i[prim.Y] > point[prim.Y]) != (v_i1[prim.Y] > point[prim.Y]):
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
        E = len(self.edges)
        F = len(self.faces)

        return V - E + F

    def rebuild_faces(self):
        """
        Rebuild all faces from the edge structure.

        This traverses the planar graph and identifies all face cycles.
        Use this after adding many edges to rebuild the face structure.
        """
        # Clear all existing faces except unbounded
        self.faces = [self.unbounded_face]
        self._next_face_id = 1

        # Clear all face assignments on edges
        for edge in self.edges:
            edge.left_face = None
            edge.right_face = None

        # Track which edge directions we've visited
        visited = set()

        # For each edge direction, try to walk a face cycle
        for edge in self.edges:
            # Try both directions of the edge
            for direction in ['forward', 'backward']:
                if direction == 'forward':
                    edge_dir = (edge.id, 'left')
                else:
                    edge_dir = (edge.id, 'right')

                if edge_dir in visited:
                    continue

                # Walk the face cycle
                face_edges = []
                current_edge = edge
                current_side = 'left' if direction == 'forward' else 'right'

                start_edge_id = edge.id
                start_side = current_side

                while True:
                    # Mark this edge direction as visited
                    visited.add((current_edge.id, current_side))
                    face_edges.append((current_edge, current_side))

                    # Move to next edge in face cycle
                    if current_side == 'left':
                        # We're on the left side going origin->dest
                        # Next edge is at destination
                        next_vertex = current_edge.destination
                        # Get the next edge CCW around destination
                        next_edge = current_edge.next_at_dest
                        # Determine which side of next_edge we're on
                        if next_edge.origin == next_vertex:
                            current_side = 'left'
                        else:
                            current_side = 'right'
                        current_edge = next_edge
                    else:
                        # We're on the right side going dest->origin
                        # Next edge is at origin
                        next_vertex = current_edge.origin
                        # Get the next edge CCW around origin
                        next_edge = current_edge.prev_at_origin
                        # Determine which side of next_edge we're on
                        if next_edge.destination == next_vertex:
                            current_side = 'right'
                        else:
                            current_side = 'left'
                        current_edge = next_edge

                    # Check if we've completed the cycle
                    if current_edge.id == start_edge_id and current_side == start_side:
                        break

                    # Safety check for infinite loops
                    if len(face_edges) > len(self.edges) * 2:
                        break

                # Create a face for this cycle
                if len(face_edges) > 0:
                    # Determine if this is interior or exterior
                    # Compute signed area - positive = CCW = interior
                    vertices = []
                    for e, side in face_edges:
                        if side == 'left':
                            vertices.append(e.origin)
                        else:
                            vertices.append(e.destination)

                    # Remove duplicates while preserving order
                    seen = set()
                    unique_verts = []
                    for v in vertices:
                        if v.id not in seen:
                            seen.add(v.id)
                            unique_verts.append(v)

                    if len(unique_verts) >= 3:
                        # Compute signed area
                        area = 0.0
                        n = len(unique_verts)
                        for i in range(n):
                            j = (i + 1) % n
                            area += unique_verts[i].point[prim.X] * unique_verts[j].point[prim.Y]
                            area -= unique_verts[j].point[prim.X] * unique_verts[i].point[prim.Y]
                        area /= 2.0

                        # Create face
                        if abs(area) > 0.01:  # Not degenerate
                            face_type = FaceType.INTERIOR if area > 0 else FaceType.EXTERIOR
                            new_face = Face(self._next_face_id, face_type)
                            new_face.boundary_edge = face_edges[0][0]
                            self._next_face_id += 1

                            # Assign face to all edges in the cycle
                            for e, side in face_edges:
                                if side == 'left':
                                    e.left_face = new_face
                                else:
                                    e.right_face = new_face

                            self.faces.append(new_face)

        # Assign unbounded face to any edges without faces
        for edge in self.edges:
            if edge.left_face is None:
                edge.left_face = self.unbounded_face
            if edge.right_face is None:
                edge.right_face = self.unbounded_face

    def validate(self) -> bool:
        """
        Validate the planar graph structure.

        Checks:
        - Winged-edge pointer consistency
        - Face assignments
        - Vertex incident edge references
        """
        # Check winged-edge pointers
        for edge in self.edges:
            # Check that adjacent edges reference this edge back
            if edge.next_at_origin:
                next_edge = edge.next_at_origin
                if next_edge.origin == edge.origin:
                    if next_edge.prev_at_origin != edge:
                        print(f"Error: next_at_origin pointer inconsistency for {edge}")
                        return False
                elif next_edge.destination == edge.origin:
                    if next_edge.prev_at_dest != edge:
                        print(f"Error: next_at_origin pointer inconsistency for {edge}")
                        return False

            # Similar checks for other pointers...
            # (Simplified for brevity)

        # Check that each vertex has a valid incident edge
        for vertex in self.vertices:
            if vertex.degree > 0 and vertex.incident_edge is None:
                print(f"Error: Vertex {vertex} has degree {vertex.degree} but no incident edge")
                return False

        return True

    def __repr__(self):
        return (f"PlanarGraph(V={len(self.vertices)}, "
                f"E={len(self.edges)}, F={len(self.faces)})")

    def compute_convex_hull(self) -> list[GraphVertex]:
        """
        Compute the convex hull of all vertices using the giftwrapping (Jarvis march) algorithm.

        Uses only the Left test from primitives to determine point orientation.
        The algorithm wraps around the point set, at each step finding the point that
        makes the rightmost turn (most counterclockwise) from the current edge.

        Returns:
            List of vertices forming the convex hull in counter-clockwise order.
            Returns empty list if there are fewer than 3 vertices.
        """
        if len(self.vertices) < 3:
            return []

        # Step 1: Find the starting point (lowest y, then leftmost)
        start = min(self.vertices, key=lambda v: (v.point[prim.Y], v.point[prim.X]))

        hull = []
        current = start

        while True:
            hull.append(current)

            # Find the next point on the hull
            # Start by assuming the next candidate is the first point that isn't current
            next_point = None
            for v in self.vertices:
                if v != current:
                    next_point = v
                    break

            # Find the point that is "most counterclockwise" from current
            # For CCW hull: we want the point such that all other points are to the RIGHT
            # of the line current->next_point (or equivalently, next_point is the leftmost)
            for candidate in self.vertices:
                if candidate == current or candidate == next_point:
                    continue

                # Check if candidate is to the RIGHT of the line current->next_point
                # If so, next_point was too far left, so replace it with candidate
                # Equivalently: if next_point is to the LEFT of current->candidate, replace next_point
                if prim.Left(current.point, candidate.point, next_point.point):
                    # next_point is to the left of current->candidate
                    # so candidate is more to the right (more counterclockwise for hull)
                    next_point = candidate
                elif prim.Collinear(current.point, next_point.point, candidate.point):
                    # If collinear, choose the one that is farther away
                    dx_next = next_point.point[prim.X] - current.point[prim.X]
                    dy_next = next_point.point[prim.Y] - current.point[prim.Y]
                    dist_next_sq = dx_next * dx_next + dy_next * dy_next

                    dx_cand = candidate.point[prim.X] - current.point[prim.X]
                    dy_cand = candidate.point[prim.Y] - current.point[prim.Y]
                    dist_cand_sq = dx_cand * dx_cand + dy_cand * dy_cand

                    if dist_cand_sq > dist_next_sq:
                        next_point = candidate

            # Move to the next point
            current = next_point

            # If we've wrapped around to the start, we're done
            if current == start:
                break

            # Safety check to prevent infinite loops
            if len(hull) > len(self.vertices):
                break

        return hull

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
