import polygon as poly
import primitives as prim
import planar_graph as pg
import search_and_intersection as si
import kirkpatrick as kp

# Test 1: Create a simple triangle
print("=== Test 1: Triangle ===")
poly.create_polygon([
    (0, 0),
    (4, 0),
    (2, 3)
])

twice_area = prim.AreaPoly2()
print(f"Twice the area: {twice_area}")
print(f"Actual area: {twice_area / 2}")

# Test 2: Create a square and triangulate
print("\n=== Test 2: Square Triangulation ===")
poly.create_polygon([
    (0, 0),
    (4, 0),
    (4, 4),
    (0, 4)
])

print("Before triangulation:")
twice_area = prim.AreaPoly2()
print(f"Twice the area: {twice_area}")

print("\nTriangulating...")
poly.Triangulate()

# Test 3: Pentagon
print("\n=== Test 3: Pentagon Triangulation ===")
poly.create_polygon([
    (0, 0),
    (5, 0),
    (6, 4),
    (2, 6),
    (-1, 3)
])

print("Triangulating pentagon...")
poly.Triangulate()

# Test 4: Simple Square Independent Set
print("\n=== Test 4: Simple Square Independent Set ===")
square_graph = pg.PlanarGraph()
square_graph.from_polygon([
    (0, 0),
    (4, 0),
    (4, 4),
    (0, 4)
])

print(f"Graph structure: {square_graph}")
print(f"Graph is valid: {square_graph.validate()}")
print(f"Euler characteristic: {square_graph.euler_characteristic()}")

print(f"\nVertices in graph: {len(square_graph.vertices)}")
for v in square_graph.vertices:
    print(f"  {v}")

print(f"\nConstructing independent set...")
try:
    independent_set = si.ConstructIndependentSet(square_graph)
    print(f"Independent set size: {len(independent_set)}")
    print(f"Independent set vertices: {sorted([v.id for v in independent_set])}")
    
    # Verify it's a valid independent set
    print(f"\nVerifying independent set...")
    is_valid = True
    for v1 in independent_set:
        neighbors = v1.get_neighbors()
        for v2 in independent_set:
            if v1 != v2 and v2 in neighbors:
                print(f"ERROR: {v1} and {v2} are neighbors but both in independent set!")
                is_valid = False
    if is_valid:
        print("✓ Independent set is valid (no two vertices share an edge)")
    
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Complex Hexagon Independent Set
print("\n=== Test 5: Complex Hexagon Independent Set ===")
# Create a hexagon with internal vertex and triangulation
hexagon_graph = pg.PlanarGraph()
hexagon_vertices = [
    (0, 0),
    (3, 0),
    (4.5, 2),
    (3, 4),
    (0, 4),
    (-1.5, 2)
]
hexagon_graph.from_polygon(hexagon_vertices)

print(f"Graph structure: {hexagon_graph}")
print(f"Graph is valid: {hexagon_graph.validate()}")
print(f"Euler characteristic: {hexagon_graph.euler_characteristic()}")

print(f"\nVertices in hexagon graph: {len(hexagon_graph.vertices)}")
for v in hexagon_graph.vertices:
    neighbors = v.get_neighbors()
    print(f"  {v} - degree={v.degree}, neighbors=[{', '.join([f'V{n.id}' for n in neighbors])}]")

# Add some internal vertices for complexity
internal_vertex = hexagon_graph.add_vertex(1.5, 2)
print(f"\nAdded internal vertex: {internal_vertex}")

# Add diagonals to create a more complex structure
if len(hexagon_graph.vertices) >= 7:
    hexagon_graph.add_edge(hexagon_graph.vertices[0], hexagon_graph.vertices[3], pg.EdgeType.DIAGONAL)
    hexagon_graph.add_edge(hexagon_graph.vertices[1], hexagon_graph.vertices[4], pg.EdgeType.DIAGONAL)
    hexagon_graph.add_edge(hexagon_graph.vertices[2], hexagon_graph.vertices[5], pg.EdgeType.DIAGONAL)
    hexagon_graph.add_edge(hexagon_graph.vertices[0], internal_vertex, pg.EdgeType.INTERNAL)
    hexagon_graph.add_edge(hexagon_graph.vertices[2], internal_vertex, pg.EdgeType.INTERNAL)
    hexagon_graph.add_edge(hexagon_graph.vertices[4], internal_vertex, pg.EdgeType.INTERNAL)

print(f"\nUpdated graph: {hexagon_graph}")
print(f"Updated graph vertices: {len(hexagon_graph.vertices)}")
for v in hexagon_graph.vertices:
    neighbors = v.get_neighbors()
    print(f"  {v} - degree={v.degree}, neighbors=[{', '.join([f'V{n.id}' for n in neighbors])}]")

print(f"\nConstructing independent set for hexagon...")
try:
    independent_set = si.ConstructIndependentSet(hexagon_graph)
    print(f"Independent set size: {len(independent_set)}")
    print(f"Independent set vertices: {sorted([v.id for v in independent_set])}")
    
    # Verify it's a valid independent set
    print(f"\nVerifying independent set...")
    is_valid = True
    for v1 in independent_set:
        neighbors = v1.get_neighbors()
        for v2 in independent_set:
            if v1 != v2 and v2 in neighbors:
                print(f"ERROR: {v1} and {v2} are neighbors but both in independent set!")
                is_valid = False
    if is_valid:
        print("✓ Independent set is valid (no two vertices share an edge)")
    
    # Count how many vertices are marked
    marked_count = len(independent_set)
    unmarked_count = len(hexagon_graph.vertices) - marked_count
    print(f"\nStatistics:")
    print(f"  Independent set vertices: {marked_count}")
    print(f"  Excluded vertices: {unmarked_count}")
    print(f"  Total vertices: {len(hexagon_graph.vertices)}")
    
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# Test 6: Planar Graph Triangulation
print("\n=== Test 6: Planar Graph Triangulation ===")
triangle_graph = pg.PlanarGraph()
triangle_graph.from_polygon([
    (0, 0),
    (5, 0),
    (2.5, 4)
])

print(f"Triangle graph (already triangulated):")
print(f"  Structure: {triangle_graph}")
print(f"  Valid: {triangle_graph.validate()}")
print(f"  Euler: {triangle_graph.euler_characteristic()}")

print(f"\n--- Testing square triangulation ---")
square_tri_graph = pg.PlanarGraph()
square_tri_graph.from_polygon([
    (0, 0),
    (4, 0),
    (4, 4),
    (0, 4)
])

print(f"Before triangulation:")
print(f"  Structure: {square_tri_graph}")
print(f"  Vertices: {len(square_tri_graph.vertices)}")
print(f"  Edges: {len(square_tri_graph.edges) // 2}")
for v in square_tri_graph.vertices:
    print(f"    {v} - degree={v.degree}")

print(f"\nTriangulating square graph...")
triangles = square_tri_graph.triangulate()
print(f"Triangles returned: {triangles}")

print(f"\nAfter triangulation:")
print(f"  Structure: {square_tri_graph}")
print(f"  Vertices: {len(square_tri_graph.vertices)}")
print(f"  Edges: {len(square_tri_graph.edges) // 2}")
for v in square_tri_graph.vertices:
    neighbors = v.get_neighbors()
    print(f"    {v} - degree={v.degree}, neighbors=[{', '.join([f'V{n.id}' for n in neighbors])}]")
print(f"  Valid: {square_tri_graph.validate()}")
print(f"  Euler: {square_tri_graph.euler_characteristic()}")

print(f"\n--- Testing pentagon triangulation ---")
pent_graph = pg.PlanarGraph()
pent_graph.from_polygon([
    (0, 0),
    (4, 0),
    (5, 3),
    (2, 5),
    (-1, 2)
])

print(f"Before pentagon triangulation:")
print(f"  Structure: {pent_graph}")
print(f"  Edges: {len(pent_graph.edges) // 2}")

print(f"\nTriangulating pentagon...")
pent_triangles = pent_graph.triangulate()
print(f"Triangles returned: {pent_triangles}")

print(f"\nAfter pentagon triangulation:")
print(f"  Structure: {pent_graph}")
print(f"  Edges: {len(pent_graph.edges) // 2}")
print(f"  Valid: {pent_graph.validate()}")
print(f"  Euler: {pent_graph.euler_characteristic()}")

# Test 7: Kirkpatrick Point Location - Simple Triangle
print("\n" + "="*60)
print("=== Test 7: Kirkpatrick DAG - Simple Triangle ===")
print("="*60)
try:
    triangle_pl = pg.PlanarGraph()
    triangle_pl.from_polygon([
        (0, 0),
        (6, 0),
        (3, 5)
    ])

    print(f"Initial graph: {triangle_pl}")
    print(f"Vertices: {len(triangle_pl.vertices)}")

    # Clone before building DAG (since BuildDAG modifies the graph)
    triangle_clone = triangle_pl.clone()

    print("\nBuilding DAG...")
    root_node = kp.BuildDAG(triangle_clone)

    if root_node:
        print(f"✓ DAG root created successfully!")
        print(f"  Root triangle: {root_node.value}")
        print(f"  Root has {len(root_node.children)} children")

        # Count total nodes in DAG
        def count_nodes(node, visited=None):
            if visited is None:
                visited = set()
            if id(node) in visited:
                return 0
            visited.add(id(node))
            count = 1
            for child in node.children:
                count += count_nodes(child, visited)
            return count

        total_nodes = count_nodes(root_node)
        print(f"  Total DAG nodes: {total_nodes}")
    else:
        print("ERROR: BuildDAG returned None")

except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# Test 8: Kirkpatrick Point Location - Square
print("\n" + "="*60)
print("=== Test 8: Kirkpatrick DAG - Square ===")
print("="*60)
try:
    square_pl = pg.PlanarGraph()
    square_pl.from_polygon([
        (0, 0),
        (8, 0),
        (8, 8),
        (0, 8)
    ])

    print(f"Initial graph: {square_pl}")
    print(f"Vertices: {len(square_pl.vertices)}")

    square_clone = square_pl.clone()

    print("\nBuilding DAG...")
    root_node = kp.BuildDAG(square_clone)

    if root_node:
        print(f"✓ DAG root created successfully!")
        print(f"  Root triangle: {root_node.value}")
        print(f"  Root has {len(root_node.children)} children")

        # Count nodes and depth
        def analyze_dag(node, depth=0, visited=None):
            if visited is None:
                visited = set()
            if id(node) in visited:
                return 0, depth
            visited.add(id(node))

            count = 1
            max_depth = depth
            for child in node.children:
                child_count, child_depth = analyze_dag(child, depth + 1, visited)
                count += child_count
                max_depth = max(max_depth, child_depth)
            return count, max_depth

        total_nodes, max_depth = analyze_dag(root_node)
        print(f"  Total DAG nodes: {total_nodes}")
        print(f"  DAG depth: {max_depth}")

        # Print hierarchy structure
        print(f"\n  DAG Structure:")
        def print_dag(node, level=0, visited=None):
            if visited is None:
                visited = set()
            if id(node) in visited:
                return
            visited.add(id(node))

            indent = "  " * level
            print(f"{indent}Level {level}: Triangle {node.value} -> {len(node.children)} children")
            for child in node.children:
                print_dag(child, level + 1, visited)

        print_dag(root_node)
    else:
        print("ERROR: BuildDAG returned None")

except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# Test 9: Kirkpatrick Point Location - Hexagon
print("\n" + "="*60)
print("=== Test 9: Kirkpatrick DAG - Hexagon ===")
print("="*60)
try:
    hex_pl = pg.PlanarGraph()
    hex_pl.from_polygon([
        (0, 0),
        (4, 0),
        (6, 3),
        (4, 6),
        (0, 6),
        (-2, 3)
    ])

    print(f"Initial graph: {hex_pl}")
    print(f"Vertices: {len(hex_pl.vertices)}")

    hex_clone = hex_pl.clone()

    print("\nBuilding DAG...")
    root_node = kp.BuildDAG(hex_clone)

    if root_node:
        print(f"✓ DAG root created successfully!")
        print(f"  Root triangle: {root_node.value}")

        total_nodes, max_depth = analyze_dag(root_node)
        print(f"  Total DAG nodes: {total_nodes}")
        print(f"  DAG depth: {max_depth}")

        # Verify hierarchy size
        print(f"\n  Modified graph after DAG construction: {hex_clone}")
        print(f"  Vertices after: {len(hex_clone.vertices)}")
    else:
        print("ERROR: BuildDAG returned None")

except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# Test 10: Point Location Query Function
print("\n" + "="*60)
print("=== Test 10: Point Location Queries ===")
print("="*60)

def point_location_query(root_node, query_point):
    """
    Perform a point location query in the DAG.
    Traverses from root down to find the triangle containing the query point.
    """
    if root_node is None:
        return None

    current = root_node
    path = []

    while True:
        path.append(current.value)

        # If this is a leaf node (no children), we found the triangle
        if not current.children:
            return current, path

        # Find which child contains the query point
        found_child = None
        for child in current.children:
            # Get the triangle vertices
            triangle_vertices = child.value
            # This assumes child.value is a tuple of vertex IDs
            # We would need to map these to actual coordinates
            # For now, we'll just traverse to the first child
            found_child = child
            break

        if found_child:
            current = found_child
        else:
            # No child found, return current
            return current, path

try:
    # Build a test graph
    test_pl = pg.PlanarGraph()
    test_pl.from_polygon([
        (0, 0),
        (10, 0),
        (10, 10),
        (0, 10)
    ])

    test_clone = test_pl.clone()
    print("Building DAG for point location queries...")
    root = kp.BuildDAG(test_clone)

    if root:
        print(f"✓ DAG constructed successfully")

        # Test some point queries
        test_points = [
            (5, 5),   # Center
            (2, 2),   # Inside, lower-left
            (8, 8),   # Inside, upper-right
            (0, 0),   # Vertex
            (5, 0),   # Edge
        ]

        print(f"\nTesting point location queries:")
        for px, py in test_points:
            query_pt = prim.Point(px, py)
            result, path = point_location_query(root, query_pt)
            print(f"  Query point ({px}, {py}):")
            print(f"    Path length: {len(path)}")
            print(f"    Final triangle: {result.value if result else 'None'}")
    else:
        print("ERROR: Could not build DAG")

except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("All Kirkpatrick Point Location tests completed!")
print("="*60)

# Test 11: Custom PSLG from user specification
print("\n" + "="*60)
print("=== Test 11: Custom PSLG DAG - User Provided Graph ===")
print("="*60)
try:
    # Create a properly triangulated planar graph with the specified vertices and edges
    # User provided coordinates and connectivity:
    # v1 (332, 392) -> incident to v4, v6
    # v2 (196, 436) -> incident to v3, v4, v5
    # v3 (176, 304) -> incident to v2, v5, v6
    # v4 (232, 484) -> incident to v1, v2, v5, v6
    # v5 (244, 388) -> incident to v2, v3, v4, v6
    # v6 (336, 328) -> incident to v1, v3, v4, v5

    # To create a valid planar subdivision, we need a boundary polygon
    # Examining the coordinates, the convex hull vertices in CCW order are:
    # v3 (176, 304), v6 (336, 328), v1 (332, 392), v4 (232, 484), v2 (196, 436)

    custom_pslg = pg.PlanarGraph()

    # Create outer boundary polygon (convex hull in CCW order)
    boundary_coords = [
        (176, 304),   # v3
        (336, 328),   # v6
        (332, 392),   # v1
        (232, 484),   # v4
        (196, 436),   # v2
    ]
    custom_pslg.from_polygon(boundary_coords)

    print(f"Created boundary polygon: {custom_pslg}")
    print(f"  Vertices: {len(custom_pslg.vertices)}")
    print(f"  Edges: {len(custom_pslg.edges) // 2}")
    print(f"  Euler: {custom_pslg.euler_characteristic()}")

    # Add internal vertex v5 (244, 388)
    v5_internal = custom_pslg.add_vertex(244, 388)
    print(f"\nAdded internal vertex v5: {v5_internal}")

    # Get references to boundary vertices by coordinates
    vertices_by_coords = {
        (176, 304): custom_pslg.get_vertex_by_coords(176, 304),  # v3
        (336, 328): custom_pslg.get_vertex_by_coords(336, 328),  # v6
        (332, 392): custom_pslg.get_vertex_by_coords(332, 392),  # v1
        (232, 484): custom_pslg.get_vertex_by_coords(232, 484),  # v4
        (196, 436): custom_pslg.get_vertex_by_coords(196, 436),  # v2
    }

    v3 = vertices_by_coords[(176, 304)]
    v6 = vertices_by_coords[(336, 328)]
    v1 = vertices_by_coords[(332, 392)]
    v4 = vertices_by_coords[(232, 484)]
    v2 = vertices_by_coords[(196, 436)]
    v5 = v5_internal

    print(f"\nVertex mapping:")
    print(f"  v1 (id={v1.id}): {v1}")
    print(f"  v2 (id={v2.id}): {v2}")
    print(f"  v3 (id={v3.id}): {v3}")
    print(f"  v4 (id={v4.id}): {v4}")
    print(f"  v5 (id={v5.id}): {v5} (internal)")
    print(f"  v6 (id={v6.id}): {v6}")

    # Now triangulate the graph
    print(f"\nTriangulating...")
    triangles = custom_pslg.triangulate()

    print(f"\nAfter triangulation:")
    print(f"  Graph: {custom_pslg}")
    print(f"  Triangles: {triangles}")
    print(f"  Euler: {custom_pslg.euler_characteristic()}")

    print(f"\nVertex degrees after triangulation:")
    for v in custom_pslg.vertices:
        neighbors = v.get_neighbors()
        neighbor_ids = sorted([n.id for n in neighbors])
        print(f"  {v} - degree={v.degree}, neighbors={neighbor_ids}")

    is_valid = custom_pslg.validate()
    print(f"\nValidating graph: {is_valid}")

    if is_valid and triangles:
        print(f"\nCloning graph for DAG construction...")
        custom_clone = custom_pslg.clone()

        print(f"\nBuilding DAG...")
        print(f"NOTE: BuildDAG should return a single root node (the head of the DAG hierarchy)")
        root_node = kp.BuildDAG(custom_clone)

        if root_node:
            print(f"✓ DAG root created successfully!")
            print(f"  Root triangle: {root_node.value}")
            print(f"  Root has {len(root_node.children)} children")

            # Analyze DAG structure
            def analyze_dag(node, depth=0, visited=None):
                if visited is None:
                    visited = set()
                if id(node) in visited:
                    return 0, depth
                visited.add(id(node))

                count = 1
                max_depth = depth
                for child in node.children:
                    child_count, child_depth = analyze_dag(child, depth + 1, visited)
                    count += child_count
                    max_depth = max(max_depth, child_depth)
                return count, max_depth

            total_nodes, max_depth = analyze_dag(root_node)
            print(f"  Total DAG nodes: {total_nodes}")
            print(f"  DAG depth: {max_depth}")

            # Print DAG structure
            print(f"\n  DAG Structure (first 8 levels):")
            def print_dag(node, level=0, visited=None, max_levels=8):
                if visited is None:
                    visited = set()
                if id(node) in visited or level >= max_levels:
                    if level >= max_levels:
                        print(f"{'  ' * level}... (truncated)")
                    return
                visited.add(id(node))

                indent = "  " * level
                tri_str = f"({node.value[0]}, {node.value[1]}, {node.value[2]})"
                print(f"{indent}L{level}: Tri{tri_str} -> {len(node.children)} children")
                for child in node.children:
                    print_dag(child, level + 1, visited, max_levels)

            print_dag(root_node)

            print(f"\n  Modified graph after DAG construction:")
            print(f"    {custom_clone}")
            print(f"    Vertices after: {len(custom_clone.vertices)}")
        else:
            print("ERROR: BuildDAG returned None")
    else:
        print("ERROR: Graph is invalid or not triangulated, cannot build DAG")

except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("Custom PSLG test completed!")
print("="*60)

# Test 12: New Custom PSLG with 6 vertices
print("\n" + "="*60)
print("=== Test 12: Custom 6-Vertex PSLG DAG ===")
print("="*60)
try:
    # Vertex coordinates and connectivity:
    # v1 -> (332, 392) -> incident to v4, v6
    # v2 -> (196, 436) -> incident to v3, v4, v5
    # v3 -> (176, 304) -> incident to v2, v5, v6
    # v4 -> (232, 484) -> incident to v1, v2, v5, v6
    # v5 -> (244, 388) -> incident to v2, v3, v4, v6
    # v6 -> (336, 328) -> incident to v1, v3, v4, v5

    # This is a complete graph K6, so we need to find a proper outer boundary
    # Looking at the coordinates, we can determine the convex hull
    # Let me compute the convex hull manually:
    # v3 (176, 304) - leftmost
    # v2 (196, 436) - top-left
    # v4 (232, 484) - topmost
    # v1 (332, 392) - right side
    # v6 (336, 328) - rightmost
    # Convex hull in CCW order: v3 -> v6 -> v1 -> v4 -> v2 -> v3

    custom_graph = pg.PlanarGraph()

    # Create the convex hull as the boundary polygon
    boundary_coords = [
        (176, 304),   # v3
        (336, 328),   # v6
        (332, 392),   # v1
        (232, 484),   # v4
        (196, 436),   # v2
    ]
    custom_graph.from_polygon(boundary_coords)

    print(f"Created boundary polygon: {custom_graph}")
    print(f"  Vertices: {len(custom_graph.vertices)}")
    print(f"  Edges: {len(custom_graph.edges) // 2}")

    # Add internal vertex v5
    v5_vertex = custom_graph.add_vertex(244, 388)
    print(f"Added internal vertex v5: {v5_vertex}")

    # Get vertices by their coordinates
    v3 = custom_graph.get_vertex_by_coords(176, 304)
    v6 = custom_graph.get_vertex_by_coords(336, 328)
    v1 = custom_graph.get_vertex_by_coords(332, 392)
    v4 = custom_graph.get_vertex_by_coords(232, 484)
    v2 = custom_graph.get_vertex_by_coords(196, 436)
    v5 = v5_vertex

    print(f"\nVertex mapping:")
    print(f"  v1 (id={v1.id}): {v1}")
    print(f"  v2 (id={v2.id}): {v2}")
    print(f"  v3 (id={v3.id}): {v3}")
    print(f"  v4 (id={v4.id}): {v4}")
    print(f"  v5 (id={v5.id}): {v5} (internal)")
    print(f"  v6 (id={v6.id}): {v6}")

    # Add internal edges according to the specification
    # The boundary already has edges: v3-v6, v6-v1, v1-v4, v4-v2, v2-v3
    # We need to add:
    # v1 to v4 (already on boundary), v1 to v6 (already on boundary) - v1 OK
    # v2 to v3 (already on boundary), v2 to v4 (already on boundary), v2 to v5 (need)
    # v3 to v2 (already on boundary), v3 to v5 (need), v3 to v6 (already on boundary)
    # v4 to v1 (already on boundary), v4 to v2 (already on boundary), v4 to v5 (need), v4 to v6 (need diagonal)
    # v5 to v2 (need), v5 to v3 (need), v5 to v4 (need), v5 to v6 (need)
    # v6 to v1 (already on boundary), v6 to v3 (already on boundary), v6 to v4 (need diagonal), v6 to v5 (need)

    custom_graph.add_edge(v2, v5, pg.EdgeType.INTERNAL)
    custom_graph.add_edge(v3, v5, pg.EdgeType.INTERNAL)
    custom_graph.add_edge(v4, v5, pg.EdgeType.INTERNAL)
    custom_graph.add_edge(v4, v6, pg.EdgeType.DIAGONAL)
    custom_graph.add_edge(v5, v6, pg.EdgeType.INTERNAL)

    print(f"\nGraph structure after adding internal edges: {custom_graph}")
    print(f"  Vertices: {len(custom_graph.vertices)}")
    print(f"  Edges: {len(custom_graph.edges) // 2}")

    print(f"\nVertex degrees:")
    for v in custom_graph.vertices:
        neighbors = v.get_neighbors()
        neighbor_ids = sorted([n.id for n in neighbors])
        print(f"  {v} - degree={v.degree}, neighbors={neighbor_ids}")

    # Verify connectivity matches specification
    print(f"\nVerifying connectivity matches specification:")
    print(f"  v1 ({v1.id}) should be incident to v4 ({v4.id}), v6 ({v6.id})")
    v1_neighbors = set(n.id for n in v1.get_neighbors())
    print(f"    Actual neighbors: {sorted(v1_neighbors)}")

    print(f"  v2 ({v2.id}) should be incident to v3 ({v3.id}), v4 ({v4.id}), v5 ({v5.id})")
    v2_neighbors = set(n.id for n in v2.get_neighbors())
    print(f"    Actual neighbors: {sorted(v2_neighbors)}")

    print(f"  v3 ({v3.id}) should be incident to v2 ({v2.id}), v5 ({v5.id}), v6 ({v6.id})")
    v3_neighbors = set(n.id for n in v3.get_neighbors())
    print(f"    Actual neighbors: {sorted(v3_neighbors)}")

    print(f"  v4 ({v4.id}) should be incident to v1 ({v1.id}), v2 ({v2.id}), v5 ({v5.id}), v6 ({v6.id})")
    v4_neighbors = set(n.id for n in v4.get_neighbors())
    print(f"    Actual neighbors: {sorted(v4_neighbors)}")

    print(f"  v5 ({v5.id}) should be incident to v2 ({v2.id}), v3 ({v3.id}), v4 ({v4.id}), v6 ({v6.id})")
    v5_neighbors = set(n.id for n in v5.get_neighbors())
    print(f"    Actual neighbors: {sorted(v5_neighbors)}")

    print(f"  v6 ({v6.id}) should be incident to v1 ({v1.id}), v3 ({v3.id}), v4 ({v4.id}), v5 ({v5.id})")
    v6_neighbors = set(n.id for n in v6.get_neighbors())
    print(f"    Actual neighbors: {sorted(v6_neighbors)}")

    # Validate graph structure
    is_valid = custom_graph.validate()
    print(f"\nGraph validation: {is_valid}")

    if is_valid:
        # Clone the graph before running BuildDAG
        print(f"\nCloning graph for DAG construction...")
        custom_clone = custom_graph.clone()

        print(f"\nBuilding DAG...")
        root_node = kp.BuildDAG(custom_clone)

        if root_node:
            print(f"✓ DAG root created successfully!")
            print(f"  Root triangle: {root_node.value}")
            print(f"  Root has {len(root_node.children)} children")

            # Analyze DAG structure
            def analyze_dag_test12(node, depth=0, visited=None):
                if visited is None:
                    visited = set()
                if id(node) in visited:
                    return 0, depth
                visited.add(id(node))

                count = 1
                max_depth = depth
                for child in node.children:
                    child_count, child_depth = analyze_dag_test12(child, depth + 1, visited)
                    count += child_count
                    max_depth = max(max_depth, child_depth)
                return count, max_depth

            total_nodes, max_depth = analyze_dag_test12(root_node)
            print(f"  Total DAG nodes: {total_nodes}")
            print(f"  DAG depth: {max_depth}")

            # Print DAG structure
            print(f"\n  DAG Structure:")
            def print_dag_test12(node, level=0, visited=None, max_levels=10):
                if visited is None:
                    visited = set()
                if id(node) in visited or level >= max_levels:
                    if level >= max_levels:
                        print(f"{'  ' * level}... (truncated)")
                    return
                visited.add(id(node))

                indent = "  " * level
                tri_str = f"({node.value[0]}, {node.value[1]}, {node.value[2]})"
                print(f"{indent}L{level}: Tri{tri_str} -> {len(node.children)} children")
                for child in node.children:
                    print_dag_test12(child, level + 1, visited, max_levels)

            print_dag_test12(root_node)

            print(f"\n  Modified graph after DAG construction:")
            print(f"    {custom_clone}")
            print(f"    Vertices after: {len(custom_clone.vertices)}")
        else:
            print("ERROR: BuildDAG returned None")
    else:
        print("ERROR: Graph is invalid, cannot build DAG")

except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("Test 12 completed!")
print("="*60)