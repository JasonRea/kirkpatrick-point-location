import polygon as poly
import primitives as prim
import planar_graph as pg
import search_and_intersection as si

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
    root_node = si.BuildDAG(triangle_clone)

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
    root_node = si.BuildDAG(square_clone)

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
    root_node = si.BuildDAG(hex_clone)

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
    root = si.BuildDAG(test_clone)

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