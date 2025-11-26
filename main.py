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