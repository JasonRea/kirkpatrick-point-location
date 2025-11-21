import polygon as poly
import primitives as prim

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