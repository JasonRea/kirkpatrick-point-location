from manim import *

class TreeConstruction(MovingCameraScene):
    def construct(self):
        # Zoom out the camera to see the full tree
        self.camera.frame.scale(2)

        # Create 5 nodes for the tree
        # Node positions: root at top, 2 children in middle, 2 children at bottom
        node_positions = {
            0: UP * 2,                    # Root
            1: UP * 0.5 + LEFT * 2,      # Left child of root
            2: UP * 0.5 + RIGHT * 2,     # Right child of root
            3: DOWN * 1 + LEFT * 3,      # Left child of node 1
            4: DOWN * 1 + RIGHT * 1,     # Right child of node 1
        }

        # Create nodes (circles with labels)
        nodes = {}
        for i, pos in node_positions.items():
            circle = Circle(radius=0.3, color=BLUE, fill_opacity=0.7)
            circle.move_to(pos)
            label = Text(str(i), font_size=28, color=WHITE)
            label.move_to(pos)
            nodes[i] = VGroup(circle, label)

        # Define edges (parent -> child relationships)
        edges = [
            (0, 1),  # Root to left child
            (0, 2),  # Root to right child
            (1, 3),  # Node 1 to left child
            (1, 4),  # Node 1 to right child
        ]

        # Create edge lines
        edge_lines = []
        for parent, child in edges:
            line = Line(
                node_positions[parent],
                node_positions[child],
                color=GRAY,
                stroke_width=3
            )
            edge_lines.append(line)

        # Animation sequence
        # Step 1: Create root node
        self.play(Create(nodes[0]), run_time=1)
        self.wait(0.5)

        # Step 2: Add first level children (nodes 1 and 2)
        self.play(
            Create(edge_lines[0]),
            Create(edge_lines[1]),
            run_time=0.8
        )
        self.play(
            Create(nodes[1]),
            Create(nodes[2]),
            run_time=1
        )
        self.wait(0.5)

        # Step 3: Add second level children (nodes 3 and 4)
        self.play(
            Create(edge_lines[2]),
            Create(edge_lines[3]),
            run_time=0.8
        )
        self.play(
            Create(nodes[3]),
            Create(nodes[4]),
            run_time=1
        )
        self.wait(1)

        # Step 4: Highlight the tree structure
        all_nodes = VGroup(*[nodes[i] for i in range(5)])
        all_edges = VGroup(*edge_lines)

        self.play(
            all_nodes.animate.set_color(GREEN),
            all_edges.animate.set_color(YELLOW),
            run_time=1.5
        )
        self.wait(2)
