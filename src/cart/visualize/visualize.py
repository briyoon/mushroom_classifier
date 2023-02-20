from igraph import Graph, plot

def create_visualization(root, save_path):
    level = dict()
    level[root] = 0
    q = [root]
    seen = []
    depth = 0
    edges = []
    ids = dict()
    index = 0
    ids[root] = index
    index += 1

    nodes = [root]

    while q:
        curr = q.pop(0)
        for branch_val in curr.children:
            child = curr.children[branch_val]
            if child not in ids:
                ids[child] = index
                index += 1
            edges.append((ids[curr], ids[child]))
            if child not in seen:
                level[child] = level[curr] + 1
                seen.append(child)
                q.append(child)
                nodes.append(child)

    depth = max(level.values())
    print(f'depth={depth}')

    # nr_vertices = len(seen)
    # v_label = list(map(str, range(nr_vertices)))
    G = Graph(edges)  # 2 stands for children number

    def color_setter(node):
        if node.is_leaf:
            if node.attr_name == "p":
                return "red"
            return "green"
        return "blue"

    scale = 5

    style = {}
    style["vertex_size"] = [25 * scale for _ in nodes]
    style["vertex_label"] = [str(node) for node in nodes]
    style["vertex_label_size"] = 15 * scale
    style["vertex_label_dist"] = [0 if node.is_leaf else scale / 3 for node in nodes]
    style["vertex_color"] = [color_setter(node) for node in nodes]

    style["edge_label"] = [str(nodes[edge[1]].value) for edge in edges]
    style["edge_label_size"] = 15 * scale

    layout = G.layout(layout='tree', root=[0])
    plot(G, layout=layout, bbox=[6000, 6000], margin=20 * scale, **style, target=save_path)