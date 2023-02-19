from src.cart import AttrNode


def tree_depth(root: AttrNode):
    level = dict()
    level[root] = 0
    q = [root]
    seen = []
    while q:
        curr = q.pop(0)
        for branch_val in curr.children:
            child = curr.children[branch_val]
            if child not in seen:
                level[child] = level[curr] + 1
                seen.append(child)
                q.append(child)

    return max(level.values())


def avg_tree_depth(root: AttrNode):
    level = dict()
    level[root] = 0
    q = [root]
    seen = []
    leaf_cnt = 0
    depth_sum = 0
    while q:
        curr = q.pop(0)
        for branch_val in curr.children:
            child = curr.children[branch_val]
            if child.is_leaf:
                leaf_cnt += 1
                depth_sum += level[curr] + 1
            if child not in seen:
                level[child] = level[curr] + 1
                seen.append(child)
                q.append(child)

    return depth_sum / leaf_cnt
