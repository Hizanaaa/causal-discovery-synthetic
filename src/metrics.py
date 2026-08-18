def skeleton_edges(edges):
    """
    Convert directed/undirected edges into unordered pairs.

    Example:
        A -> B becomes {A, B}
        A -- B becomes {A, B}
    """
    return {frozenset(edge) for edge in edges}


def edge_precision_recall_f1(true_edges, predicted_edges):
    """
    Calculate precision, recall, and F1 on the graph skeleton.

    Parameters
    ----------
    true_edges : iterable
        True graph edges represented as (source, target).

    predicted_edges : iterable
        Recovered graph edges represented as (source, target).

    Returns
    -------
    precision : float
    recall : float
    f1 : float
    """

    true_skeleton = skeleton_edges(true_edges)
    predicted_skeleton = skeleton_edges(predicted_edges)

    true_positive = len(true_skeleton & predicted_skeleton)

    false_positive = len(predicted_skeleton - true_skeleton)
    false_negative = len(true_skeleton - predicted_skeleton)

    if true_positive + false_positive == 0:
        precision = 0.0
    else:
        precision = true_positive / (true_positive + false_positive)

    if true_positive + false_negative == 0:
        recall = 0.0
    else:
        recall = true_positive / (true_positive + false_negative)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def structural_hamming_distance(true_edges, predicted_edges):
    """
    Calculate Structural Hamming Distance (SHD).

    A missing edge costs 1.
    An extra edge costs 1.
    A reversed edge costs 1.

    Parameters
    ----------
    true_edges : iterable
        True directed edges as (source, target).

    predicted_edges : iterable
        Predicted directed edges as (source, target).

    Returns
    -------
    int
        Structural Hamming Distance.
    """

    true_edges = set(true_edges)
    predicted_edges = set(predicted_edges)

    shd = 0

    # Correctly oriented edges
    correct = true_edges & predicted_edges

    # Remove correct edges so only errors remain.
    remaining_true = true_edges - correct
    remaining_predicted = predicted_edges - correct

    # Check for reversed edges.
    reversed_edges = set()

    for source, target in remaining_true:
        if (target, source) in remaining_predicted:
            reversed_edges.add((source, target))

    # Each reversal costs one operation.
    shd += len(reversed_edges)

    # Remove reversed edges from both sets.
    for source, target in reversed_edges:
        remaining_true.discard((source, target))
        remaining_predicted.discard((target, source))

    # Remaining edges are deletions or insertions.
    shd += len(remaining_true)
    shd += len(remaining_predicted)

    return shd

def causal_learn_edges(graph):
    """
    Convert a causal-learn GeneralGraph into a simple edge representation.

    Returns
    -------
    list
        List of tuples:
        (source, target, edge_type)

    edge_type can be:
        "directed"
        "undirected"
        "bidirected"
        "partially_oriented"
    """

    edges = []

    for edge in graph.get_graph_edges():
        node1 = edge.get_node1().get_name()
        node2 = edge.get_node2().get_name()

        endpoint1 = str(edge.get_endpoint1())
        endpoint2 = str(edge.get_endpoint2())

        if endpoint1 == "TAIL" and endpoint2 == "ARROW":
            edge_type = "directed"
            source = node1
            target = node2

        elif endpoint1 == "ARROW" and endpoint2 == "TAIL":
            edge_type = "directed"
            source = node2
            target = node1

        elif endpoint1 == "TAIL" and endpoint2 == "TAIL":
            edge_type = "undirected"
            source, target = sorted([node1, node2])

        elif endpoint1 == "ARROW" and endpoint2 == "ARROW":
            edge_type = "bidirected"
            source = node1
            target = node2

        else:
            edge_type = "partially_oriented"
            source = node1
            target = node2

        edges.append((source, target, edge_type))

    return edges

def causal_learn_skeleton(graph):
    """
    Extract the undirected skeleton from a causal-learn graph.

    Direction is intentionally ignored.

    Returns
    -------
    set
        Unordered node pairs representing adjacencies.
    """

    skeleton = set()

    for source, target, _ in causal_learn_edges(graph):
        skeleton.add(frozenset((source, target)))

    return skeleton

def skeleton_precision_recall_f1(true_skeleton, predicted_skeleton):
    """
    Calculate precision, recall, and F1 for graph skeletons.

    Parameters
    ----------
    true_skeleton : iterable
        Unordered node pairs representing true adjacencies.

    predicted_skeleton : iterable
        Unordered node pairs representing predicted adjacencies.

    Returns
    -------
    precision : float
    recall : float
    f1 : float
    """

    true_skeleton = {frozenset(edge) for edge in true_skeleton}
    predicted_skeleton = {frozenset(edge) for edge in predicted_skeleton}

    true_positive = len(true_skeleton & predicted_skeleton)
    false_positive = len(predicted_skeleton - true_skeleton)
    false_negative = len(true_skeleton - predicted_skeleton)

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive > 0
        else 0.0
    )

    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return precision, recall, f1

def orientation_accuracy(true_edges, predicted_edges):
    """
    Calculate orientation accuracy for correctly recovered adjacencies.

    Undirected predicted edges are not counted as correctly oriented.

    Parameters
    ----------
    true_edges : iterable
        True directed edges as (source, target).

    predicted_edges : iterable
        Predicted edges as (source, target, edge_type).

    Returns
    -------
    float
        Fraction of correctly oriented edges among true adjacencies
        that were recovered by the predicted graph.
    """

    true_edges = set(true_edges)

    predicted_edges = list(predicted_edges)

    correct_skeleton_edges = 0
    correctly_oriented = 0

    for source, target, edge_type in predicted_edges:
        predicted_pair = frozenset((source, target))

        matching_true_edges = [
            edge for edge in true_edges
            if frozenset(edge) == predicted_pair
        ]

        if not matching_true_edges:
            continue

        correct_skeleton_edges += 1

        if edge_type == "directed":
            true_source, true_target = matching_true_edges[0]

            if source == true_source and target == true_target:
                correctly_oriented += 1

    if correct_skeleton_edges == 0:
        return 0.0

    return correctly_oriented / correct_skeleton_edges

def dag_to_cpdag(true_edges, node_names):
    """
    Convert a known DAG into its CPDAG.

    Parameters
    ----------
    true_edges : iterable of (source, target)
        Directed edges of the true DAG.

    node_names : list[str]
        Names of all nodes.

    Returns
    -------
    cg : CausalGraph
        CPDAG represented using causal-learn's graph conventions.
    """

    from causallearn.graph.GraphClass import CausalGraph
    from causallearn.graph.Edge import Edge
    from causallearn.graph.Endpoint import Endpoint

    cg = CausalGraph(
        len(node_names),
        node_names=node_names,
    )

    node_map = {
        name: i
        for i, name in enumerate(node_names)
    }

    # Start with the complete undirected graph created by CausalGraph.
    # Remove edges that are not present in the true DAG skeleton.
    true_edges = set(true_edges)

    skeleton = {
        frozenset((source, target))
        for source, target in true_edges
    }

    for i in range(len(node_names)):
        for j in range(i + 1, len(node_names)):

            if frozenset((node_names[i], node_names[j])) not in skeleton:
                edge = cg.G.get_edge(
                    cg.G.nodes[i],
                    cg.G.nodes[j],
                )

                if edge is not None:
                    cg.G.remove_edge(edge)

    # Identify and orient unshielded colliders.
    #
    # X -> Y <- Z is an unshielded collider when:
    #   X and Z are both parents of Y
    #   X and Z are not adjacent.
    parents = {
        node: []
        for node in node_names
    }

    for source, target in true_edges:
        parents[target].append(source)

    for middle in node_names:

        middle_parents = parents[middle]

        for i in range(len(middle_parents)):
            for j in range(i + 1, len(middle_parents)):

                left = middle_parents[i]
                right = middle_parents[j]

                if frozenset((left, right)) in skeleton:
                    continue

                # Orient left -> middle.
                edge = cg.G.get_edge(
                    cg.G.nodes[node_map[left]],
                    cg.G.nodes[node_map[middle]],
                )

                if edge is not None:
                    cg.G.remove_edge(edge)

                cg.G.add_edge(
                    Edge(
                        cg.G.nodes[node_map[left]],
                        cg.G.nodes[node_map[middle]],
                        Endpoint.TAIL,
                        Endpoint.ARROW,
                    )
                )

                # Orient right -> middle.
                edge = cg.G.get_edge(
                    cg.G.nodes[node_map[right]],
                    cg.G.nodes[node_map[middle]],
                )

                if edge is not None:
                    cg.G.remove_edge(edge)

                cg.G.add_edge(
                    Edge(
                        cg.G.nodes[node_map[right]],
                        cg.G.nodes[node_map[middle]],
                        Endpoint.TAIL,
                        Endpoint.ARROW,
                    )
                )

    # Apply Meek's orientation rules.
    from causallearn.search.ConstraintBased.PC import Meek

    cg = Meek.meek(cg)

    return cg

def cpdag_structural_hamming_distance(graph1, graph2):
    """
    Compute Structural Hamming Distance (SHD) between two CPDAGs.

    An error is counted when an edge is:
    - missing
    - extra
    - oriented in the wrong direction

    Undirected edges are treated as distinct from directed edges.

    Parameters
    ----------
    graph1 : causal-learn GeneralGraph
        First CPDAG.

    graph2 : causal-learn GeneralGraph
        Second CPDAG.

    Returns
    -------
    int
        Structural Hamming Distance.
    """

    nodes1 = graph1.get_nodes()
    nodes2 = graph2.get_nodes()

    if len(nodes1) != len(nodes2):
        raise ValueError("Graphs must contain the same number of nodes.")

    node_names1 = [node.get_name() for node in nodes1]
    node_names2 = [node.get_name() for node in nodes2]

    if set(node_names1) != set(node_names2):
        raise ValueError("Graphs must contain the same node names.")

    node_map1 = {
        node.get_name(): node
        for node in nodes1
    }

    node_map2 = {
        node.get_name(): node
        for node in nodes2
    }

    def edge_type(graph, node_a, node_b):
        edge = graph.get_edge(node_a, node_b)

        if edge is None:
            return "none"

        endpoint1 = edge.get_endpoint1()
        endpoint2 = edge.get_endpoint2()

        from causallearn.graph.Endpoint import Endpoint

        if endpoint1 == Endpoint.TAIL and endpoint2 == Endpoint.TAIL:
            return "undirected"

        if endpoint1 == Endpoint.TAIL and endpoint2 == Endpoint.ARROW:
            return "directed"

        if endpoint1 == Endpoint.ARROW and endpoint2 == Endpoint.TAIL:
            return "reverse_directed"

        if endpoint1 == Endpoint.ARROW and endpoint2 == Endpoint.ARROW:
            return "bidirected"

        return "other"

    distance = 0

    node_names = sorted(set(node_names1))

    for i in range(len(node_names)):
        for j in range(i + 1, len(node_names)):

            name_a = node_names[i]
            name_b = node_names[j]

            type1 = edge_type(
                graph1,
                node_map1[name_a],
                node_map1[name_b],
            )

            type2 = edge_type(
                graph2,
                node_map2[name_a],
                node_map2[name_b],
            )

            if type1 != type2:
                distance += 1

    return distance