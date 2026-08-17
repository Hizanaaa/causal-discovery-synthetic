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
            source = node1
            target = node2

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

def true_cpdag_edges(true_edges):
    """
    Return the CPDAG representation for a small DAG.

    For the current sanity-check graphs:
        A -> B -> C  ->  A -- B -- C
        A <- B -> C  ->  A -- B -- C
        A -> B <- C  ->  A -> B <- C

    This is intentionally limited to the three primitive DAGs.
    We will replace this with a general DAG-to-CPDAG implementation
    before the medium DAG experiments.
    """

    true_edges = set(true_edges)

    chain = {
        ("X1", "X2"),
        ("X2", "X3"),
    }

    collider = {
        ("X1", "X2"),
        ("X3", "X2"),
    }

    if true_edges == chain:
        return {
            frozenset(("X1", "X2")),
            frozenset(("X2", "X3")),
        }, set()

    if true_edges == collider:
        return {
            frozenset(("X1", "X2")),
            frozenset(("X2", "X3")),
        }, {
            ("X1", "X2"),
            ("X3", "X2"),
        }

    raise ValueError("Unsupported DAG for primitive CPDAG conversion.")