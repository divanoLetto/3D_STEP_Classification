from .Node import GenericNode, FlatNode
from .utils import split_recursive, split_composed_arguments, remove_first_last_space


def get_nodes_type_hystogramm(datas):
    nodes_types = {}
    for line in datas:
        if line == 'ENDSEC':
            break
        type = line.split("=")
        type = type[1]
        type = type.split("(")[0]
        if type not in nodes_types:
            nodes_types[type] = 1
        else:
            nodes_types[type] += 1
    return nodes_types


def get_num_neighbor_for_node_type(graph):
    nodes_types = {}
    for node_id in graph.nodes():
        node = graph.nodes()[node_id]
        type = node["type"]
        if type not in nodes_types.keys():
            nodes_types[type] = []
        num_neighbor = len([g for g in graph.neighbors(node_id)])
        nodes_types[type].append(num_neighbor)
    return nodes_types


def get_composed_node_types(graph):
    nodes_types = {}
    for node_id in graph.nodes():
        node = graph.nodes()[node_id]
        if node_id[0] == "#" and node_id[1] == "#":
            type = node["type"]
            if type not in nodes_types.keys():
                nodes_types[type] = 0
            nodes_types[type] += 1
    return nodes_types


def get_nodes_from_datas(datas):
    """
    Generate nodes and archs from a list of lines that compose a .stl file. Each node is characterized by a type and a list of arguments.
    Args:
        datas: list of lines that compose the .stl file.
    Returns:
        all_flat_nodes: list of FlatNode generated from data.
    """
    fast_dict_search = {}
    all_flat_nodes = []
    composed_id = 0
    for line in datas:
        if line == 'ENDSEC':
            break
        id_type_arguments = line.split("=")
        id = id_type_arguments[0]
        id = remove_first_last_space(id)

        type_arguments = id_type_arguments[1]
        type_arguments = remove_first_last_space(type_arguments)

        if type_arguments[0] != '(':  # Normal elements
            type_arguments = type_arguments.split("(", 1)
            type = type_arguments[0]
            arguments = type_arguments[1]
            arguments, _ = split_recursive(arguments, 0)
        else:  # Composed elements
            multiple_obj = type_arguments[1:][:-1]
            multiple_obj = split_composed_arguments(multiple_obj)
            arguments = []
            for i, m in enumerate(multiple_obj):
                m_type_arguments = m.split('(', 1)
                m_type = m_type_arguments[0]
                m_arguments = m_type_arguments[1]
                m_arguments, _ = split_recursive(m_arguments, 0)
                m_id = '##' + str(composed_id)  # TODO fa casino??? --> no
                composed_id += 1
                # node = GenericNode(m_id, m_type, m_arguments)
                flat_node = FlatNode(m_id, m_type, m_arguments)
                # all_nodes.append(node)
                all_flat_nodes.append(flat_node)
                fast_dict_search[m_id] = flat_node
                # Composed object properties
                if i == 0:
                    type = 'COMPOSED_' + m_type
                arguments.append(m_id)

        # node = GenericNode(id, type, arguments)
        flat_node = FlatNode(id, type, arguments)
        fast_dict_search[id] = flat_node
        # all_nodes.append(node)
        all_flat_nodes.append(flat_node)

    print("   Number of composed id: " + str(composed_id))
    return all_flat_nodes, fast_dict_search


def get_scomposed_nodes_from_datas(datas):
    # Get all element from file. stp
    all_flat_nodes = []
    composed_id = 0
    for line in datas:
        if line == 'ENDSEC':
            break
        id_type_arguments = line.split("=")
        id = id_type_arguments[0]
        id = remove_first_last_space(id)

        type_arguments = id_type_arguments[1]
        type_arguments = remove_first_last_space(type_arguments)

        if type_arguments[0] != '(':  # Normal elements
            type_arguments = type_arguments.split("(", 1)
            type = type_arguments[0]
            arguments = type_arguments[1]
            arguments, _ = split_recursive(arguments, 0)
        else:  # Composed elements
            multiple_obj = type_arguments[1:][:-1]
            multiple_obj = split_composed_arguments(multiple_obj)
            arguments = []
            for i, m in enumerate(multiple_obj):
                m_type_arguments = m.split('(', 1)
                m_type = m_type_arguments[0]
                m_arguments = m_type_arguments[1]
                m_arguments, _ = split_recursive(m_arguments, 0)
                m_id = '##' + str(composed_id)  # TODO fa casino??? --> no
                composed_id += 1
                node = GenericNode(m_id, m_type, m_arguments)
                flat_node = FlatNode(m_id, m_type, m_arguments)
                # all_nodes.append(node)
                all_flat_nodes.append(flat_node)
                # Composed object properties
                if i == 0:
                    type = 'COMPOSED_' + m_type
                arguments.append(m_id)

        node = GenericNode(id, type, arguments)
        flat_node = FlatNode(id, type, arguments)
        # all_nodes.append(node)
        all_flat_nodes.append(flat_node)

    print("   Number of composed id: " + str(composed_id))
    return all_flat_nodes # all_nodes,


def flat_nodes_match(node1, node2):
    for i, par in enumerate(node1.parameters):
        if par != node2.parameters[i]:
            return False
    return True


def get_all_neighbor_nodes(graph, node):
    list_of_nodes = [node]
    for neighbor in graph.neighbors(node):
        list_of_nodes.extend(get_all_neighbor_nodes(graph, neighbor))
    return list_of_nodes

