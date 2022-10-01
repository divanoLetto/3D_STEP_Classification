import time
from pathlib import Path
from ..Graphh.Node import FlatNode
from ..Graphh.Node_utils import get_nodes_from_datas
from .Parser import parse_file
from .parser_utils import check_just_conteiner
from ..Graphh.utils import replace_nodes
import networkx as nx
import os


def all_nodes_to_graph(G, all_nodes, name, graph_saves_paths):
    """
    Generate (write in G) a Direct Graph with linked nodes. Each node is characterized by numeric/not numeric parameters.
    Args:
        G: empty graph.
        all_nodes: list of FlatNodes. Each FlatNode has a ID, Type and a list parameters those include numeric
                   attributes, not numeric attributes and neighbour nodes.
        name: name of the graph
        graph_saves_paths: directory where to save the graph.
    Returns: None
    """
    start_time = time.time()
    for flat_node in all_nodes:
        # G.add_node(flat_node)
        dict_protery = FlatNode.get_dict_parameters(flat_node)
        G.add_nodes_from([(flat_node.id, dict_protery)])
    for flat_node in all_nodes:
        numeric_paramenters = []
        for par in flat_node.parameters:
            if isinstance(par, FlatNode):
                G.add_edge(flat_node.id, par.id)
            else:
                numeric_paramenters.append(par)
        flat_node.parameters = numeric_paramenters

    print("   Graphh of " + name + " model realizing time: %s seconds" % (time.time() - start_time))
    nx.write_graphml(G, graph_saves_paths)


def make_graph_simplex_direct(file_name, graph_saves_base_paths, dataset_path):
    '''
    Args:
        file_name: name file .step, es: ABC.stp
        graph_saves_base_paths: path where to save .graphml file
        dataset_path: path where to find file_name .stp file
    '''
    name = os.path.splitext(file_name)[0]
    graph_saves_paths = graph_saves_base_paths + name + '.graphml'
    graph_path = Path(graph_saves_paths)

    if graph_path.exists():
        print("Loading", file_name, "graph")
        G_simplex_d = nx.read_graphml(graph_saves_paths)
        return G_simplex_d
    else:
        print("Making", file_name, "graph")
        headers, datas = parse_file(file_name, dataset_path=dataset_path)
        print("file " + file_name + " parsed")
        all_flat_nodes, fast_dict_search = get_nodes_from_datas(datas)
        print("   All nodes obtained")
        replace_nodes(all_flat_nodes, fast_dict_search)
        print("   All edges obtained")
        G_simplex_d = nx.DiGraph()
        all_nodes_to_graph(G_simplex_d, all_flat_nodes, name, graph_saves_paths)

        return G_simplex_d


def spit_graph_in_parts(graph):
    partitions = {}
    shape_rep_found = False
    count = 0
    count_occ = 0
    for node in graph:
        if graph.nodes[node]["type"] == "SHAPE_REPRESENTATION":
            shape_rep_found = True
            name = graph.nodes[node]["SHAPE_REPRESENTATION_0"]
            list_of_nodes = [node]
            last_layer_list = [node]
            while len(last_layer_list) != 0:
                tmp_list = []
                for n in last_layer_list:
                    inner_neighbor = graph.predecessors(n)
                    tmp_list.extend(inner_neighbor)
                last_layer_list = tmp_list
                list_of_nodes.extend(last_layer_list)
                list_of_nodes = list(set(list_of_nodes))
            last_len = -1
            last_layer_list = list_of_nodes
            while len(list_of_nodes) != last_len:
                last_len = len(list_of_nodes)
                tmp_list = []
                for n in last_layer_list:
                    inner_neighbor = graph.successors(n)
                    tmp_list.extend(inner_neighbor)
                last_layer_list = tmp_list
                list_of_nodes.extend(last_layer_list)
                list_of_nodes = list(set(list_of_nodes))

            new_partition_graph = graph.subgraph(list_of_nodes)

            # check if is just the conteiner of all the parts:
            if not check_just_conteiner(new_partition_graph, name):

                number_of_occurance = 0
                for multiple_occ in new_partition_graph:
                    if new_partition_graph.nodes[multiple_occ]["type"] == "NEXT_ASSEMBLY_USAGE_OCCURRENCE":
                        number_of_occurance += 1
                if number_of_occurance <= 1:
                    if name in partitions.keys():
                        name = name + "_" + str(count)
                        count += 1
                    partitions[name] = new_partition_graph
                else:
                    for i in range(number_of_occurance):
                        count_occ += 1
                        partitions[name + "_occ_" + str(count_occ)] = new_partition_graph

    if shape_rep_found:
        return partitions
    else:
        for node in graph:
            if graph.nodes[node]["type"] == "PRODUCT":
                name = graph.nodes[node]["PRODUCT_0"]
                list_of_nodes = [node]
                last_layer_list = [node]
                while len(last_layer_list) != 0:
                    tmp_list = []
                    for n in last_layer_list:
                        inner_neighbor = graph.predecessors(n)
                        tmp_list.extend(inner_neighbor)
                    last_layer_list = tmp_list
                    list_of_nodes.extend(last_layer_list)
                    list_of_nodes = list(set(list_of_nodes))
                last_len = -1
                last_layer_list = list_of_nodes
                while len(list_of_nodes) != last_len:
                    last_len = len(list_of_nodes)
                    tmp_list = []
                    for n in last_layer_list:
                        inner_neighbor = graph.successors(n)
                        tmp_list.extend(inner_neighbor)
                    last_layer_list = tmp_list
                    list_of_nodes.extend(last_layer_list)
                    list_of_nodes = list(set(list_of_nodes))

                new_partition_graph = graph.subgraph(list_of_nodes)
                partitions[name] = new_partition_graph

        return partitions





