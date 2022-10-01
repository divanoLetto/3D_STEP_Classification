from ..Parser.parser_utils import check_just_conteiner
import numpy as np


class Graphh:
    def __init__(self, graph, name, ext=None, search_child=True, graph_class=None):
        self.full_graph = graph
        self.name = name
        self.extention = ext
        self.parts = []
        self.parts_occurences = []
        if search_child:
            self.split_graph_in_parts()
        labels = set()
        for node in self.full_graph:
            labels.add(self.full_graph.nodes[node]["type"])
        self.labels = labels
        self.graph_class = graph_class

    def get_class(self):
        return self.graph_class

    def get_name(self):
        return self.name

    def get_labels(self):
        return self.labels

    def get_name_of_part(self, idx):
        return self.parts[idx].get_name()

    def get_parts_occurences(self):
        return self.parts_occurences

    def get_tot_num_parts_occurences(self):
        return sum(self.parts_occurences)

    def get_parts_graphs(self):
        return self.parts

    def get_num_parts(self):
        return len(self.parts)

    def number_of_nodes(self):
        return self.full_graph.number_of_nodes()

    def number_of_edges(self):
        return self.full_graph.number_of_edges()

    def get_full_graph(self):
        return self.full_graph

    def add_part(self, part_graph, name, number_occurance):
        if number_occurance < 1:
            number_occurance = 1

        part_graphh = Graphh(part_graph, name, search_child=False)
        self.parts.append(part_graphh)
        self.parts_occurences.append(number_occurance)

    def print_composition(self):
        print(self.name + " has: " + str(self.get_full_graph().number_of_nodes()) + " nodes, " + str(self.get_full_graph().number_of_edges()) + " edges, " + str(self.get_num_parts()) + " components:")
        for i in range(len(self.parts)):
            print("   " + (self.parts[i].get_name() + " has: " + str(self.parts[i].get_full_graph().number_of_nodes()) + " nodes, " + str(self.parts_occurences[i]) + " occurance"))

    def get_ingoing_then_outgoing_nodes_from_start_node(self, node):
        list_of_nodes = [node]
        last_layer_list = [node]
        while len(last_layer_list) != 0:
            tmp_list = []
            for n in last_layer_list:
                inner_neighbor = self.full_graph.predecessors(n)
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
                inner_neighbor = self.full_graph.successors(n)
                tmp_list.extend(inner_neighbor)
            last_layer_list = tmp_list
            list_of_nodes.extend(last_layer_list)
            list_of_nodes = list(set(list_of_nodes))
        return list_of_nodes

    def split_graph_in_parts(self):
        # if basic model
        design_type = "PRODUCT"
        for node in self.full_graph:
            # if complex model
            if self.full_graph.nodes[node]["type"] == "SHAPE_REPRESENTATION":
                design_type = "SHAPE_REPRESENTATION"

        design_type_name = design_type + "_0"
        # list of all names of subgraphs found
        names_found = []
        count = 0
        for node in self.full_graph:
            if self.full_graph.nodes[node]["type"] == design_type:
                shape_rep_found = True
                if design_type_name in self.full_graph.nodes[node].keys():
                    name = self.full_graph.nodes[node][design_type_name]
                else:
                    # A volte definiscno un prodotto con stringa vuota e questa viene filtrata
                    name = ""

                graph_name = name
                if name in names_found:
                    graph_name = graph_name + "_" + str(count)
                    count += 1
                names_found.append(graph_name)

                list_of_nodes = self.get_ingoing_then_outgoing_nodes_from_start_node(node)
                new_partition_graph = self.full_graph.subgraph(list_of_nodes)

                # check if is just the conteiner of all the parts:
                if not check_just_conteiner(new_partition_graph, name):
                    number_of_occurance = 0
                    for node_occ in new_partition_graph:
                        if new_partition_graph.nodes[node_occ]["type"] == "NEXT_ASSEMBLY_USAGE_OCCURRENCE":
                            number_of_occurance += 1
                    self.add_part(new_partition_graph, graph_name, number_of_occurance)
                    # self.add_part(new_partition_graph, name, number_of_occurance)
                else:
                    new_nodes = []
                    for node in self.full_graph:
                        if self.full_graph.nodes[node]["type"] == "PROPERTY_DEFINITION_REPRESENTATION":
                            new_nodes.append(node)
                    last_layer_list = new_nodes
                    while len(last_layer_list) != 0:
                        tmp_list = []
                        for n in last_layer_list:
                            inner_neighbor = self.full_graph.successors(n)
                            tmp_list.extend(inner_neighbor)
                        last_layer_list = tmp_list
                        new_nodes.extend(last_layer_list)
                        new_nodes = list(set(new_nodes))
                    list_of_nodes.extend(new_nodes)
                    self.parts_conteiner = self.full_graph.subgraph(list_of_nodes)


    @staticmethod
    def get_occ_match_matrix(match_matrix, gh_1, gh_2):
        total_parts_match_matrix = np.copy(match_matrix)
        for i, num_occ_i in enumerate(gh_1.get_parts_occurences()):
            for _ in range(1, num_occ_i):
                # add row
                row = match_matrix[i]
                total_parts_match_matrix = np.append(total_parts_match_matrix, [row], 0)
        for j, num_occ_j in enumerate(gh_2.get_parts_occurences()):
            for _ in range(1, num_occ_j):
                # add row
                col = total_parts_match_matrix[:, j]
                total_parts_match_matrix = np.append(total_parts_match_matrix, [[c] for c in col], 1)
        return total_parts_match_matrix


