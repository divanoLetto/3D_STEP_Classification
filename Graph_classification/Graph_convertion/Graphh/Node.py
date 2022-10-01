class Node:
    def __init__(self, id, type):
        self.id = id
        self.type = type


class GenericNode(Node):
    def __init__(self, id, type, parameters):
        super().__init__(id, type)
        self.parameters = parameters


class FlatNode(Node):
    def __init__(self, id, type, parameters):
        super().__init__(id, type)
        self.id = id.replace(" ", "")
        self.type = type.replace(" ", "")
        self.parameters = self.get_flat_parameters(parameters)

    def get_flat_parameters(self, parameters):
        flat_parameters = []
        if isinstance(parameters, list):
            for par in parameters:
                if isinstance(par, list):
                    for nestet_par in par:
                        new_list = self.get_flat_parameters(nestet_par)
                        flat_parameters = flat_parameters + new_list
                else:
                    flat_parameters.append(par)
        else:
            flat_parameters.append(parameters)
        return flat_parameters

    @staticmethod
    def get_dict_parameters(node):
        dict_property = {"type": node.type}
        for j, par in enumerate(node.parameters):
            if not isinstance(par, FlatNode):
                prob_name = node.type + "_" + str(j)
                dict_property[prob_name] = par
        return dict_property

    @staticmethod
    def node_simarity(n1, n2):
        if n1.type != n2.type:
            return 0
        else:
            diff = 0
            for i, par_i in enumerate(n1.parameters):
                a = print("TODO")
