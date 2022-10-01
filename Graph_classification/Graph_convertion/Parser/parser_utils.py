def check_just_conteiner(graph, name):
    count = 0
    for node in graph:
        if graph.nodes[node]["type"] == "NEXT_ASSEMBLY_USAGE_OCCURRENCE":
            if name != graph.nodes[node]["NEXT_ASSEMBLY_USAGE_OCCURRENCE_0"]:
                return True
    return False