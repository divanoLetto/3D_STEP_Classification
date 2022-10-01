import os
from .Parser.Make_graph import make_graph_simplex_direct


def make_graphh_dataset(path_stp, path_graph):
    if not path_stp.endswith("/"):
        path_stp = path_stp + "/"
    if not path_graph.endswith("/"):
        path_graph = path_graph + "/"

    if not os.path.exists(path_graph):
        os.makedirs(path_graph)
    # we take all step files and their class label (from the directory they are in)
    for dir in os.listdir(path_stp):
        for file in os.listdir(path_stp + dir + "/"):
            if file.endswith(".stp") or file.endswith(".step") or file.endswith(".STEP"):

                if not os.path.exists(path_graph + dir + "/"):
                    os.makedirs(path_graph + dir + "/")

                g = make_graph_simplex_direct(file_name=dir + "/" + file,graph_saves_base_paths=path_graph,dataset_path=path_stp)


if __name__ == "__main__":
    path_stp = "path1"
    path_graph = "path2"

    make_graphh_dataset(path_stp, path_graph)
