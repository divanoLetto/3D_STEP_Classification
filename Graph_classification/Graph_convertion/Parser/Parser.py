def parse_file(file_name, dataset_path):
    file_name = dataset_path + file_name
    with open(file_name) as f:
        data = f.read()
        data = data.replace("\n","")
        data = data.replace("\'",'')
        datas = data.split(";")
    headers = []
    start_data = 0
    for i, line in enumerate(datas):
        if line != 'DATA':
            headers.append(line)
        else:
            start_data = i + 1
            break
    for i in range(0, start_data):
        datas.pop(0)

    return headers, datas


def parse_header(file_name):
    file_name = "C:/Users/Computer/PycharmProjects/graphStepSimilarity/Datasets/" + file_name
    with open(file_name) as f:
        data = f.read()
        datas = data.split("\n")
    headers = []
    for i, line in enumerate(datas):
        if line != 'DATA;':
            headers.append(line)
        else:
            break
    return headers
