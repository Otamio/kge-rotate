import sys

if __name__ == "__main__":

    dataset_path = sys.argv[1]
    entities = set()
    relations = set()

    with open(f"{dataset_path}/train.txt") as fd:
        for line in fd:
            s, r, o = [x.strip() for x in line.split()]
            entities.add(s)
            relations.add(r)
            entities.add(o)

    with open(f"{dataset_path}/entities.dict", 'w') as fd:
        for i, entity in enumerate(entities):
            fd.write(f"{i}\t{entity}\n")

    with open(f"{dataset_path}/relations.dict", 'w') as fd:
        for i, relation in enumerate(relations):
            fd.write(f"{i}\t{relation}\n")
