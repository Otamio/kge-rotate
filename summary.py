import glob
from collections import namedtuple

Result = namedtuple("Result", "mr mrr hits_1 hits_3 hits_10")


class EpocResult(namedtuple("EpocResult", "epoc validation test")):
    def __gt__(self, other):
        return self.validation.mrr > other.validation.mrr

    def __lt__(self, other):
        return self.validation.mrr < other.validation.mrr

    def __eq__(self, other):
        return self.validation.mrr == other.validation.mrr


def convert_result(dic):
    return Result(dic['mr'],
                  dic['mrr'],
                  dic['hits@1'],
                  dic['hits@3'],
                  dic['hits@10'])


def main():

    results = {}

    for fname in glob.iglob("out/*/train.log"):
        with open(fname) as fd:
            epoc = 0
            result = []
            valid, test = {}, {}
            for line in fd:
                line = line.strip().split("INFO")[1].strip()
                if line.startswith("Valid MRR at step"):
                    epoc = int(line.split(':')[0].split()[-1])
                if len(test) > 0 and line.startswith("Valid MRR at step"):
                    result.append(EpocResult(epoc,
                                             convert_result(valid),
                                             convert_result(test)))
                    valid, test = {}, {}
                if line.startswith("Valid HITS@10 at step"):
                    valid["hits@10"] = float(line.split(':')[1])
                elif line.startswith("Valid HITS@3 at step"):
                    valid["hits@3"] = float(line.split(':')[1])
                elif line.startswith("Valid HITS@1 at step"):
                    valid["hits@1"] = float(line.split(':')[1])
                elif line.startswith("Valid MR at step"):
                    valid["mr"] = float(line.split(':')[1])
                elif line.startswith("Valid MRR at step"):
                    valid["mrr"] = float(line.split(':')[1])
                elif line.startswith("Test HITS@10 at step"):
                    test["hits@10"] = float(line.split(':')[1])
                elif line.startswith("Test HITS@3 at step"):
                    test["hits@3"] = float(line.split(':')[1])
                elif line.startswith("Test HITS@1 at step"):
                    test["hits@1"] = float(line.split(':')[1])
                elif line.startswith("Test MR at step"):
                    test["mr"] = float(line.split(':')[1])
                elif line.startswith("Test MRR at step"):
                    test["mrr"] = float(line.split(':')[1])
            try:
                result.append(EpocResult(epoc, convert_result(valid),
                                         convert_result(test)))
            except KeyError:
                pass
        results[fname.split('/')[1].split('.')[0].strip()] = result

    for exp, res in sorted(results.items()):
        if len(res) == 0:
            continue
        best = max(res)
        print(exp)
        print('\t', best.epoc, '\t', best.test.mrr, '\t', best.test.hits_1, '\t', best.test.hits_10)


if __name__ == "__main__":
    main()
