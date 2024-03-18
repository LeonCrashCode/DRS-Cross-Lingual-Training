import argparse
from ud_boxer.sbn import SBNGraph


def getset(filename, num_group):
    cnt = 0
    lines = []
    for line in open(filename):
        line = line.strip()
        if cnt % num_group == 0:
            lines.append([line])
        else:
            lines[-1].append(line)
        cnt += 1
    return lines

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_set', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--num_group', type=int, default=10)
    return parser

parser = get_parser()
args = parser.parse_args()


input_sets = getset(args.input_set, args.num_group)

with open(args.output, "w") as f:
    finals = []
    cnt = 0
    for input_set in input_sets:
        notfound = True
        for one in input_set:
            try:
                penman_str = SBNGraph().from_string(one.replace("?"," ?").replace("***","\n")).to_penman_string()
                finals.append(one)
                notfound = False
                break
            except:
                pass
        if notfound:
            finals.append(input_set[0])
    f.write("\n".join(finals)+"\n")
    f.close()     
