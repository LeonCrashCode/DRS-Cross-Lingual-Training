import logging
from argparse import ArgumentParser, Namespace

from tqdm.contrib.logging import logging_redirect_tqdm

from ud_boxer.sbn import SBNGraph

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def get_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument(
        "-i",
        "--input_path",
        type=str,
        required=True,
        help="Path to SBN file.",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default="./output.penman",
        help="Path to save penman output to.",
    )

    parser.add_argument(
        "--format",
        type=str,
        default="***")
        

    return parser.parse_args()


def main():
    args = get_args()

    penman_strs = []
    cnt = 0
    for line in open(args.input_path):
        line = line.strip()
        line = line.replace("?"," ?")
        if args.format == "***":
            line = line.replace("***","\n")
        elif args.format == "simple":
            pass
        try:
            penman_str = SBNGraph().from_string(line).to_penman_string()
            penman_strs.append(penman_str)
        except:
            penman_strs.append('(b0 / "alwaysfalse" )')
            cnt += 1 
        
    print("skips: {}".format(cnt))
    with open(args.output_path, "w") as f:
        for item in penman_strs:
            f.write(item)
            f.write("\n\n")
        f.close()

if __name__ == "__main__":
    with logging_redirect_tqdm():
        main()
