from nmt import *
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-load-path', help="ckpt path", type=str)
    parser.add_argument('-save-path', help="save model path", type=str)
    parser.add_argument('-file-type', choices=["pdf", "img"])
    args = parser.parse_args()

    
