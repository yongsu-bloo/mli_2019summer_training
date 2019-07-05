import torch
from argparse import ArgumentParser
import matplotlib.pyplot as plt
def plot_and_save(load_path, save_path, file_type, checkpoint=None):
    # data load
    checkpoint = torch.load(load_path) if checkpoint is None else checkpoint
    epoch = checkpoint['epoch']
    losses = checkpoint['losses']
    scores = checkpoint['scores']
    load_args = checkpoint['args']

    spec = {"NUM_LAYERS": load_args.num_layers,
            "EMD_DIM": load_args.emd_dim,
            "HIDDEN_DIM": load_args.hidden_dim,
            "Dropout": load_args.dropout,
            "Optimizer": load_args.opt
            }
    if load_args.bidirectional:
        spec["Bidirectional"] = ""
    if load_args.no_reverse:
        spec["Not reversed"] = ""
    spec = ", ".join([ "{}: {}".format(k, spec[k]) if spec[k] != "" else str(k) for k in spec])
    # train loss
    fig, ax = plt.subplots(figsize=(9,5.5))
    ax.plot(losses)
    ax.set_title('Train Loss')
    ax.set_xlabel('Loss')
    ax.set_ylabel('# epochs')
    ax.legend()
    fig.text(.5, .05, spec, ha='center')
    plt.show()
    plt.tight_layout()
    fig.savefig('{save_path}_train.{file_type}', format=file_type)
    # test bleu
    fig2, ax2 = plt.subplots(figsize=(9,5.5))
    ax2.plot(scores)
    ax2.set_title('Validation BLEU')
    ax2.set_xlabel('BLEU')
    ax2.set_ylabel('# epochs')
    ax2.legend()
    fig2.text(.5, .05, spec, ha='center')
    plt.show()
    plt.tight_layout()
    fig2.savefig('{save_path}_eval.{file_type}', format=file_type)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-load-path', help="ckpt path", type=str)
    parser.add_argument('-save-path', help="save model path", type=str, default="./plots")
    parser.add_argument('-file-type', help="png, pdf, svg, ...", type=str, default="png")
    args = parser.parse_args()
    load_path = load_path
    save_path = save_path
    file_type = file_type
    plot_and_save(load_path, save_path, file_type)
