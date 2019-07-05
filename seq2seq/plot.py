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
    scores = [29.5180, 35.1436 , 36.3166, 36.7998, 38.1695, 38.1610, 38.8785, 39.3129, 39.0157, 39.2539]
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
    print(f"Loss: {losses}\nBLEU: {scores}\nSpec: {spec}")
    # train loss
    fig = plt.figure(figsize=(9,5.5))
    ax = fig.add_axes((0.1, 0.2, 0.8, 0.7))
    ax.plot(losses)
    ax.set_title('Train Loss')
    ax.set_ylabel('Loss')
    ax.set_xlabel('# epochs')
    # ax.legend()
    fig.text(.5, .05, spec, ha='center')
    plt.show()
    # plt.tight_layout()
    fig.savefig(f'plots/{save_path}_train.{file_type}', format=file_type)
    print(f'plots/{save_path}_train.{file_type} saved')
    # test bleu
    fig2 = plt.figure(figsize=(9,5.5))
    ax2 = fig2.add_axes((0.1, 0.2, 0.8, 0.7))
    ax2.plot(scores)
    ax2.set_title('Validation BLEU')
    ax2.set_ylabel('BLEU')
    ax2.set_xlabel('# epochs')
    # ax2.legend()
    fig2.text(.5, .05, spec, ha='center')
    plt.show()
    # plt.tight_layout()
    fig2.savefig(f'plots/{save_path}_eval.{file_type}', format=file_type)
    print(f'plots/{save_path}_eval.{file_type} saved')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-load-path', help="ckpt path", type=str)
    parser.add_argument('-save-path', help="save model path", type=str, default="")
    parser.add_argument('-file-type', help="png, pdf, svg, ...", type=str, default="png")
    args = parser.parse_args()
    load_path = args.load_path
    save_path = args.save_path
    file_type = args.file_type
    plot_and_save(load_path, save_path, file_type)
