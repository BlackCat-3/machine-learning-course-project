"""Regenerate submission from the saved full-data checkpoint without retraining."""
import argparse
import json
import torch
from pipeline import ROOT, UNet, export


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default='baseline_v1')
    args = parser.parse_args()
    out = ROOT/'outputs'/args.run
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(out/'full_train/final.pt',map_location=device,weights_only=False)
    model = UNet().to(device)
    model.load_state_dict(checkpoint['model'])
    selection = json.loads((out/'selection.json').read_text(encoding='utf-8'))
    export(model,selection['threshold'],out,device)
    print(out/'submission.csv')


if __name__ == '__main__':
    main()
