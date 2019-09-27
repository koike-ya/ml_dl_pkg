"""
in: 学習済みモデルとちゃんとtarget用のアノテーションされたデータ
out: よりdomain不変な特徴量を作成できるようになった学習済みモデル
"""

"""
Implements ADDA:
Adversarial Discriminative Domain Adaptation, Tzeng et al. (2017)
"""

from copy import deepcopy
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm, trange

from ml.utils import train_args
from ml.utils.utils import set_eeg_conf, init_device, init_seed, concat_manifests, set_model


def adda_args(parser):
    adda_parser = parser.add_argument_group("ADDA options")

    adda_parser.add_argument('--source-manifests', type=str, help='manifest files to use as source',
                             default='input/test_manifest.csv,input/test_manifest.csv')
    adda_parser.add_argument('--target-manifests', type=str, help='manifest files to use as target',
                             default='input/test_manifest.csv,input/test_manifest.csv')
    adda_parser.add_argument('--iterations', type=int, default=500)
    adda_parser.add_argument('--adda-epochs', type=int, default=5)
    adda_parser.add_argument('--k-disc', type=int, default=5)
    adda_parser.add_argument('--k-clf', type=int, default=10)
    return parser


def loop_iterable(iterable):
    while True:
        yield from iterable


def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad


def adda(args, source_model, eeg_conf, label_func, class_names, target_criterion, device,
         source_manifest, target_manifest):
    target_model = deepcopy(source_model)
    source_model.load_state_dict(torch.load(args.model_path))
    source_model.eval()
    set_requires_grad(source_model, requires_grad=False)

    clf = source_model
    source_model = source_model.features

    target_model.load_state_dict(torch.load(args.model_path))
    in_features = target_model.classifier[0].in_features
    target_classifier = target_model.classifier
    target_classifier.eval()
    target_model = target_model.features

    discriminator = nn.Sequential(
        nn.Linear(in_features, 400),
        nn.ReLU(),
        nn.Linear(400, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    ).to(device)

    half_batch = args.batch_size // 2

    source_dataset = EEGDataSet(source_manifest, eeg_conf, label_func, class_names, args.to_1d, device=device)
    source_loader = EEGDataLoader(source_dataset, batch_size=half_batch, num_workers=args.num_workers,
                                      pin_memory=True, drop_last=True)

    target_dataset = EEGDataSet(target_manifest, eeg_conf, label_func, class_names, args.to_1d, device=device)
    target_loader = EEGDataLoader(target_dataset, batch_size=half_batch, num_workers=args.num_workers,
                                      pin_memory=True, drop_last=True)

    discriminator_optim = torch.optim.Adam(discriminator.parameters())
    target_optim = torch.optim.Adam(target_model.parameters())
    disc_criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, args.adda_epochs + 1):
        batch_iterator = zip(loop_iterable(source_loader), loop_iterable(target_loader))

        disc_loss = 0
        disc_acc = 0
        target_loss = 0
        target_acc = 0
        for _ in trange(args.iterations, leave=False):
            # Train discriminator
            set_requires_grad(target_model, requires_grad=False)
            set_requires_grad(discriminator, requires_grad=True)
            for _ in range(args.k_disc):
                (source_x, _), (target_x, _) = next(batch_iterator)
                source_x, target_x = source_x.to(device), target_x.to(device)

                source_features = source_model(source_x).view(source_x.shape[0], -1)
                target_features = target_model(target_x).view(target_x.shape[0], -1)

                discriminator_x = torch.cat([source_features, target_features])
                discriminator_y = torch.cat([torch.ones(source_x.shape[0], device=device),
                                             torch.zeros(target_x.shape[0], device=device)])

                preds = discriminator(discriminator_x).squeeze()
                disc_loss = disc_criterion(preds, discriminator_y)

                discriminator_optim.zero_grad()
                disc_loss.backward()
                discriminator_optim.step()

                disc_loss += disc_loss.item()
                disc_acc += ((preds > 0).long() == discriminator_y.long()).float().mean().item()

            # Train classifier
            set_requires_grad(target_model, requires_grad=True)
            set_requires_grad(discriminator, requires_grad=False)
            for _ in range(args.k_clf):
                _, (target_x, target_y) = next(batch_iterator)
                target_x, target_y = target_x.to(device), target_y.to(device)
                target_features = target_model(target_x).view(target_x.shape[0], -1)

                # flipped labels
                discriminator_y = torch.ones(target_x.shape[0], device=device)

                preds = discriminator(target_features).squeeze()
                disc_loss = disc_criterion(preds, discriminator_y)

                target_optim.zero_grad()
                disc_loss.backward()
                target_optim.step()

                x = target_model(target_x)
                x = x.view(x.size(0), -1)
                x = target_classifier(x)
                target_preds = nn.Softmax(dim=-1)(x)
                target_loss += target_criterion(target_preds, target_y).item()
                target_acc += ((torch.max(target_preds, 1)[1] > 0.5).long() == target_y.long()).float().mean().item()

        mean_target_loss = target_loss / (args.iterations * args.k_clf)
        mean_target_acc = target_acc / (args.iterations * args.k_clf)
        mean_disc_loss = disc_loss / (args.iterations * args.k_disc)
        mean_disc_acc = disc_acc / (args.iterations * args.k_disc)
        tqdm.write(f'EPOCH {epoch:03d}: disc_loss: {mean_disc_loss:.4f}\t '
                   f'disc_acc={mean_disc_acc:.4f}\t target_loss={mean_target_loss:.4f}\t '
                   f'target_acc={mean_target_acc:.4f}')

        # Create the full target model and save it
        clf.feature_extractor = target_model
        torch.save(clf.state_dict(), Path(args.model_path).parent / 'adda.pt')


def main(args, class_names, label_func, metrics):
    init_seed(args)
    Path(args.model_path).parent.mkdir(exist_ok=True, parents=True)

    # init setting
    classes = [i for i in range(len(class_names))]
    device = init_device(args)
    eeg_conf = set_eeg_conf(args)
    model = set_model(args, classes, eeg_conf, device)
    args.weight = list(map(float, args.loss_weight.split('-')))
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(args.weight).to(device))

    source_manifest = concat_manifests(args.source_manifests.split(','), 'source')
    target_manifest = concat_manifests(args.target_manifests.split(','), 'target')

    adda(args, model, eeg_conf, label_func, class_names, criterion, device,
         source_manifest, target_manifest)


if __name__ == '__main__':
    args = add_adda_args(train_args()).parse_args()
    class_names = ['interictal', 'preictal']
    from ml.src import Metric

    metrics = [Metric('loss', initial_value=1000, inequality='less', save_model=True),
               # Metric('recall'),
               # Metric('far')
               ]

    def label_func(path):
        return path.split('/')[-2].split('_')[2]

    main(args, class_names, label_func, metrics)
