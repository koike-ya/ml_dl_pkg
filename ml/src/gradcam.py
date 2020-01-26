import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from ml.models.pretrained_models import construct_pretrained
from ml.models.panns_cnn14 import construct_panns
from ml.src.signal_processor import istft


class GradCAM:
    def __init__(self, model, feature_layer):
        self.model = model
        self.feature_layer = feature_layer
        self.model.eval()
        self.feature_grad = None
        self.feature_map = None
        self.hooks = []

        # 最終層逆伝播時の勾配を記録する
        def save_feature_grad(module, in_grad, out_grad):
            self.feature_grad = out_grad[0]
        self.hooks.append(self.feature_layer.register_backward_hook(save_feature_grad))

        # 最終層の出力 Feature Map を記録する
        def save_feature_map(module, inp, outp):
            self.feature_map = outp[0]
        self.hooks.append(self.feature_layer.register_forward_hook(save_feature_map))

    def forward(self, x):
        return self.model(x)

    def backward_on_target(self, output, target):
        self.model.zero_grad()
        one_hot_output = torch.zeros([1, output.size()[-1]])
        one_hot_output[0][target] = 1
        output.backward(gradient=one_hot_output, retain_graph=True)

    def clear_hook(self):
        for hook in self.hooks:
            hook.remove()


def visualize_panns(model, cnn_last_layer, input_, wave, i):
    org_img = model.logmel_extractor(model.spectrogram_extractor(input_))[0][0]
    input_orig_size = org_img.transpose(0, 1).size()
    VISUALIZE_SIZE = input_orig_size
    # print(input_orig_size)

    grad_cam = GradCAM(model=model, feature_layer=cnn_last_layer)
    model_output = grad_cam.forward(input_)
    # print(model_output)
    target = model_output.argmax(1).item()
    print(target)
    grad_cam.backward_on_target(model_output, target)

    feature_grad = grad_cam.feature_grad.data.numpy()[0]
    # Get weights from gradient
    weights = np.mean(feature_grad, axis=(1, 2))  # Take averages for each gradient
    # Get features outputs
    feature_map = grad_cam.feature_map.data.numpy()
    grad_cam.clear_hook()

    # Get cam
    cam = np.sum((weights * feature_map.T), axis=2).T
    cam = np.maximum(cam, 0)  # apply ReLU to cam

    cam = cv2.resize(cam, VISUALIZE_SIZE)
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
    cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize

    activation_heatmap = cam
    # org_img = np.asarray(input_[0].resize(VISUALIZE_SIZE))
    # print(input_[0].size())

    # print(org_img.size())
    img_with_heatmap = np.multiply(np.float32(activation_heatmap), np.float32(org_img))
    img_with_heatmap = img_with_heatmap / np.max(img_with_heatmap)
    # org_img = cv2.resize(org_img, input_orig_size)
    _ = cv2.resize(np.uint8(255 * img_with_heatmap), input_orig_size)
    # exit()

    plt.figure(figsize=(20, 10))
    plt.subplot(3, 1, 1)
    pd.Series(wave).plot(figsize=(17, 10))
    plt.subplot(3, 1, 2)
    plt.imshow(org_img.transpose(0, 1))
    plt.subplot(3, 1, 3)
    # plt.imshow(cv2.resize(np.uint8(255 * img_with_heatmap), input_orig_size))
    plt.imshow(np.uint8(255 * img_with_heatmap).T)
    # plt.show()
    plt.savefig(f'../visualize/gradcam/panns_{i}.png')


def visualize_2d(model, cnn_last_layer, input_, wave, i):
    VISUALIZE_SIZE = (64, 401)  # 可視化する時に使うサイズ。PyTorch ResNet の Pre-Train モデルのデフォルト入力サイズを使います
    input_orig_size = input_[0][0].size()

    grad_cam = GradCAM(model=model, feature_layer=cnn_last_layer)
    model_output = grad_cam.forward(input_)
    # print(model_output)
    target = model_output.argmax(1).item()
    print(target)
    grad_cam.backward_on_target(model_output, target)

    feature_grad = grad_cam.feature_grad.data.numpy()[0]
    # Get weights from gradient
    weights = np.mean(feature_grad, axis=(1, 2))  # Take averages for each gradient
    # Get features outputs
    feature_map = grad_cam.feature_map.data.numpy()
    grad_cam.clear_hook()

    # Get cam
    cam = np.sum((weights * feature_map.T), axis=2).T
    cam = np.maximum(cam, 0)  # apply ReLU to cam

    cam = cv2.resize(cam, VISUALIZE_SIZE)
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
    cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize

    activation_heatmap = np.expand_dims(cam, axis=0).transpose(1, 2, 0)
    # org_img = np.asarray(input_[0].resize(VISUALIZE_SIZE))
    # print(input_[0].size())
    org_img = input_[0].transpose(0, 2).transpose(0, 1)
    # print(org_img.size())
    img_with_heatmap = np.multiply(np.float32(activation_heatmap), np.float32(org_img))
    img_with_heatmap = img_with_heatmap / np.max(img_with_heatmap)
    # org_img = cv2.resize(org_img, input_orig_size)

    plt.figure(figsize=(20, 10))
    plt.subplot(3, 1, 1)
    pd.Series(wave).plot(figsize=(17, 10))
    plt.subplot(3, 1, 2)
    plt.imshow(org_img.transpose(0, 1)[:, :, 0])
    plt.subplot(3, 1, 3)
    # plt.imshow(cv2.resize(np.uint8(255 * img_with_heatmap), input_orig_size))
    plt.imshow(np.uint8(255 * img_with_heatmap).transpose(1, 0, 2)[:, :, 0])
    # plt.show()
    plt.savefig(f'../visualize/gradcam/resnet_{i}.png')


def gradcam_main(cfg, dataloader, load_func):
    Path('../visualize/gradcam/').mkdir(exist_ok=True)

    if cfg['model_type'] == 'panns':
        model = construct_panns(cfg, infer=True)
    else:
        model = construct_pretrained(cfg, n_classes=2)
        model.load_state_dict(torch.load(cfg['checkpoint_path'], map_location=torch.device('cuda')))
    # model.to(torch.device('cuda'))
    print(f'Ans\t\tPred')
    for i in range(10):
        # i = len(dataloader.dataset) - i - 1
        wave = load_func(dataloader.dataset.path_df.iloc[i, :]).reshape((-1,))
        input_, label = dataloader.dataset[i]
        print(label, end='\t\t')

        sr = 2000
        win_length = int(sr * cfg['window_size'])
        hop_length = int(sr * cfg['window_stride'])

        # _ = istft(input_[0].numpy().T, win_length, hop_length, window='hamming')
        # pd.Series(wave).plot()
        # plt.show()
        # pd.Series(_).plot()
        # plt.show()
        # exit()

        if cfg['model_type'] == 'panns':
            cnn_last_layer = list(model.conv_block6.children())[-1]
            visualize_panns(model, cnn_last_layer, input_.unsqueeze(dim=0), wave, i)
        else:
            cnn_last_layer = list(model.feature_extractor.children())[-2]
            visualize_2d(model, cnn_last_layer, input_.unsqueeze(dim=0), wave, i)
