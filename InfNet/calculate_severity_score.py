import os
import csv
import cv2
import numpy as np
from PIL import Image
import torch
import argparse
from shutil import copyfile
import torchvision.transforms as transforms
import torch.nn.functional as F
import imageio
from sklearn.metrics import f1_score, precision_score, recall_score


from InfNet.Code.model_lung_infection.InfNet_ResNet import Inf_Net
from InfNet.Code.model_lung_infection.InfNet_UNet import Inf_Net_UNet

os.environ['KMP_DUPLICATE_LIB_OK']='True'

score_to_severe = {0: 'Regular', 1: 'Severe', 2: 'Critically ill'}
severe_to_score = {'Regular': 0, 'Control': 0, 'Mild': 0, 'Severe': 1, 'Critically ill': 2}


def create_imgs_ictcf(ictcf_input_dir, input_dir, ictcf_output_dir):
    parenchymas = os.listdir(input_dir)
    for parenchyma in parenchymas:
        if 'Patient' not in parenchyma:
            continue

        patient_filename = parenchyma.split('.')[0]
        patient_info = patient_filename.split("_")
        if len(patient_info) == 1:
            patient_name = patient_info[0]
            patient_img_index = 0
        else:
            patient_name = patient_info[0]
            patient_img_index = patient_info[1]

        patient_dir = os.path.join(ictcf_input_dir, patient_name)
        patient_img = os.path.join(patient_dir, f'{patient_img_index}.jpg')
        copyfile(patient_img, os.path.join(ictcf_output_dir, f'{patient_filename}.jpg'))


def calculate_severity(input_dir, parenchyma_input_dir, save_segment_path, save_binary_segment_path, severity_dict, model,
                       save_multi_segment_path, save_binary_multi_segment_path, multi_model, device):
    os.makedirs(save_segment_path, exist_ok=True)
    os.makedirs(save_binary_segment_path, exist_ok=True)
    os.makedirs(save_multi_segment_path, exist_ok=True)
    os.makedirs(save_binary_multi_segment_path, exist_ok=True)

    predictions = []
    ground_truths = []

    input_images = sorted(os.listdir(input_dir))
    parenchyma_images = sorted(os.listdir(parenchyma_input_dir))

    transform = transforms.Compose([
        transforms.Resize((352, 352)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])

    multi_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])

    for index, input_image in enumerate(input_images):
        if 'Patient' not in input_image:
            continue

        # get ground truth score
        # patient_name = input_image.split('.')[0].split('_')[0]
        # gronud_truth_severity = severity_dict[patient_name]
        # ground_truth_score = severe_to_score.get(gronud_truth_severity, None)
        # if ground_truth_score is None:
        #     continue
        #
        input_image_filename = os.path.join(input_dir, input_image)
        img = Image.open(input_image_filename)
        img = img.convert('RGB')
        image = transform(img).unsqueeze(0).to(device)

        lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2, lateral_edge = model(image)
        res = lateral_map_2
        res = F.upsample(res, size=(352, 352), mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res_numerator = res - res.min()
        res_denominator = res.max() - res.min() + 1e-8
        prediction = res_numerator / res_denominator

        imageio.imwrite(os.path.join(save_segment_path, input_image.replace('.jpg', '.png')), prediction)
        cv2.imwrite(os.path.join(save_binary_segment_path, input_image.replace('.jpg', '.png')), prediction)

        # multi segmentation
        multi_input_image = cv2.imread(input_image_filename)
        multi_input_image = cv2.resize(multi_input_image, (352, 352))
        pseudo_image_filename = os.path.join(save_segment_path, input_image.replace('.jpg', 'png'))
        pseudo = cv2.imread(pseudo_image_filename)
        pseudo = cv2.resize(pseudo, (352, 352))

        multi_input_image = multi_transform(multi_input_image).unsqueeze(0).to(device)
        pseudo = multi_transform(pseudo).unsqueeze(0).to(device)

        output = multi_model(torch.cat((multi_input_image, pseudo), dim=1))
        output = torch.sigmoid(output)  # output.shape is torch.Size([4, 2, 160, 160])
        b, _, w, h = output.size()
        # output b*n_class*h*w -- > b*h*w
        pred = output.cpu().permute(0, 2, 3, 1).contiguous().view(-1, 3).max(1)[1].view(b, w, h).numpy().squeeze()
        pred_rgb = (np.arange(3) == pred[..., None]).astype(float)
        # swap the rgb content so the background is black instead of red
        pred = np.zeros(pred_rgb.shape)
        pred[:, :, 0] = pred_rgb[:, :, 1]
        pred[:, :, 1] = pred_rgb[:, :, 2]

        imageio.imwrite(os.path.join(save_multi_segment_path, input_image), pred)
        cv2.imwrite(os.path.join(save_binary_multi_segment_path, input_image), pred)

    #     parenchyma_image = parenchyma_images[index]
    #     parenchyma_filename = os.path.join(parenchyma_input_dir, parenchyma_image)
    #     parenchyma_img = Image.open(parenchyma_filename)
    #     np_parenchyma_img = np.array(parenchyma_img)
    #     np_parenchyma_img[np_parenchyma_img > 0] = 1
    #
    #     severity_score = 0
    #     ratio = prediction.sum() / np_parenchyma_img.sum()
    #     if ratio < 0.01:
    #         severity_score = 0
    #     elif 0.01 < ratio < 0.5:
    #         severity_score = 1
    #     elif 0.5 < ratio < 1:
    #         severity_score = 2
    #
    #     predictions.append(severity_score)
    #     ground_truths.append(ground_truth_score)
    #
    # f1 = f1_score(ground_truths, predictions, average='micro')
    # precision = precision_score(ground_truths, predictions, average='micro')
    # recall = recall_score(ground_truths, predictions, average='micro')
    #
    # print(f'f1 score: {f1}')
    # print(f'precision score: {precision}')
    # print(f'recall score: {recall}')


def process_csv_to_get_severity(data_filename):
    severity_dict = {}
    with open(data_filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for index, row in enumerate(csv_reader):
            if index == 0:
                continue
            else:
                severity_dict[row[0]] = row[5]
    return severity_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--parenchyma_input_dir', type=str, required=True)
    parser.add_argument('--csv_severity_file', type=str)
    parser.add_argument('--save_segment_path', type=str)
    parser.add_argument('--save_binary_segment_path', type=str)

    parser.add_argument('--ictcf_input_dir', type=str)
    parser.add_argument('--ictcf_output_dir', type=str)
    parser.add_argument('--load_net_path', type=str)
    parser.add_argument('--net_channel', type=int, default=32)
    parser.add_argument('--n_classes', type=int, default=1)
    parser.add_argument('--device', default='cpu')

    # multi
    parser.add_argument('--save_multi_segment_path', type=str)
    parser.add_argument('--save_binary_multi_segment_path', type=str)
    parser.add_argument('--load_multi_net_path', type=str)

    args = parser.parse_args()
    # os.makedirs(args.ictcf_output_dir, exist_ok=True)

    model = Inf_Net(channel=args.net_channel, n_class=args.n_classes).to(args.device)
    multi_model = Inf_Net_UNet(n_channels=6, n_classes=3).to(args.device)
    if args.load_net_path:
        print('loading weights')
        net_state_dict = torch.load(args.load_net_path, map_location=torch.device(args.device))
        net_state_dict = {k: v for k, v in net_state_dict.items() if k in model.state_dict()}
        model.load_state_dict(net_state_dict)

    if args.load_multi_net_path:
        print('loading multi weights')
        net_state_dict = torch.load(args.load_multi_net_path, map_location=torch.device(args.device))
        net_state_dict = {k: v for k, v in net_state_dict.items() if k in multi_model.state_dict()}
        multi_model.load_state_dict(net_state_dict)

    severity_dict = process_csv_to_get_severity(args.csv_severity_file)
    calculate_severity(args.input_dir, args.parenchyma_input_dir, args.save_segment_path, args.save_binary_segment_path,
                       severity_dict, model,
                       save_multi_segment_path=args.save_multi_segment_path,
                       save_binary_multi_segment_path=args.save_binary_multi_segment_path,
                       multi_model=multi_model, device=args.device)
    # create_imgs_ictcf(args.ictcf_input_dir, args.input_dir, args.ictcf_output_dir)