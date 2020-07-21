import os
import cv2
from PIL import Image
import torch
import argparse
from shutil import copyfile
import torchvision.transforms as transforms
import torch.nn.functional as F


from InfNet.Code.model_lung_infection.InfNet_ResNet import Inf_Net


def create_imgs_ictcf(ictcf_input_dir, input_dir, ictcf_output_dir):
    parenchymas = os.listdir(input_dir)
    for parenchyma in parenchymas:
        if 'Patient' not in parenchyma:
            continue

        patient = parenchyma.split('.')[0]
        patient_dir = os.path.join(ictcf_input_dir, patient)
        patient_img = os.path.join(patient_dir, '0.jpg')
        copyfile(patient_img, os.path.join(ictcf_output_dir, f'{patient}.jpg'))


def calculate_severity(input_dir, parenchyma_input_dir, model):
    input_images = sorted(os.listdir(input_dir))
    parenchyma_images = sorted(os.listdir(parenchyma_input_dir))

    transform = transforms.Compose([
        transforms.Resize((352, 352)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])
    for index, input_image in enumerate(input_images):
        input_image_filename = os.path.join(input_dir, input_image)
        img = Image.open(input_image_filename)
        img = img.convert('RGB')
        image = transform(img).unsqueeze(0)

        lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2, lateral_edge = model(image)
        res = lateral_map_2
        res = F.upsample(res, size=(352, 352), mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        # res_numerator = res - res.min()
        res_denominator = res.max() - res.min() + 1e-8
        # result = res_numerator / res_denominator

        parenchyma_image = parenchyma_images[index]
        parenchyma_filename = os.path.join()
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--parenchyma_input_dir', type=str, required=True)

    parser.add_argument('--ictcf_input_dir', type=str)
    parser.add_argument('--ictcf_output_dir', type=str)
    parser.add_argument('--load_net_path', type=str)
    parser.add_argument('--net_channel', type=int, default=32)
    parser.add_argument('--n_classes', type=int, default=1)
    parser.add_argument('--device', default='cpu')

    args = parser.parse_args()
    # os.makedirs(args.ictcf_output_dir, exist_ok=True)

    model = Inf_Net(channel=args.net_channel, n_class=args.n_classes).to(args.device)
    if args.load_net_path:
        net_state_dict = torch.load(args.load_net_path, map_location=torch.device(args.device))
        net_state_dict = {k: v for k, v in net_state_dict.items() if k in model.state_dict()}
        model.load_state_dict(net_state_dict)

    calculate_severity(args.input_dir, args.parenchyma_input_dir, model)
    # create_imgs_ictcf(args.ictcf_input_dir, args.input_dir, args.ictcf_output_dir)