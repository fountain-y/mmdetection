from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
import os.path as osp
import argparse
import os
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet inference a model')
    parser.add_argument(
        '--image-path', help='path of images to be tested')
    parser.add_argument(
        '--result-path', help='path of results to be saved')
    parser.add_argument(
        '--config-file', help='config file to be used')
    parser.add_argument(
        '--ckpt-path', help='checkpoint file to be used')
    parser.add_argument(
        '--score-thr', default=0.3, help='threshold score to show the result')
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    
    image_path = args.image_path
    config_file = args.config_file
    checkpoint_file = args.ckpt_path
    result_path = args.result_path
    score_thr = args.score_thr
    mmcv.mkdir_or_exist(result_path)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = init_detector(config_file, checkpoint_file, device=device)
    names = []
    for root, dirs, files in os.walk(image_path):
        for name in files:
            names.append(name)

    print('tot : {}'.format(len(names)))
    
    for name in names:
        img = osp.join(image_path, name)
        file_name = name.split('.')[0]
        result = inference_detector(model, img)
        if hasattr(model, 'module'):
            model = model.module
        img = model.show_result(img, result, score_thr=score_thr, show=False)
        mmcv.imwrite(img, osp.join(result_path, file_name + '.png'))
        print(f'{file_name} finished.')



if __name__ == '__main__':
    main()


