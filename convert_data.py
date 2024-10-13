import os
import argparse
import zipfile
import shutil


ACTSIONS = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08',
            'A09', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16',
            'A17', 'A18', 'A19', 'A20', 'A21', 'A22', 'A23', 'A24',
            'A25', 'A26', 'A27']


def get_args():
    parser = argparse.ArgumentParser(description="Code implementation with MMFi dataset and library")
    parser.add_argument("--file", default='/data/szy4017/data/mmfi_all/S01.zip', type=str, help="file path")
    args = parser.parse_args()

    return args


def main():
    args = get_args()
    zip_file_path = args.file
    name = zip_file_path.split('/')[-1].split('.')[0]

    rgb_path = '/data/szy4017/data/mmfi_all/E01_rgb'
    target_path = '/data/szy4017/data/mmfi_all/E01'

    # 解压zip到目标路径
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(target_path)

    for a in ACTSIONS:
        target = os.path.join(target_path, name, a, 'rgb')
        source = os.path.join(rgb_path, name, a, 'rgb')

        shutil.rmtree(target)  # 先删除目标文件夹及其内容
        shutil.copytree(source, target)  # 将源文件夹复制到目标文件夹路径

        print(f'{source} replace {target}')


if __name__ == '__main__':
    main()