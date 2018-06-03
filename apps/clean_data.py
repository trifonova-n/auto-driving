from pathlib import Path
import argparse


def clean_train(path):
    path = Path(path)
    list_dir = path / 'list_train'
    video_dir = path / 'train_color'
    mask_dir = path / 'train_label'
    list_files = sorted(list_dir.glob('*.txt'))
    for i, file_path in enumerate(list_files):
        filtered_content = []
        file_modified = False
        with file_path.open() as f:
            for line in f:
                sline = line.strip()
                if not sline:
                    continue
                fields = sline.split('\t')
                img_file = fields[0].split('\\')[-1]
                mask_file = fields[1].split('\\')[-1]
                if (video_dir / img_file).exists() and (mask_dir / mask_file).exists():
                    filtered_content.append(line)
                else:
                    file_modified = True
        if file_modified:
            if not filtered_content:
                print('Remove list file', file_path)
                file_path.unlink()
            else:
                print('Modified link file', file_path)
                with file_path.open('w') as f:
                    f.writelines(filtered_content)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='Data directory')
    args = parser.parse_args()
    clean_train(args.path)
