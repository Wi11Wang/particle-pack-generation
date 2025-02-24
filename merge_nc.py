"""
This file is used to merge multiple .nc files into a single numpy memory map.

Workflow:
Iteratively place the array into memory map
"""
import argparse
import os
from preprocess import merge_nc, tomo_to_int16, mask_to_int16


def _parse_args():
    parser = argparse.ArgumentParser(description='A program merges multiple nc files')
    parser.add_argument('--in_dir', type=str,
                        default='../0_data/0_raw_data/mask',
                        metavar='path', help='path to the directory to merge nc files')
    parser.add_argument('--out_dir', type=str,
                        default='../0_data/0_raw_data/mask',
                        metavar='path', help='path to the directory to merge nc files')
    parser.add_argument('--key', type=str, choices=['tomo', 'labels'],
                        default='../0_data/0_raw_data/mask',
                        metavar='key', help='type of tomogram, either "tomo" or "labels"')
    parser.add_argument('--blocks', type=int,
                        default=0,
                        metavar='slices', help='max index of tomogram block')
    args = parser.parse_args()
    return args


def main():
    args = _parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.key == 'tomo':
        convert_fn = tomo_to_int16
    else:
        convert_fn = mask_to_int16
    merged_nc_shape = merge_nc(args.in_dir, args.out_dir, args.key, args.blocks, convert_fn)
    print(f'Merged tomogram has a shape of {merged_nc_shape} at {args.out_dir}')


if __name__ == '__main__':
    main()
