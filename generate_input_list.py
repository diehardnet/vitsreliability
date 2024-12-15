#! /usr/bin/env python3

import os
import sys
import pandas as pd
import argparse
import configs

LOW_CONFIDENCE = True
HIGH_CONFIDENCE = False


def parse_args():
    parser = argparse.ArgumentParser(description='Generate input list for the training')
    parser.add_argument('--model', type=str, help='Model to use for the training', choices=[configs.VIT_BASE_PATCH16_224, configs.SWIN_BASE_PATCH4_WINDOW7_224], default=configs.VIT_BASE_PATCH16_224)
    parser.add_argument('--num-elem', type=int, help='Number of wanted elements in the dataset', default=32)
    parser.add_argument('--dataset', type=str, help='Dataset', choices=[configs.IMAGENET], default=configs.IMAGENET)
    parser.add_argument('--precision', type=str, help='Precision', choices=[configs.FP32, configs.FP16], default=configs.FP32)
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()

    model = args.model
    num_elem = args.num_elem
    dataset = args.dataset
    precision = args.precision

    input_file = f"input_generation/{model}_{dataset}_{precision}_top5prob_FULL.csv"

    confidences = [
        LOW_CONFIDENCE, 
        HIGH_CONFIDENCE,
    ]

    df = pd.read_csv(input_file)
    df = df[
        (df['model'] == model) &
        (df['dataset'] == dataset) &
        (df['precision'] == precision)
    ]
    for confidence in confidences:
        print("="*50)
        df = df.sort_values(by='top_diff', ascending=confidence, ignore_index=False)
        # print(df["top_diff"][:num_elem])

        output_file = f"data/input_images_{model}_{dataset}_{precision}_{'low' if confidence == LOW_CONFIDENCE else 'high'}.txt"

        indices = ""
        count = 0
        total_inputs = 0
        for i, row in df.iterrows():
            count += 1
            if row['ground_truth'] != row['pred'] or  count < num_elem:
                continue

            total_inputs += 1
            
            print(f"Image: {i}, Ground Truth: {row['ground_truth']}, Prediction: {row['pred']}, Top Diff: {row['top_diff']}")
            indices += f"{i}\n"
            if total_inputs == num_elem:
                break

        with open(output_file, 'w') as f:
            f.write(indices)


if __name__ == "__main__":
    main()
