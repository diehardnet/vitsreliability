#! /usr/bin/env python3

import os
import sys
import pandas as pd
import argparse
import configs
import enum

# LOW_CONFIDENCE = True
# HIGH_CONFIDENCE = False

class InputType(enum.Enum):
    LOW_CONFIDENCE = 1
    HIGH_CONFIDENCE = 2
    FAULTY_SWFI = 3

    def __str__(self):
        return self.name
    
    def __repr__(self):
        return self.name

def parse_args():
    parser = argparse.ArgumentParser(description='Generate input list for the training')
    parser.add_argument('--model', type=str, help='Model to use for the training', choices=[configs.VIT_BASE_PATCH16_224, configs.SWIN_BASE_PATCH4_WINDOW7_224], default=configs.VIT_BASE_PATCH16_224)
    parser.add_argument('--num-elem', type=int, help='Number of wanted elements in the dataset', default=32)
    parser.add_argument('--dataset', type=str, help='Dataset', choices=[configs.IMAGENET], default=configs.IMAGENET)
    parser.add_argument('--precision', type=str, help='Precision', choices=[configs.FP32, configs.FP16], default=configs.FP32)
    parser.add_argument('--input-type', type=lambda intype: InputType[intype], help='Type of input to generate', choices=list(InputType), default=str(InputType.LOW_CONFIDENCE))
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()

    model = args.model
    num_elem = args.num_elem
    dataset = args.dataset
    precision = args.precision
    input_type = args.input_type

    confidences = [
        InputType.LOW_CONFIDENCE, 
        InputType.HIGH_CONFIDENCE,
    ]

    
    if input_type in confidences:
        input_file = f"input_generation/{model}_{dataset}_{precision}_top5prob_FULL.csv"

        df = pd.read_csv(input_file)
        df = df[
            (df['model'] == model) &
            (df['dataset'] == dataset) &
            (df['precision'] == precision)
        ]

        if input_type == InputType.LOW_CONFIDENCE:
            confidence = True
        else:
            confidence = False

        print("="*50)
        df = df.sort_values(by='top_diff', ascending=confidence, ignore_index=False)
        # print(df["top_diff"][:num_elem])

        output_file = f"data/input_images_{model}_{dataset}_{precision}_{'low' if input_type == InputType.LOW_CONFIDENCE else 'high'}2.txt"

        indices = ""
        count = 0
        total_inputs = 0
        for i, row in df.iterrows():
            count += 1
            if row['ground_truth'] != row['pred']:
                continue

            total_inputs += 1
            
            print(f"Image: {i}, Ground Truth: {row['ground_truth']}, Prediction: {row['pred']}, Top Diff: {row['top_diff']}")
            indices += f"{i}\n"
            if total_inputs == num_elem:
                break

        with open(output_file, 'w') as f:
            f.write(indices)
    elif input_type == InputType.FAULTY_SWFI:
        input_file = f"input_generation/faulty_swfi_indices.csv"

        df = pd.read_csv(input_file)
        df = df[
            df['model'] == model
        ]

        output_file = f"data/input_images_{model}_{dataset}_{precision}_faulty_swfi.txt"
        indices = ""
        total_inputs = 0
        for i, row in df.iterrows():
            indices += f"{row["image_id_in_full_imagenet"]}\n"
            total_inputs += 1
            if total_inputs == num_elem:
                break

        with open(output_file, 'w') as f:
            f.write(indices)
    else:
        raise ValueError(f"Input type {input_type} not supported.")

if __name__ == "__main__":
    main()
