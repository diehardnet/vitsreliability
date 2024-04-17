#!/usr/bin/python3

import torch
import os
from setuppuretorch import load_dataset, parse_args, check_and_setup_gpu, Timer, load_model
import console_logger
import configs


def main():
    timer = Timer()
    args, args_text_list = parse_args()
    # Define DNN goal
    dnn_goal = configs.DNN_GOAL[args.model]
    dataset = configs.DATASETS[dnn_goal]
    # Load the model
    model, transform = load_model(model_name=args.model)
    # Terminal console
    main_logger_name = str(os.path.basename(__file__)).replace(".py", "")
    terminal_logger = console_logger.ColoredLogger(main_logger_name)
    terminal_logger.debug("\n".join(args_text_list))

    terminal_logger.debug("Loading dataset from torch")
    file_path = "../data/test.pt"
    # First step is to load the inputs in the memory
    timer.tic()
    input_list, input_labels = load_dataset(batch_size=args.batchsize, dataset=dataset,
                                            test_sample=args.testsamples,
                                            transform=transform)
    timer.toc()
    input_load_time = timer.diff_time_str

    # save on the file
    input_data = [input_list, input_labels, transform]
    torch.save(input_data, file_path)
    del input_data
    terminal_logger.debug("Loading dataset from file")
    # Calculate to load again
    timer.tic()
    new_input_data = torch.load(file_path)
    timer.toc()
    input_load_time_from_file = timer.diff_time_str
    print(new_input_data[0][0].shape)
    terminal_logger.debug(
        f"Time to load from dataset {input_load_time}, time to load from file {input_load_time_from_file}")


if __name__ == '__main__':
    main()
