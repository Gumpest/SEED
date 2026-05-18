import argparse
import os

import torch
from seed.data_selection.utils import select_mwis_subset


def parse_args():
    argparser = argparse.ArgumentParser(
        description='Script for selecting the data for training')
    argparser.add_argument('--train_file_names', type=str,
                           nargs='+', help='The path to the score file')
    argparser.add_argument('--train_files', type=str, nargs='+',
                           help='The path of the training file that corresponds to the score file')
    argparser.add_argument('--target_task_names', type=str,
                           nargs='+', help='The name of the target task')
    argparser.add_argument('--output_path', type=str,
                           default="selected_data", help='The path to the output')
    argparser.add_argument('--max_samples', type=int,
                           default=None, help='The maximum number of samples')
    argparser.add_argument('--percentage', type=float, default=None,
                           help='The percentage of the data to be selected')

    args = argparser.parse_args()

    return args


def count_lines(filename):
    with open(filename, 'r', encoding='utf-8', errors='ignore') as file:
        line_count = 0
        for line in file:
            line_count += 1
    return line_count


if __name__ == "__main__":
    args = parse_args()
    assert len(args.train_file_names) == len(args.train_files)
    assert args.percentage is not None or args.max_samples is not None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_train_files = len(args.train_file_names)

    for target_task in args.target_task_names:
        output_path = os.path.join(args.output_path, target_task)

        score_paths = [os.path.join(
            output_path, f"{task_name}_influence_score.pt") for task_name in args.train_file_names]

        training_info_paths = [os.path.join(
            output_path, f"{task_name}_training_info.pt") for task_name in args.train_file_names]
        
        num_samples = []
        for score_path in score_paths:
            num_samples.append(
                len(torch.load(score_path, map_location=device)))
        cumsum_num_samples = torch.cumsum(torch.tensor(num_samples), dim=0)

        total_samples = sum(num_samples)
        if args.percentage is not None:
            args.max_samples = int(args.percentage * total_samples)
            data_amount_name = f"p{args.percentage}"
        else:
            data_amount_name = f"num{args.max_samples}"

        all_scores = []
        all_training_info = []
        for score_path, training_info_path in zip(score_paths, training_info_paths):
            score = torch.load(score_path, map_location=device)
            all_scores.append(score)
            training_info = torch.load(training_info_path, map_location=device)
            all_training_info.append(training_info)
        all_scores = torch.cat(all_scores, dim=0) # [total_samples]
        all_training_info = torch.cat(all_training_info, dim=0) 

        # sort the scores and output the corresponding data index
        file_specific_index = torch.cat(
            [torch.arange(line_num) for line_num in num_samples]).to(device)
        data_from = torch.cat([torch.ones(line_num, dtype=torch.long)
                              * i for i, line_num in enumerate(num_samples)]).to(device)

        selected_indices = select_mwis_subset(
            training_info=all_training_info,
            influence_score=all_scores.float(),
            k_select=args.max_samples,
            knn_k=20,
            sim_threshold=0.7
        )

        sorted_score_file = os.path.join(output_path, f"sorted.csv")
        final_index_list = file_specific_index[selected_indices].tolist()
        final_data_from = data_from[selected_indices].tolist()

        sorted_score_file = os.path.join(output_path, f"sorted.csv")
        if not os.path.exists(sorted_score_file):
            with open(sorted_score_file, 'w', encoding='utf-8') as file:
                file.write("file name, index, score\n")
                for score, index, name in zip(all_scores[selected_indices], final_index_list, final_data_from):
                    file.write(
                        f"{args.train_file_names[name]}, {index}, {round(score.item(), 6)}\n"
                    )

        all_lines = []
        for i, train_file in enumerate(args.train_files):
            with open(train_file, 'r', encoding='utf-8', errors='ignore') as file:
                all_lines.append(file.readlines()[:num_samples[i]])

        with open(os.path.join(output_path, f"top_{data_amount_name}.jsonl"), 'w', encoding='utf-8', errors='ignore') as file:
            for index, data_from_idx in zip(final_index_list, final_data_from):
                try:
                    file.write(all_lines[data_from_idx][index])
                except:
                    import pdb
                    pdb.set_trace()
