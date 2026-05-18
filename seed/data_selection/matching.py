import argparse
import os

import torch

argparser = argparse.ArgumentParser(
    description='Script for selecting the data for training')
argparser.add_argument('--gradient_path', type=str, default="{} ckpt{}",
                       help='The path to the gradient file')
argparser.add_argument('--train_file_names', type=str, nargs='+',
                       help='The name of the training file')
argparser.add_argument('--ckpts', type=int, nargs='+',
                       help="Checkpoint numbers.")
argparser.add_argument('--checkpoint_weights', type=float, nargs='+',
                       help="checkpoint weights")
argparser.add_argument('--target_task_names', type=str,
                       nargs='+', help="The name of the target tasks")
argparser.add_argument('--validation_gradient_path', type=str,
                       default="{} ckpt{}", help='The path to the validation gradient file')
argparser.add_argument('--output_path', type=str, default="selected_data",
                       help='The path to the output')


args = argparser.parse_args()

N_SUBTASKS = {"mmlu": 57, "bbh": 27, "tydiqa": 9}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_influence_score(training_info: torch.Tensor, validation_info: torch.Tensor):
    """Calculate the influence score.

    Args:
        training_info (torch.Tensor): training info (gradients/representations) stored in a tensor of shape N x N_DIM
        validation_info (torch.Tensor): validation info (gradients/representations) stored in a tensor of shape N_VALID x N_DIM
    """
    # N x N_VALID
    influence_scores = torch.matmul(
        training_info, validation_info.transpose(0, 1)) # [100000, 8192] @ [8192, 9] = [100000, 9]
    return influence_scores


def calculate_influence_score_subspace(training_info: torch.Tensor,
                              validation_info: torch.Tensor):
    """
    Influence score with subspace selection (training-free).

    Args:
        training_info: [N_cand, C]
        validation_info: [N_val, C]

    Returns:
        influence_scores: [N_cand, N_val]
        channel_mask: [C] (0/1)
    """

    # 1. training-free channel importance
    m_cand = training_info.abs().mean(dim=0)      # [C]
    m_val  = validation_info.abs().mean(dim=0)    # [C]
    # print(training_info.sum(), training_info)

    # 2. mean thresholding
    mask_cand = m_cand > m_cand.mean()
    mask_val  = m_val  > m_val.mean()
    # print(f"Selected {mask_cand.sum()} candidates and {mask_val.sum()} validations")

    channel_mask = mask_cand & mask_val  # boolean AND

    # 3. apply mask
    training_masked = training_info[:, channel_mask]
    validation_masked = validation_info[:, channel_mask]
    print(f"Selected {channel_mask.sum()} channels")

    # 4. influence score
    influence_scores = torch.matmul(
        training_masked, validation_masked.transpose(0, 1)
    )

    return influence_scores


# renormalize the checkpoint weights
if sum(args.checkpoint_weights) != 1:
    s = sum(args.checkpoint_weights)
    args.checkpoint_weights = [i/s for i in args.checkpoint_weights]

# calculate the influence score for each validation task
for target_task_name in args.target_task_names:
    for train_file_name in args.train_file_names:
        influence_score = 0
        all_training_info = []
        for i, ckpt in enumerate(args.ckpts):
            validation_path = args.validation_gradient_path.format(
            target_task_name, ckpt)
            # validation_path = args.validation_gradient_path.format(
            #     ckpt, target_task_name)
            if os.path.isdir(validation_path):
                validation_path = os.path.join(validation_path, "all_orig.pt")
            validation_info = torch.load(validation_path)

            if not torch.is_tensor(validation_info):
                validation_info = torch.tensor(validation_info)
            validation_info = validation_info.to(device).float()
            gradient_path = args.gradient_path.format(train_file_name, ckpt)
            # gradient_path = args.gradient_path.format(ckpt, train_file_name)
            if os.path.isdir(gradient_path):
                gradient_path = os.path.join(gradient_path, "all_orig.pt")
            training_info = torch.load(gradient_path)

            if not torch.is_tensor(training_info):
                training_info = torch.tensor(training_info)
            training_info = training_info.to(device).float()
            all_training_info.append(training_info)

            influence_score += args.checkpoint_weights[i] * \
                calculate_influence_score_subspace(
                    training_info=training_info, validation_info=validation_info)
        influence_score = influence_score.reshape(
            influence_score.shape[0], N_SUBTASKS[target_task_name], -1).mean(-1).max(-1)[0]
        output_dir = os.path.join(args.output_path, target_task_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file = os.path.join(
            args.output_path, target_task_name, f"{train_file_name}_influence_score.pt")
        torch.save(influence_score, output_file)

        all_training_info = torch.mean(torch.stack(all_training_info, dim=0), dim=0)    
        print(all_training_info.shape)
        torch.save(all_training_info, os.path.join(output_dir, f"{train_file_name}_training_info.pt"))
        print("Saved influence score to {}".format(output_file))
