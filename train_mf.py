import os
import pickle
import pandas as pd
import argparse
import torch
from base_model.MF import MF

def main(args):
    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset info
    dataset_path = os.path.join(args.data_root, args.dataset)
    info_path = os.path.join(dataset_path, "convert_dict.pkl")
    print(f"Loading dataset info from {info_path}")
    overall_info = pickle.load(open(info_path, "rb"))

    # Load data
    train_data_path = args.train_data if args.train_data else os.path.join(dataset_path, "warm_emb.csv")
    print(f"Loading training data from {train_data_path}")
    training_data = pd.read_csv(train_data_path)
    warm_valid_data = pd.read_csv(os.path.join(dataset_path, "warm_val.csv"))
    warm_test_data = pd.read_csv(os.path.join(dataset_path, "warm_test.csv"))
    cold_valid_data = pd.read_csv(os.path.join(dataset_path, "cold_item_val.csv"))
    cold_test_data = pd.read_csv(os.path.join(dataset_path, "cold_item_test.csv"))

    # Construct overall sets for full evaluation
    overall_valid_data = pd.concat([warm_valid_data, cold_valid_data], 
                                   axis=0, ignore_index=True).sample(frac=1, random_state=args.seed).reset_index(drop=True)
    overall_test_data = pd.concat([warm_test_data, cold_test_data], 
                                   axis=0, ignore_index=True).sample(frac=1, random_state=args.seed).reset_index(drop=True)

    user_num = overall_info["user_num"]
    item_num = overall_info["item_num"]
    warm_user_idx = overall_info["warm_user"]
    warm_item_idx = overall_info["warm_item"]
    cold_user_idx = overall_info["cold_user"]
    cold_item_idx = overall_info["cold_item"]

    # Instantiate MF trainer
    # Note: MF class expectation for arguments:
    # args, training_data, warm_valid_data, cold_valid_data, all_valid_data,
    # warm_test_data, cold_test_data, all_test_data, user_num, item_num,
    # warm_user_idx, warm_item_idx, cold_user_idx, cold_item_idx, device
    
    mf = MF(args=args, 
            training_data=training_data.to_numpy().tolist(),
            warm_valid_data=warm_valid_data.to_numpy().tolist(),
            cold_valid_data=cold_valid_data.to_numpy().tolist(),
            all_valid_data=overall_valid_data.to_numpy().tolist(),
            warm_test_data=warm_test_data.to_numpy().tolist(),
            cold_test_data=cold_test_data.to_numpy().tolist(),
            all_test_data=overall_test_data.to_numpy().tolist(),
            user_num=user_num,
            item_num=item_num,
            warm_user_idx=warm_user_idx,
            warm_item_idx=warm_item_idx,
            cold_user_idx=cold_user_idx,
            cold_item_idx=cold_item_idx,
            device=device)

    # Load pre-trained weights if specified
    if args.load_model:
        mf.load_emb(args.pretrain_root, args.load_model)

    # Ensure save directory exists
    if args.save_emb:
        os.makedirs(args.save_root, exist_ok=True)

    # Run training and evaluation
    if args.test_only:
        print("Running evaluation only...")
        mf.test_only()
    else:
        print("Starting training...")
        mf.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument('--data_root', type=str, default='../dataset', help='Root directory for dataset')
    parser.add_argument('--pretrain_root', type=str, default='./model_weight', help='Root directory for loading pre-trained weights')
    parser.add_argument('--save_root', type=str, default='./model_weight', help='Root directory for saving checkpoints')
    parser.add_argument('--train_data', type=str, default='', help='Concrete path to the training data file')
    parser.add_argument('--dataset', type=str, default='CiteULike', help='Dataset name')
    
    # Model Hyperparameters
    parser.add_argument('--model', type=str, default='MF', help='Model name')
    parser.add_argument('--emb_size', type=int, default=64, help='Embedding size')
    parser.add_argument('--epochs', type=int, default=20, help='Max epochs')
    parser.add_argument('--bs', type=int, default=1024, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--reg', type=float, default=1e-4, help='Regularization')
    parser.add_argument('--topN', type=str, default='10,20,50', help='Top N for evaluation')
    
    # Execution settings
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_emb', action='store_true', help='Save embeddings after training')
    parser.add_argument('--test_only', action='store_true', help='Only run evaluation')
    parser.add_argument('--load_model', type=str, default='', help='Model prefix to load pre-trained weights from')
    parser.add_argument('--cold_object', type=str, default='item', help='Cold object type (user/item)')
    parser.add_argument('--LLM_type', type=str, default='Llama2-7B', help='LLM type for path naming')

    args = parser.parse_args()
    
    # Adjust paths if on Kaggle
    if os.path.exists('/kaggle/input'):
        print("Kaggle environment detected. Adjusting paths.")
        # Override data_root if not explicitly set to something else
        if args.data_root == '../dataset':
             # You might need to adjust this depending on how the dataset is named in Kaggle
             args.data_root = '/kaggle/input'
        if args.pretrain_root == './model_weight':
            args.pretrain_root = '/kaggle/input/model-weights' # Example path, adjust as needed
        if args.save_root == './model_weight':
            args.save_root = '/kaggle/working/model_weight'
            os.makedirs(args.save_root, exist_ok=True)

    main(args)
