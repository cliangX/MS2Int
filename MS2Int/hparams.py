import argparse


def get_hparams():
    parser = argparse.ArgumentParser(description="VAT training configuration parser")

    parser.add_argument("--experiment_name", type=str, default="MS2Int_VAT", help="Experiment name")
    parser.add_argument("--world_size", type=int, default=1, help="Number of distributed processes")
    parser.add_argument("--train_data_path", type=str, default="data/train.h5", help="Training data path (H5)")
    parser.add_argument("--fine_tune", type=str, default="data/finetune.h5", help="Fine-tune data path (H5)")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/ms2int_vat/", help="Checkpoint output directory")
    parser.add_argument("--log_path", type=str, default="logs/train.log", help="Log file path")
    parser.add_argument("--pth", type=str, default="checkpoints/pretrained.pth", help="Pretrained model weights path")

    parser.add_argument("--train_batch_size", type=int, default=512, help="Batch size for training")
    parser.add_argument("--val_batch_size", type=int, default=1024, help="Batch size for validation")
    parser.add_argument("--test_batch_size", type=int, default=1024, help="Batch size for testing")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of data loading workers")
    parser.add_argument("--train_data_size", type=float, default=0.9, help="Train split ratio")
    parser.add_argument("--val_data_size", type=float, default=0.1, help="Validation split ratio")

    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--warmup_iters", type=int, default=4000, help="Number of warmup iterations")
    parser.add_argument("--max_iters", type=int, default=1000000, help="Maximum number of iterations")
    parser.add_argument("--n_layer", type=int, default=4, help="Number of Mamba layers")
    parser.add_argument("--d_model", type=int, default=512, help="Model hidden dimension (must match pretrained ckpt)")

    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum training epochs")

    parser.add_argument("--vat_alpha", type=float, default=1.0, help="VAT loss weight (total = sup_loss + alpha * vat_loss)")
    parser.add_argument("--vat_eps", type=float, default=2.0, help="VAT perturbation radius")
    parser.add_argument("--vat_xi", type=float, default=1e-6, help="VAT initial perturbation scale")
    parser.add_argument("--vat_ip", type=int, default=1, help="VAT power iteration steps")

    parser.add_argument("--server", type=str, default="localhost", help="Master address for distributed training")
    parser.add_argument("--port", type=str, default=29500, help="Master port for distributed training")

    return parser.parse_args()
