import argparse


def get_hparams():
    parser = argparse.ArgumentParser(description="VAT training configuration parser")

    parser.add_argument("--experiment_name", type=str, default="MS2Int_VAT", help="实验名称")

    parser.add_argument("--world_size", type=int, default=1, help="模式")
    parser.add_argument(
        "--train_data_pth",
        type=str,
        default="data/train.h5",
        help="训练数据路径（H5）。建议运行时通过命令行覆盖。",
    )
    parser.add_argument("--fine_tune", type=str, default="data/finetune.h5", help="数据路径（可选，占位默认值）")
    # parser.add_argument('--train_data_pth', type=str, default="/mnt/data/lcy/train/train_29_31_004732.h5", help='数据路径')
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/ms2int_vat/", help="模型输出目录（会自动创建）")
    parser.add_argument("--log_path", type=str, default="logs/train.log", help="日志路径（会自动创建父目录）")
    parser.add_argument(
        "--pth",
        type=str,
        default="checkpoints/pretrained.pth",
        help="初始模型权重路径（用于微调）。建议运行时通过命令行覆盖。",
    )

    # batch_size 的设置
    parser.add_argument("--train_batch_size", type=int, default=512, help="Batch size for training")
    parser.add_argument("--val_batch_size", type=int, default=1024, help="Batch size for validation")
    parser.add_argument("--test_batch_size", type=int, default=1024, help="Batch size for testing")
    parser.add_argument("--num_workers", type=int, default=16, help="Batch size for testing")
    parser.add_argument("--train_data_size", type=float, default=0.9, help="Batch size for testing")
    parser.add_argument("--val_data_size", type=float, default=0.1, help="Batch size for testing")

    # 模型架构参数
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for training")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for optimizer")
    parser.add_argument("--warmup_iters", type=int, default=4000, help="Number of warmup iterations")
    parser.add_argument("--max_iters", type=int, default=1000000, help="Maximum number of iterations")

    parser.add_argument("--n_layer", type=int, default=4, help="Number of warmup iterations")
    # 该 ckpt 实际使用的隐藏维度为 512（从权重形状推断），不要改成 1024
    parser.add_argument("--d_model", type=int, default=512, help="模型隐藏维度，需与预训练 ckpt 保持一致")

    # 训练器的参数
    parser.add_argument("--max_epochs", type=int, default=100, help="Type of hardware accelerator")

    # VAT 参数（MSE 一致性约束）
    parser.add_argument("--vat_alpha", type=float, default=1.0, help="VAT 损失权重（总损失 = 监督损失 + vat_alpha * vat_loss）")
    parser.add_argument("--vat_eps", type=float, default=2.0, help="VAT 扰动半径 eps（作用在序列 embedding 输出上）")
    parser.add_argument("--vat_xi", type=float, default=1e-6, help="VAT 初始扰动尺度 xi（用于估计方向）")
    parser.add_argument("--vat_ip", type=int, default=1, help="VAT power iteration 次数（默认 1）")

    # 服务器设置
    parser.add_argument("--server", type=str, default="localhost", help="服务器地址")
    parser.add_argument("--port", type=str, default=29500, help="端口号")

    return parser.parse_args()
