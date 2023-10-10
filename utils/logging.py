#  Copyright (c) 2023 Andrew
#  Email: andrewlee1807@gmail.com

from utils.directory import create_file
import sys
import yaml
import os
# Get all arguments from command
def arg_parse(parser):
    parser.add_argument('--dataset_name', type=str, default='cnu', help='Dataset Name: household; cnu; spain; gyeonggi')
    parser.add_argument('--model_name', type=str, default='model1', help='Model Name: model1; model2; model3')
    parser.add_argument('--dataset_path', type=str, default='../dataset/', help='Dataset path')
    parser.add_argument('--config_path', type=str, help='Configuration file path')
    parser.add_argument('--output_length', type=int, default=1, help='Prediction Length')
    parser.add_argument('--device', type=int, default=0, help='CUDA Device')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    parser.add_argument('--write_log_file', type=bool, default=False,
                        help='Export to log file')  # if --write_log_file added, then we will export to log file
    return parser.parse_args()


def initialize_logging(file_name, pred_length):
    orig_stdout = sys.stdout
    if pred_length == 1:  # firstly execute, so we need to create a new file
        file_name = create_file(f'{file_name}')
    f = open(file_name, 'a')
    sys.stdout = f
    # Because of using a file to log, so we need to print the time to know when the program is running
    import datetime
    current_time = datetime.datetime.now().time()
    print("Running time - ", current_time)
    return f, orig_stdout


def close_logging(f, orig_stdout):
    print("====================================================================================================\n\n\n")
    sys.stdout = orig_stdout
    f.close()


def warming_up(args):
    """
    Configuring environment
    """
    print('Setting up environment...')
    # setup CUDA device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    print('Loading configuration file...')
    # read configuration
    with open(args.config_path, encoding='utf-8') as file:
        config = yaml.safe_load(file)
    config["output_length"] = args.output_length
    config["dataset_name"] = args.dataset_name
    config["output_dir"] = args.output_dir
    config["tensorboard_log_dir"] = f'{args.output_dir}/tensorboard_log/{config["output_length"]}'
    if "epochs" not in config.keys():
        config["epochs"] = 100
    if "batch_size" not in config.keys():
        config["batch_size"] = 64
    print("Loaded configuration successfully ", args.config_path)

    print('Setting up output directory...')
    # initialize output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    print("Output directory: ", args.output_dir)

    # initialize log file
    if args.write_log_file:
        print("Starting running background")
        print(
            f"To check output running, open file \"{os.path.join(args.output_dir, args.dataset_name)}_{config['output_length']} \"")
        file, orig_stdout = initialize_logging(f'{os.path.join(args.output_dir, args.dataset_name)}_training.log',
                                               config["output_length"])
        config["file"] = file
        config["orig_stdout"] = orig_stdout

    print("Configuration:")
    import pprint

    pprint.pprint(config)
    return config
