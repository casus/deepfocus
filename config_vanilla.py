# settings for vanilla 2D Unet
# training parameters

import argparse

def configuration():
    parser = argparse.ArgumentParser()
    
    # Dataset arguments
    parser.add_argument('--data_path', type=str, default='./data/interim/inputData/', help='Dataset path')  # input of the data
    parser.add_argument('--input_shape', type=int, default=[256, 256, 1], help='[W, H, CH]')
    
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size")
    parser.add_argument('--learning_rate', type=float, default=0.0001, help="learning rate")
    parser.add_argument('--optimizer', type=str, default='Adam', help="optimizer")
    
    parser.add_argument('--neptune_document', type=bool, default=True, help="neptune doc")
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--patience', type=int, default=10, help='patience for early stopping')
    
    # training parameters
    parser.add_argument('--tensorboard_path', type=str, default='./tensorboard/UNet2D/', help='tensorboard')
    parser.add_argument('--model_path', type=str, default='./models_weight/', help='trained model')
    
    args, unknown = parser.parse_known_args()
    
    return args