import argparse

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--model_name', type=str, default='resnet18')
    args = parser.parse_args()

    config = vars(args)  # Converts args to a dictionary
    return config

if __name__ == "__main__":
    config = get_config()
    print(config)
