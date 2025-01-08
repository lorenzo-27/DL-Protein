import torch
from torch import nn
import yaml


class FCN(nn.Module):
    def __init__(self, in_channels, out_channels, base_filters, kernel_size, dropout_rate):
        """
        1D Fully Convolutional Network (FCN) implementation based on Long et al. 2015,
        adapted for protein secondary structure prediction.
        """
        super().__init__()

        # VGG16-style encoder (adapted to 1D)
        self.conv1_1 = nn.Conv1d(in_channels, base_filters * 2, kernel_size, padding="same")
        self.conv1_2 = nn.Conv1d(base_filters * 2, base_filters * 2, kernel_size, padding="same")
        self.pool1 = nn.MaxPool1d(2, stride=2, ceil_mode=True)

        self.conv2_1 = nn.Conv1d(base_filters * 2, base_filters * 4, kernel_size, padding="same")
        self.conv2_2 = nn.Conv1d(base_filters * 4, base_filters * 4, kernel_size, padding="same")
        self.pool2 = nn.MaxPool1d(2, stride=2, ceil_mode=True)

        self.conv3_1 = nn.Conv1d(base_filters * 4, base_filters * 8, kernel_size, padding="same")
        self.conv3_2 = nn.Conv1d(base_filters * 8, base_filters * 8, kernel_size, padding="same")
        self.conv3_3 = nn.Conv1d(base_filters * 8, base_filters * 8, kernel_size, padding="same")
        self.pool3 = nn.MaxPool1d(2, stride=2, ceil_mode=True)

        self.conv4_1 = nn.Conv1d(base_filters * 8, base_filters * 16, kernel_size, padding="same")
        self.conv4_2 = nn.Conv1d(base_filters * 16, base_filters * 16, kernel_size, padding="same")
        self.conv4_3 = nn.Conv1d(base_filters * 16, base_filters * 16, kernel_size, padding="same")
        self.pool4 = nn.MaxPool1d(2, stride=2, ceil_mode=True)

        self.conv5_1 = nn.Conv1d(base_filters * 16, base_filters * 16, kernel_size, padding="same")
        self.conv5_2 = nn.Conv1d(base_filters * 16, base_filters * 16, kernel_size, padding="same")
        self.conv5_3 = nn.Conv1d(base_filters * 16, base_filters * 16, kernel_size, padding="same")
        self.pool5 = nn.MaxPool1d(2, stride=2, ceil_mode=True)

        # FCN-specific layers
        self.fc6 = nn.Conv1d(base_filters * 16, base_filters * 64, 1)
        self.fc7 = nn.Conv1d(base_filters * 64, base_filters * 64, 1)

        # Score layers for multi-scale prediction
        self.score_pool3 = nn.Conv1d(base_filters * 8, out_channels, 1)
        self.score_pool4 = nn.Conv1d(base_filters * 16, out_channels, 1)
        self.score_final = nn.Conv1d(base_filters * 64, out_channels, 1)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        input_size = x.size(-1)

        # Encoder
        x = self.relu(self.conv1_1(x))
        x = self.relu(self.conv1_2(x))
        x = self.pool1(x)

        x = self.relu(self.conv2_1(x))
        x = self.relu(self.conv2_2(x))
        x = self.pool2(x)

        x = self.relu(self.conv3_1(x))
        x = self.relu(self.conv3_2(x))
        x = self.relu(self.conv3_3(x))
        pool3 = self.pool3(x)

        x = self.relu(self.conv4_1(pool3))
        x = self.relu(self.conv4_2(x))
        x = self.relu(self.conv4_3(x))
        pool4 = self.pool4(x)

        x = self.relu(self.conv5_1(pool4))
        x = self.relu(self.conv5_2(x))
        x = self.relu(self.conv5_3(x))
        x = self.pool5(x)

        # FCN conversion of FC layers
        x = self.relu(self.fc6(x))
        x = self.dropout(x)
        x = self.relu(self.fc7(x))
        x = self.dropout(x)

        # Multi-scale prediction
        score_final = self.score_final(x)
        score_pool4 = self.score_pool4(pool4)
        score_pool3 = self.score_pool3(pool3)

        # Upsample and fuse scores
        score_final = nn.functional.interpolate(score_final, size=score_pool4.size()[-1], mode='linear', align_corners=True)
        score_final += score_pool4

        score_final = nn.functional.interpolate(score_final, size=score_pool3.size()[-1], mode='linear', align_corners=True)
        score_final += score_pool3

        # Final upsampling to match input size
        score_final = nn.functional.interpolate(score_final, size=input_size, mode='linear', align_corners=True)
        # Final normalization
        score_final = self.bn(score_final)

        return score_final


def load_model(config_path):
    """Load FCN1D model based on parameters provided in a YAML file."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    model_params = config["model"]
    model = FCN(
        in_channels=model_params["in_channels"],
        out_channels=model_params["out_channels"],
        base_filters=model_params["base_filters"],
        kernel_size=model_params["kernel_size"],
        dropout_rate=model_params["dropout_rate"],
    )
    return model


def main():
    """Funzione di test per verificare l'architettura del modello."""
    from rich.console import Console
    from torchinfo import summary

    console = Console()

    batch_size = 32
    input_length = 700
    input_data = torch.randn(batch_size, 41, input_length)

    config_path = "m_fcn.yaml"
    model = load_model(config_path)

    model_stats = summary(
        model,
        input_data=input_data,
        col_names=["input_size", "output_size", "num_params"],
        row_settings=("var_names",),
        col_width=18,
        depth=4,
        verbose=0,
    )
    console.print(model_stats)


if __name__ == "__main__":
    from ipdb import launch_ipdb_on_exception

    with launch_ipdb_on_exception():
        main()