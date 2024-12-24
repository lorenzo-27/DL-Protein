import torch
import yaml
from torch import nn


class FCN(nn.Module):
    def __init__(self, in_channels, out_channels, base_filters):
        """Implementazione di un Fully Convolutional Network (FCN) 1D."""
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, base_filters, kernel_size=7, padding=3),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(),
            nn.Conv1d(base_filters, base_filters * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_filters * 2),
            nn.ReLU(),
            nn.Conv1d(base_filters * 2, base_filters * 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_filters * 4),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Conv1d(base_filters * 4, base_filters * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_filters * 2),
            nn.ReLU(),
            nn.Conv1d(base_filters * 2, base_filters, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(),
            nn.Conv1d(base_filters, out_channels, kernel_size=1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def load_fcn_model(config_path):
    """Carica il modello FCN1D basato sui parametri forniti in un file YAML."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    model_params = config["model"]
    model = FCN(
        in_channels=model_params["in_channels"],
        out_channels=model_params["out_channels"],
        base_filters=model_params["base_filters"],
    )
    return model


def main():
    """Funzione di test per verificare l'architettura del modello."""
    from rich.console import Console
    from torchinfo import summary

    console = Console()

    batch_size = 256
    input_length = 700
    input_data = torch.randn(batch_size, 21, input_length)

    config_path = "fcn_1d.yaml"
    model = load_fcn_model(config_path)

    _ = model(input_data)
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
