import torch
import yaml
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        """Blocco di convoluzione 1D con due layer convoluzionali, BatchNorm e ReLU."""
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding="same"),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding="same"),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        """Blocco di discesa 1D con MaxPooling seguito da un ConvBlock."""
        super().__init__()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv = ConvBlock(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        """Blocco di risalita con un'operazione di upsampling e concatenazione."""
        super().__init__()
        self.upconv = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        self.conv = ConvBlock(in_channels, out_channels, kernel_size)

    def forward(self, x, skip_connection):
        x = self.upconv(x)
        diff = skip_connection.size(-1) - x.size(-1)
        x = nn.functional.pad(x, [diff // 2, diff - diff // 2])
        x = torch.cat((x, skip_connection), dim=1)
        x = self.conv(x)
        return x


class UNetEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, base_filters, kernel_size, embedding_dim):
        """Implementazione della U-Net 1D con embedding layer."""
        super().__init__()
        self.embedding = nn.Embedding(in_channels, embedding_dim)
        self.encoder1 = ConvBlock(embedding_dim, base_filters, kernel_size)
        self.encoder2 = DownBlock(base_filters, base_filters * 2, kernel_size)
        self.encoder3 = DownBlock(base_filters * 2, base_filters * 4, kernel_size)
        self.encoder4 = DownBlock(base_filters * 4, base_filters * 8, kernel_size)

        self.bottleneck = ConvBlock(base_filters * 8, base_filters * 16, kernel_size)

        self.decoder4 = UpBlock(base_filters * 16, base_filters * 8, kernel_size)
        self.decoder3 = UpBlock(base_filters * 8, base_filters * 4, kernel_size)
        self.decoder2 = UpBlock(base_filters * 4, base_filters * 2, kernel_size)
        self.decoder1 = UpBlock(base_filters * 2, base_filters, kernel_size)

        self.final_conv = nn.Conv1d(base_filters, out_channels, kernel_size=1)

    def forward(self, x):
        x = torch.argmax(x, dim=1).long() # (batch_size, in_channels, sequence_length) -> (batch_size, sequence_length)
        x = self.embedding(x).permute(0, 2, 1)  # Change to (batch_size, embedding_dim, sequence_length)

        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        bottleneck = self.bottleneck(enc4)

        dec4 = self.decoder4(bottleneck, enc4)
        dec3 = self.decoder3(dec4, enc3)
        dec2 = self.decoder2(dec3, enc2)
        dec1 = self.decoder1(dec2, enc1)

        return self.final_conv(dec1)


def load_model(config_path):
    """Carica il modello U-Net basato sui parametri forniti in un file YAML."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    model_params = config["model"]
    model = UNetEmbedding(
        in_channels=model_params["in_channels"],
        out_channels=model_params["out_channels"],
        base_filters=model_params["base_filters"],
        kernel_size=model_params["kernel_size"],
        embedding_dim=model_params["embedding_dim"],
    )
    return model


def main():
    """Funzione di test per verificare l'architettura del modello."""
    from rich.console import Console
    from torchinfo import summary

    console = Console()

    batch_size = 8
    input_length = 700
    input_data = torch.randn(batch_size, 21, input_length)

    config_path = "m_unet_embedding.yaml"
    model = load_model(config_path)

    _ = model(input_data)
    model_stats = summary(
        model,
        input_data=input_data,
        col_names=["input_size", "output_size", "num_params"],
        row_settings=("var_names",),
        col_width=18,
        depth=8,
        verbose=0,
    )
    console.print(model_stats)


if __name__ == "__main__":
    from ipdb import launch_ipdb_on_exception

    with launch_ipdb_on_exception():
        main()
