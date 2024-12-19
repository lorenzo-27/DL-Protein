import torch
from torch import nn
import yaml


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        """Blocco di convoluzione con due layer convoluzionali, BatchNorm e ReLU."""
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        """Blocco di discesa con MaxPooling seguito da un ConvBlock."""
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv = ConvBlock(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        """Blocco di risalita con un'operazione di upsampling e concatenazione."""
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels, kernel_size)

    def forward(self, x, skip_connection):
        x = self.upconv(x)
        diffY = skip_connection.size()[2] - x.size()[2]
        diffX = skip_connection.size()[3] - x.size()[3]
        x = nn.functional.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat((x, skip_connection), dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, base_filters, kernel_size):
        """Implementazione della U-Net con blocchi modulari."""
        super().__init__()
        self.encoder1 = ConvBlock(in_channels, base_filters, kernel_size)
        self.encoder2 = DownBlock(base_filters, base_filters * 2, kernel_size)
        self.encoder3 = DownBlock(base_filters * 2, base_filters * 4, kernel_size)
        self.encoder4 = DownBlock(base_filters * 4, base_filters * 8, kernel_size)

        self.bottleneck = ConvBlock(base_filters * 8, base_filters * 16, kernel_size)

        self.decoder4 = UpBlock(base_filters * 16, base_filters * 8, kernel_size)
        self.decoder3 = UpBlock(base_filters * 8, base_filters * 4, kernel_size)
        self.decoder2 = UpBlock(base_filters * 4, base_filters * 2, kernel_size)
        self.decoder1 = UpBlock(base_filters * 2, base_filters, kernel_size)

        self.final_conv = nn.Conv2d(base_filters, out_channels, kernel_size=1)

    def forward(self, x):
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
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    model_params = config['model']
    model = UNet(
        in_channels=model_params['in_channels'],
        out_channels=model_params['out_channels'],
        base_filters=model_params['base_filters'],
        kernel_size=model_params['kernel_size'],
    )
    return model


def main():
    """Funzione di test per verificare l'architettura del modello."""
    from torchinfo import summary
    from rich.console import Console
    console = Console()

    batch_size = 4
    input_size = 128
    input_data = torch.randn(batch_size, 57, input_size, input_size)

    config_path = "m_unet.yaml"
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
