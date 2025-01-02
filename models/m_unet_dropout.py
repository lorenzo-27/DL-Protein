import torch
import yaml
from sympy.strategies.branch import identity
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout_rate):
        """Blocco convoluzionale 1D con BatchNorm, ReLU e Dropout."""
        super().__init__()
        self.use_residual = in_channels == out_channels
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding="same")
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding="same")
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)

        if not self.use_residual:
            self.residual_conv = nn.Conv1d(in_channels, out_channels, 1)
            self.residual_bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.use_residual:
            out += identity
        else:
            residual = self.residual_conv(identity)
            residual = self.residual_bn(residual)
            out += residual

        out = self.relu2(out)
        out = self.dropout2(out)
        return out


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout_rate):
        """Blocco di discesa 1D con MaxPooling seguito da un ConvBlock."""
        super().__init__()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv = ConvBlock(in_channels, out_channels, kernel_size, dropout_rate)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout_rate):
        """Blocco di risalita con un'operazione di upsampling e concatenazione."""
        super().__init__()
        self.upconv = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        self.conv = ConvBlock(in_channels, out_channels, kernel_size, dropout_rate)

    def forward(self, x, skip_connection):
        x = self.upconv(x)
        diff = skip_connection.size(-1) - x.size(-1)
        x = nn.functional.pad(x, [diff // 2, diff - diff // 2])
        x = torch.cat((x, skip_connection), dim=1)
        x = self.conv(x)
        return x


class UNetDropout(nn.Module):
    def __init__(self, in_channels, out_channels, base_filters, kernel_size, dropout_rate):
        """Implementazione della U-Net 1D con blocchi modulari."""
        super().__init__()
        self.input_norm = nn.BatchNorm1d(in_channels)
        self.encoder1 = ConvBlock(in_channels, base_filters, kernel_size, dropout_rate)
        self.encoder2 = DownBlock(base_filters, base_filters * 2, kernel_size, dropout_rate)
        self.encoder3 = DownBlock(base_filters * 2, base_filters * 4, kernel_size, dropout_rate)
        self.encoder4 = DownBlock(base_filters * 4, base_filters * 8, kernel_size, dropout_rate)

        self.bottleneck = ConvBlock(base_filters * 8, base_filters * 16, kernel_size, dropout_rate)

        self.decoder4 = UpBlock(base_filters * 16, base_filters * 8, kernel_size, dropout_rate)
        self.decoder3 = UpBlock(base_filters * 8, base_filters * 4, kernel_size, dropout_rate)
        self.decoder2 = UpBlock(base_filters * 4, base_filters * 2, kernel_size, dropout_rate)
        self.decoder1 = UpBlock(base_filters * 2, base_filters, kernel_size, dropout_rate)

        self.final_conv = nn.Conv1d(base_filters, out_channels, kernel_size=1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.ConvTranspose1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.input_norm(x)
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
    model = UNetDropout(
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

    batch_size = 256
    input_length = 700
    input_data = torch.randn(batch_size, 21, input_length)

    config_path = "m_unet_dropout.yaml"
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
