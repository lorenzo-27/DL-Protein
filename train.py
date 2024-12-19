import os
import logging
import argparse
from types import SimpleNamespace

import yaml
import numpy as np
import torch
import tensorflow as tf
from rich.logging import RichHandler
from ipdb import launch_ipdb_on_exception

from m_unet import UNet


def get_logger():
    FORMAT = "%(message)s"
    logging.basicConfig(
        level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
    )
    log = logging.getLogger("rich")
    return log


LOG = get_logger()


def N(x):
    """Converte un tensore Torch in un array NumPy."""
    return x.detach().cpu().numpy()


def save_checkpoint(model, optimizer, epoch, loss, opts):
    fname = os.path.join(opts.checkpoint_dir, f'e_{epoch:05d}.chp')
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss,
    }
    torch.save(checkpoint, fname)
    LOG.info(f"Saved checkpoint {fname}")


def train_loop(model, train_loader, val_loader, opts):
    """Funzione principale per il ciclo di training."""
    train_writer = tf.summary.create_file_writer(f"tensorboard/{opts.model}/train")
    val_writer = tf.summary.create_file_writer(f"tensorboard/{opts.model}/val")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=opts.learning_rate, weight_decay=opts.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    loss_function = torch.nn.CrossEntropyLoss()

    step = 0
    for epoch in range(1, opts.num_epochs + 1):
        model.train()
        train_losses, train_accuracies = [], []
        for batch_idx, (X, Y) in enumerate(train_loader):
            X, Y = X.to(opts.device), Y.to(opts.device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = loss_function(outputs, Y)
            loss.backward()
            optimizer.step()

            # Calcolo metriche
            train_losses.append(loss.item())
            predictions = torch.argmax(outputs, dim=1)
            accuracy = (predictions == Y).float().mean().item()
            train_accuracies.append(accuracy)

            if batch_idx % opts.log_every == 0:
                avg_loss = np.mean(train_losses[-opts.batch_window:])
                avg_acc = np.mean(train_accuracies[-opts.batch_window:])
                LOG.info(
                    f"Epoch {epoch}, Batch {batch_idx}: Loss={avg_loss:.6f}, Accuracy={avg_acc:.3f}"
                )

                # Scrittura su TensorBoard
                with train_writer.as_default():
                    tf.summary.scalar("loss", avg_loss, step=step)
                    tf.summary.scalar("accuracy", avg_acc, step=step)

                step += 1

        # Validazione
        val_accuracy = evaluate(model, val_loader, opts)
        with val_writer.as_default():
            tf.summary.scalar("accuracy", val_accuracy, step=epoch)

        LOG.info(f"Epoch {epoch}: Validation Accuracy={val_accuracy:.3f}")

        # Salvataggio del checkpoint
        if epoch % opts.save_every == 0:
            save_checkpoint(model, optimizer, epoch, loss.item(), opts)

        scheduler.step()


def evaluate(model, loader, opts):
    """Calcola l'accuratezza sul dataset di validazione."""
    model.eval()
    accuracies = []
    with torch.no_grad():
        for X, Y in loader:
            X, Y = X.to(opts.device), Y.to(opts.device)
            outputs = model(X)
            predictions = torch.argmax(outputs, dim=1)
            accuracy = (predictions == Y).float().mean().item()
            accuracies.append(accuracy)
    return np.mean(accuracies)


def main(opts):
    from visualize_network import visualize
    from dataset import data_load
    """Funzione principale che gestisce il caricamento del modello e dei dati."""
    LOG.info(f"Caricamento del modello {opts.model}")
    if opts.model['name'] == "unet":
        model = UNet(
            in_channels=opts.model['in_channels'],
            out_channels=opts.model['out_channels'],
            base_filters=opts.model['base_filters'],
            kernel_size=opts.model['kernel_size'],
        )
    else:
        raise ValueError(f"Modello sconosciuto {opts.model['name']}")
    visualize(model, "unet", torch.randn(opts.training['batch_size'], 57, 128, 128))
    model = model.to(opts.device)
    LOG.info("Caricamento del dataset")
    train_loader = data_load(opts.dataset['train_path'])
    val_loader = data_load(opts.dataset['test_path'])

    LOG.info("Inizio del training")
    train_loop(model, train_loader, val_loader, opts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to YAML configuration file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    opts = SimpleNamespace(**config)
    opts.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with launch_ipdb_on_exception():
        main(opts)
