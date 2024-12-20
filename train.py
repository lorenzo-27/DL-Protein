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
    fname = os.path.join(opts.training['checkpoint_dir'], f'e_{epoch:05d}.chp')
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
    train_writer = tf.summary.create_file_writer(f"tensorboard/{opts.model['name']}/train")
    val_writer = tf.summary.create_file_writer(f"tensorboard/{opts.model['name']}/val")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=opts.training['learning_rate'], weight_decay=opts.training['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    loss_function = torch.nn.CrossEntropyLoss()

    step = 0
    for epoch in range(1, opts.training['num_epochs'] + 1):
        model.train()
        train_losses, train_accuracies = [], []
        for batch_idx, (X, Y) in enumerate(train_loader):
            X, Y = X.to(opts.device), Y.to(opts.device).long()
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

            if batch_idx % opts.training['log_every'] == 0:
                avg_loss = np.mean(train_losses[-opts.training['batch_window']:])
                avg_acc = np.mean(train_accuracies[-opts.training['batch_window']:])
                LOG.info(
                    f"Epoch {epoch}, Batch {batch_idx}: Loss={avg_loss:.6f}, Accuracy={avg_acc:.3f}"
                )

                # Scrittura su TensorBoard
                with train_writer.as_default():
                    tf.summary.scalar("loss", avg_loss, step=step)
                    tf.summary.scalar("accuracy", avg_acc, step=step)

                step += 1

        # Validazione
        val_accuracy = q8_accuracy(model, val_loader, opts)
        with val_writer.as_default():
            tf.summary.scalar("accuracy", val_accuracy, step=epoch)

        LOG.info(f"Epoch {epoch}: Validation Accuracy={val_accuracy:.3f}")

        # Salvataggio del checkpoint
        if epoch % opts.training['save_every'] == 0:
            save_checkpoint(model, optimizer, epoch, loss.item(), opts)

        scheduler.step()


def q8_accuracy(model, loader, opts):
    """Calcola la Q8 accuracy sul dataset di validazione."""
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for X, Y in loader:
            X, Y = X.to(opts.device), Y.to(opts.device)
            outputs = model(X)
            predictions = torch.argmax(outputs, dim=1)
            correct_predictions += (predictions == Y).sum().item()
            total_predictions += Y.numel()
    return correct_predictions / total_predictions


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

    visualize(model, "unet", torch.randn(opts.training['batch_size'], opts.model['in_channels'], 700))
    model = model.to(opts.device)
    LOG.info("Caricamento del dataset")

    # Caricamento dei dati
    train_X, train_y, val_X, val_y, test_X, test_y = data_load(opts.dataset['train_path'])
    eval_X, eval_y = data_load(opts.dataset['test_path'])

    print("Contenuto di val_X: ", val_X)
    print("Contenuto di val_y: ", val_y)
    print("Contenuto di test_X: ", test_X)
    print("Contenuto di test_y: ", test_y)

    # Creazione dei DataLoader
    train_loader = torch.utils.data.DataLoader(
        dataset=list(zip(torch.tensor(train_X), torch.tensor(train_y))),
        batch_size=opts.training['batch_size'],
        shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=list(zip(torch.tensor(val_X), torch.tensor(val_y))),
        batch_size=opts.training['batch_size'],
        shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=list(zip(torch.tensor(test_X), torch.tensor(test_y))),
        batch_size=opts.training['batch_size'],
        shuffle=False
    )
    cb513_test_loader = torch.utils.data.DataLoader(
        dataset=list(zip(torch.tensor(eval_X), torch.tensor(eval_y))),
        batch_size=opts.training['batch_size'],
        shuffle=False
    )

    LOG.info("Inizio del training")
    train_loop(model, train_loader, val_loader, opts)

    LOG.info("Inizio della valutazione su cullpdb+profile_6133")
    q8_accuracy(model, test_loader, opts)

    LOG.info("Inizio della valutazione su cb513")
    q8_accuracy(model, cb513_test_loader, opts)


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
