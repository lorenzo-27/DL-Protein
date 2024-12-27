import argparse
import logging
import os
from types import SimpleNamespace

import numpy as np
import tensorflow as tf
import torch
import yaml
from ipdb import launch_ipdb_on_exception
from rich.logging import RichHandler

from m_cnn import CNN
from m_fcn import FCN
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
    fname = os.path.join(opts.training["checkpoint_dir"], f"e_{epoch:05d}.chp")
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss,
    }
    torch.save(checkpoint, fname)
    LOG.info(f"Saved checkpoint {fname}")


def train_loop(model, train_loader, val_loader, test_loader, cb513_test_loader, opts):
    """Funzione principale per il ciclo di training."""
    train_writer = tf.summary.create_file_writer(
        f"tensorboard/{opts.model['name']}/train"
    )
    val_writer = tf.summary.create_file_writer(f"tensorboard/{opts.model['name']}/val")
    test_writer = tf.summary.create_file_writer(
        f"tensorboard/{opts.model['name']}/test"
    )
    cb513_writer = tf.summary.create_file_writer(
        f"tensorboard/{opts.model['name']}/cb513"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opts.training["learning_rate"],
        weight_decay=opts.training["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    loss_function = torch.nn.CrossEntropyLoss()

    step = 0
    for epoch in range(1, opts.training["num_epochs"] + 1):
        model.train()
        train_losses, train_accuracies = [], []
        for batch_idx, (X, Y) in enumerate(train_loader):
            X, Y = X.to(opts.device), Y.to(opts.device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = loss_function(outputs.reshape(-1, 8), Y.reshape(-1, 8).argmax(dim=1))
            loss = torch.mean(loss)  # serve per evitare un errore di dimensione
            loss.backward()
            optimizer.step()

            # Calcolo metriche
            train_losses.append(loss.item())
            predictions = torch.argmax(outputs, dim=1)
            accuracy = (predictions == Y.argmax(dim=1)).float().mean().item()
            train_accuracies.append(accuracy)

            if batch_idx % opts.training["log_every"] == 0:
                avg_loss = np.mean(train_losses[-opts.training["batch_window"] :])
                avg_acc = np.mean(train_accuracies[-opts.training["batch_window"] :])
                LOG.info(
                    f"Epoch {epoch}, Batch {batch_idx}: Loss={avg_loss:.6f}, Accuracy={avg_acc:.3f}"
                )

                # Scrittura su TensorBoard
                with train_writer.as_default():
                    tf.summary.scalar("loss", avg_loss, step=step)
                    tf.summary.scalar("train_accuracy", avg_acc, step=step)

                step += 1

        # Validazione
        val_q8_accuracy = q8_accuracy(model, val_loader, opts)
        test_q8_accuracy = q8_accuracy(model, test_loader, opts)
        cb513_q8_accuracy = q8_accuracy(model, cb513_test_loader, opts)
        val_q3_accuracy = q3_accuracy(model, val_loader, opts)
        test_q3_accuracy = q3_accuracy(model, test_loader, opts)
        cb513_q3_accuracy = q3_accuracy(model, cb513_test_loader, opts)
        with val_writer.as_default():
            tf.summary.scalar("val_q8_accuracy", val_q8_accuracy, step=epoch)
            tf.summary.scalar("val_q3_accuracy", val_q3_accuracy, step=epoch)
        with test_writer.as_default():
            tf.summary.scalar("val_q8_accuracy", test_q8_accuracy, step=epoch)
            tf.summary.scalar("val_q3_accuracy", test_q3_accuracy, step=epoch)
        with cb513_writer.as_default():
            tf.summary.scalar("val_q8_accuracy", cb513_q8_accuracy, step=epoch)
            tf.summary.scalar("val_q3_accuracy", cb513_q3_accuracy, step=epoch)

        LOG.info(
            f"Epoch {epoch}: Validation Accuracy (Q8 | Q3) = {val_q8_accuracy:.3f} | {val_q3_accuracy:.3f}, Test Accuracy (Q8 | Q3) = {test_q8_accuracy:.3f} | {test_q3_accuracy:.3f}, CB513 Accuracy (Q8 | Q3) = {cb513_q8_accuracy:.3f} | {cb513_q3_accuracy:.3f}"
        )

        # Salvataggio del checkpoint
        if epoch % opts.training["save_every"] == 0:
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
            correct_predictions += (predictions == Y.argmax(dim=1)).sum().item()
            total_predictions += Y.numel() // 8
    return correct_predictions / total_predictions

def q3_accuracy(model, loader, opts):
    """Calcola la Q3 accuracy sul dataset di validazione."""
    # Mappatura da Q8 a Q3
    q8_to_q3 = {
        0: 0,  # H -> H (helix)
        1: 0,  # G -> H (helix)
        2: 0,  # I -> H (helix)
        3: 1,  # E -> E (strand)
        4: 1,  # B -> E (strand)
        5: 2,  # T -> C (coil)
        6: 2,  # S -> C (coil)
        7: 2,  # C -> C (coil)
    }

    model.eval()
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for X, Y in loader:
            X, Y = X.to(opts.device), Y.to(opts.device)
            outputs = model(X)
            predictions_q8 = torch.argmax(outputs, dim=1)
            predictions_q3 = torch.tensor(
                [q8_to_q3[p.item()] for p in predictions_q8.flatten()],
                device=opts.device,
            ).reshape(predictions_q8.shape)

            labels_q8 = Y.argmax(dim=1)
            labels_q3 = torch.tensor(
                [q8_to_q3[l.item()] for l in labels_q8.flatten()],
                device=opts.device,
            ).reshape(labels_q8.shape)

            correct_predictions += (predictions_q3 == labels_q3).sum().item()
            total_predictions += labels_q3.numel()

    return correct_predictions / total_predictions


def main(opts):
    from dataset import load_data
    from visualize_network import visualize

    """Funzione principale che gestisce il caricamento del modello e dei dati."""
    LOG.info(f"Caricamento del modello {opts.model}")
    if opts.model["name"] == "unet":
        model = UNet(
            in_channels=opts.model["in_channels"],
            out_channels=opts.model["out_channels"],
            base_filters=opts.model["base_filters"],
            kernel_size=opts.model["kernel_size"],
        )
        visualize(
            model,
            "unet",
            torch.randn(opts.training["batch_size"], opts.model["in_channels"], 700),
        )
    elif opts.model["name"] == "cnn":
        model = CNN(
            in_channels=opts.model["in_channels"],
            out_channels=opts.model["out_channels"],
            base_filters=opts.model["base_filters"],
            kernel_size=opts.model["kernel_size"],
        )
        visualize(
            model,
            "cnn",
            torch.randn(opts.training["batch_size"], opts.model["in_channels"], 700),
        )
    elif opts.model["name"] == "fcn":
        model = FCN(
            in_channels=opts.model["in_channels"],
            out_channels=opts.model["out_channels"],
            base_filters=opts.model["base_filters"],
        )
        visualize(
            model,
            "fcn",
            torch.randn(opts.training["batch_size"], opts.model["in_channels"], 700),
        )
    else:
        raise ValueError(f"Modello sconosciuto {opts.model['name']}")

    model = model.to(opts.device)
    LOG.info("Caricamento del dataset")

    # Caricamento dei dati
    train_X, train_y, val_X, val_y, test_X, test_y = load_data(
        opts.dataset["train_path"], opts.dataset["seed"]
    )
    cb513_test_X, cb513_test_y = load_data(
        opts.dataset["test_path"], opts.dataset["seed"]
    )

    # Creazione dei DataLoader
    train_loader = torch.utils.data.DataLoader(
        dataset=list(zip(torch.tensor(train_X), torch.tensor(train_y))),
        batch_size=opts.training["batch_size"],
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=list(zip(torch.tensor(val_X), torch.tensor(val_y))),
        batch_size=opts.training["batch_size"],
        shuffle=False,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=list(zip(torch.tensor(test_X), torch.tensor(test_y))),
        batch_size=opts.training["batch_size"],
        shuffle=False,
    )
    cb513_test_loader = torch.utils.data.DataLoader(
        dataset=list(zip(torch.tensor(cb513_test_X), torch.tensor(cb513_test_y))),
        batch_size=opts.training["batch_size"],
        shuffle=False,
    )

    LOG.info("Inizio del training")
    train_loop(model, train_loader, val_loader, test_loader, cb513_test_loader, opts)


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
