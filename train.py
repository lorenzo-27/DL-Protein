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

from models import m_fcn, m_unet, m_unet_optimized

FCN = m_fcn.FCN
UNet = m_unet.UNet
UNetOptimized = m_unet_optimized.UNetOptimized


def get_logger():
    FORMAT = "%(message)s"
    logging.basicConfig(
        level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
    )
    log = logging.getLogger("rich")
    return log

LOG = get_logger()


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
    
def log_metrics(writer, metrics, epoch):
    with writer.as_default():
        for key, value in metrics.items():
            tf.summary.scalar(key, value, step=epoch)

def train_loop(model, train_loader, cb513_test_loader, opts):
    """Funzione principale per il ciclo di training."""
    train_writer = tf.summary.create_file_writer(f"tensorboard/{opts.model['name']}/train")
    cb513_writer = tf.summary.create_file_writer(f"tensorboard/{opts.model['name']}/cb513")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opts.training["learning_rate"],
        weight_decay=opts.training["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6)
    loss_function = torch.nn.CrossEntropyLoss()

    step = 0
    best_cb513_loss = float("inf")
    best_model_state = None
    epochs_without_improvement = 0
    early_stopping_patience = opts.training["early_stopping_patience"]

    for epoch in range(1, opts.training["num_epochs"] + 1):
        model.train()

        train_total_loss = 0
        train_q8_correct = 0
        train_total_valid_positions = 0

        for batch_idx, (X, y, mask) in enumerate(train_loader):
            X, y, mask = X.to(opts.device), y.to(opts.device), mask.to(opts.device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(X)

            loss = loss_function(outputs, y)
            loss = (loss * mask).sum() / mask.sum()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_total_loss += loss.item()
            train_q8_predictions = torch.argmax(outputs, dim=1)
            train_q8_labels = torch.argmax(y, dim=1)
            train_q8_correct += ((train_q8_predictions == train_q8_labels) * mask).sum().item()
            train_total_valid_positions += mask.sum().item()

            if batch_idx % opts.training["log_every"] == 0:
                train_avg_loss = train_total_loss / (batch_idx + 1)
                train_q8_accuracy = train_q8_correct / train_total_valid_positions
                log_metrics(train_writer, {"train_loss": train_avg_loss, "train_accuracy": train_q8_accuracy}, step)
                LOG.info(f"Epoch {epoch}, Batch {batch_idx}: Loss={train_avg_loss:.6f}, [Q8]Acc={train_q8_accuracy:.3f}")
                step += 1

        if opts.training["include_q3"] == False:
            cb513_loss, cb513_q8_accuracy = evaluate(model, cb513_test_loader, loss_function, opts)
            log_metrics(cb513_writer, {"loss": cb513_loss, "accuracy": cb513_q8_accuracy}, epoch)
            LOG.info(f"Epoch {epoch}: CB513Loss={cb513_loss:.6f}, [Q8]CB513Acc={cb513_q8_accuracy:.3f}")

        else:
            cb513_loss, cb513_q8_accuracy, cb513_q3_accuracy = evaluate(model, cb513_test_loader, loss_function, opts)
            log_metrics(cb513_writer, {"loss": cb513_loss, "accuracy": cb513_q8_accuracy, "accuracy": cb513_q3_accuracy}, epoch)
            LOG.info(f"Epoch {epoch}: CB513Loss={cb513_loss:.6f}, [Q8/Q3]CB513Acc={cb513_q8_accuracy:.3f}/{cb513_q3_accuracy:.3f}")

        # Early stopping check
        if cb513_loss <= best_cb513_loss:
            best_cb513_loss = cb513_loss
            best_model_state = model.state_dict()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stopping_patience:
            LOG.info(f"Early stopping triggered after {epoch} epochs. Minimum cb513 loss: {best_cb513_loss:.6f}")
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
                save_checkpoint(model, optimizer, epoch, loss.item(), opts)
            break

        # Salvataggio del checkpoint
        if epoch % opts.training["save_every"] == 0:
            save_checkpoint(model, optimizer, epoch, loss.item(), opts)

        scheduler.step(cb513_loss)
        LOG.info(f"LR: {scheduler.get_last_lr()}")

def evaluate(model, loader, loss_function, opts):
    """Calcola la loss e l'accuratezza Q8 e Q3 sul dataset di validazione o test."""
    model.eval()

    total_loss = 0
    q8_correct = 0
    total_valid_positions = 0

    if opts.training["include_q3"]:
        q3_correct_predictions = 0
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

    with torch.no_grad():
        for X, Y, mask in loader:
            X, Y, mask = X.to(opts.device), Y.to(opts.device), mask.to(opts.device)
            outputs = model(X)

            loss = loss_function(outputs, Y)
            loss = (loss * mask).sum() / mask.sum()
            total_loss += loss.item()

            q8_predictions = torch.argmax(outputs, dim=1)
            q8_labels = torch.argmax(Y, dim=1)
            q8_correct += ((q8_predictions == q8_labels) * mask).sum().item()
            total_valid_positions += mask.sum().item()

            if opts.training["include_q3"]:
                q3_predictions = torch.tensor(
                    [q8_to_q3[p.item()] for p in q8_predictions.flatten()],
                    device=opts.device,
                ).reshape(q8_predictions.shape)

                q3_labels = torch.tensor(
                    [q8_to_q3[l.item()] for l in q8_labels.flatten()],
                    device=opts.device,
                ).reshape(q8_labels.shape)

                q3_correct_predictions = ((q3_predictions == q3_labels) * mask).sum().item()

    avg_loss = total_loss / len(loader)
    q8_accuracy = q8_correct / total_valid_positions

    if opts.training["include_q3"]:
        q3_accuracy = q3_correct_predictions / total_valid_positions
        return avg_loss, q8_accuracy, q3_accuracy
    else:
        return avg_loss, q8_accuracy

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
    elif opts.model["name"] == "fcn":
        model = FCN(
            in_channels=opts.model["in_channels"],
            out_channels=opts.model["out_channels"],
            base_filters=opts.model["base_filters"],
            kernel_size=opts.model["kernel_size"],
            dropout_rate=opts.model["dropout_rate"],
        )
        visualize(
            model,
            "fcn",
            torch.randn(opts.training["batch_size"], opts.model["in_channels"], 700),
        )
    elif opts.model["name"] == "unet_optimized":
        model = UNetOptimized(
            in_channels=opts.model["in_channels"],
            out_channels=opts.model["out_channels"],
            base_filters=opts.model["base_filters"],
            kernel_size=opts.model["kernel_size"],
            dropout_rate=opts.model["dropout_rate"],
        )
        visualize(
            model,
            "unet_optimized",
            torch.randn(opts.training["batch_size"], opts.model["in_channels"], 700),
        )
    else:
        raise ValueError(f"Modello sconosciuto {opts.model['name']}")

    model = model.to(opts.device)
    LOG.info("Caricamento del dataset")

    # Caricamento dei dati
    train_loader, cb513_test_loader = load_data(opts.training["batch_size"])

    LOG.info("Inizio del training")
    train_loop(model, train_loader, cb513_test_loader, opts)


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
