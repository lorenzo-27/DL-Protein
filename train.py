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

from models import m_cnn, m_fcn, m_unet, m_unet_dropout, m_unet_embedding, m_unet_dropout_embedding

CNN = m_cnn.CNN
FCN = m_fcn.FCN
UNet = m_unet.UNet
UNetDropout = m_unet_dropout.UNetDropout
UNetEmbedding = m_unet_embedding.UNetEmbedding
UNetDropoutEmbedding = m_unet_dropout_embedding.UNetDropoutEmbedding


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
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    loss_function = torch.nn.CrossEntropyLoss()

    step = 0
    best_val_accuracy = 0
    best_val_loss = np.inf
    best_model_state = None
    epochs_without_improvement = 0
    early_stopping_patience = opts.training["early_stopping_patience"]

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
                log_metrics(train_writer, {"train_loss": avg_loss, "train_accuracy": avg_acc}, step)
                step += 1

        if opts.training["include_q3"] == False:
            val_loss, val_q8_accuracy = evaluate(model, val_loader, loss_function, opts)
            test_loss, test_q8_accuracy = evaluate(model, test_loader, loss_function, opts)
            cb513_loss, cb513_q8_accuracy = evaluate(model, cb513_test_loader, loss_function, opts)
            log_metrics(val_writer, {"loss": val_loss, "accuracy": val_q8_accuracy}, epoch)
            log_metrics(test_writer, {"loss": test_loss, "accuracy": test_q8_accuracy}, epoch)
            log_metrics(cb513_writer, {"loss": cb513_loss, "accuracy": cb513_q8_accuracy}, epoch)
            LOG.info(f"Epoch {epoch}: Acc[Q8] - ValLoss = {val_loss:.6f}, ValAcc={val_q8_accuracy:.3f}, TestLoss={test_loss:.6f}, TestAcc={test_q8_accuracy:.3f}, CB513Loss = {cb513_loss:.6f}, CB513Acc={cb513_q8_accuracy:.3f}")

        elif opts.training["include_q3"] == True:
            val_loss, val_q8_accuracy, val_q3_accuracy = evaluate(model, val_loader, loss_function, opts)
            test_loss, test_q8_accuracy, test_q3_accuracy = evaluate(model, test_loader, loss_function, opts)
            cb513_loss, cb513_q8_accuracy, cb513_q3_accuracy = evaluate(model, cb513_test_loader, loss_function, opts)
            log_metrics(val_writer, {"loss": val_loss, "accuracy": val_q8_accuracy, "accuracy": val_q3_accuracy}, epoch)
            log_metrics(test_writer, {"loss": test_loss, "accuracy": test_q8_accuracy, "accuracy": test_q3_accuracy}, epoch)
            log_metrics(cb513_writer, {"loss": cb513_loss, "accuracy": cb513_q8_accuracy, "accuracy": cb513_q3_accuracy}, epoch)
            LOG.info(f"Epoch {epoch}: Acc[Q8/Q3] - ValLoss = {val_loss:.6f}, ValAcc={val_q8_accuracy:.3f}/{val_q3_accuracy:.3f}, TestLoss={test_loss:.6f}, TestAcc={test_q8_accuracy:.3f}/{test_q3_accuracy:.3f}, CB513Loss = {cb513_loss:.6f}, CB513Acc={cb513_q8_accuracy:.3f}/{cb513_q3_accuracy:.3f}")

        else:
            raise ValueError(f"Unknown train type {opts.training['include_q3']}")

        # Early stopping check
        if val_q8_accuracy >= best_val_accuracy and val_loss <= best_val_loss:
            best_val_accuracy = val_q8_accuracy
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stopping_patience:
            LOG.info(f"Early stopping triggered after {epoch} epochs. Maximum Validation Q8 accuracy: {best_val_accuracy:.3f}. Minimum Validation loss: {best_val_loss:.6f}")
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
                save_checkpoint(model, optimizer, epoch, loss.item(), opts)
            break

        # Salvataggio del checkpoint
        if epoch % opts.training["save_every"] == 0:
            save_checkpoint(model, optimizer, epoch, loss.item(), opts)

        # scheduler.step()
        scheduler.step(cb513_loss)

def evaluate(model, loader, loss_function, opts):
    """Calcola la loss e l'accuratezza Q8 e Q3 sul dataset di validazione o test."""
    model.eval()

    total_loss = 0
    q8_correct_predictions = 0
    q8_total_predictions = 0

    if opts.training["include_q3"] == True:
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
        q3_correct_predictions = 0
        q3_total_predictions = 0

    with torch.no_grad():
        for X, Y in loader:
            X, Y = X.to(opts.device), Y.to(opts.device)
            outputs = model(X)

            loss = loss_function(outputs.reshape(-1, 8), Y.reshape(-1, 8).argmax(dim=1))
            total_loss += loss.item()

            q8_predictions = torch.argmax(outputs, dim=1)
            q8_labels = Y.argmax(dim=1)
            q8_correct_predictions += (q8_predictions == q8_labels).sum().item()
            q8_total_predictions += q8_labels.numel()

            if opts.training["include_q3"] == True:
                q3_predictions = torch.tensor(
                    [q8_to_q3[p.item()] for p in q8_predictions.flatten()],
                    device=opts.device,
                ).reshape(q8_predictions.shape)
                q3_labels = torch.tensor(
                    [q8_to_q3[l.item()] for l in q8_labels.flatten()],
                    device=opts.device,
                ).reshape(q8_labels.shape)
                q3_correct_predictions = (q3_predictions == q3_labels).sum().item()
                q3_total_predictions = q3_labels.numel()

    avg_loss = total_loss / len(loader)
    q8_accuracy = q8_correct_predictions / q8_total_predictions

    if opts.training["include_q3"] == False:
        return avg_loss, q8_accuracy
    elif opts.training["include_q3"] == True:
        q3_accuracy = q3_correct_predictions / q3_total_predictions
        return avg_loss, q8_accuracy, q3_accuracy
    else:
        raise ValueError(f"Unknown train type {opts.training['include_q3']}")

def log_metrics(writer, metrics, epoch):
    with writer.as_default():
        for key, value in metrics.items():
            tf.summary.scalar(key, value, step=epoch)

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
        )
        visualize(
            model,
            "fcn",
            torch.randn(opts.training["batch_size"], opts.model["in_channels"], 700),
        )
    elif opts.model["name"] == "unet_dropout":
        model = UNetDropout(
            in_channels=opts.model["in_channels"],
            out_channels=opts.model["out_channels"],
            base_filters=opts.model["base_filters"],
            kernel_size=opts.model["kernel_size"],
            dropout_rate=opts.model["dropout_rate"],
        )
        visualize(
            model,
            "unet_dropout",
            torch.randn(opts.training["batch_size"], opts.model["in_channels"], 700),
        )
    elif opts.model["name"] == "unet_embedding":
        model = UNetEmbedding(
            in_channels=opts.model["in_channels"],
            out_channels=opts.model["out_channels"],
            base_filters=opts.model["base_filters"],
            kernel_size=opts.model["kernel_size"],
            embedding_dim=opts.model["embedding_dim"],
        )
        visualize(
            model,
            "unet_embedding",
            torch.randn(opts.training["batch_size"], opts.model["in_channels"], 700),
        )
    elif opts.model["name"] == "unet_dropout_embedding":
        model = UNetDropoutEmbedding(
            in_channels=opts.model["in_channels"],
            out_channels=opts.model["out_channels"],
            base_filters=opts.model["base_filters"],
            kernel_size=opts.model["kernel_size"],
            dropout_rate=opts.model["dropout_rate"],
            embedding_dim=opts.model["embedding_dim"],
        )
        visualize(
            model,
            "unet_dropout_embedding",
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
