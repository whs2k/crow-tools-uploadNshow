import torch
import torch.nn as nn
import pytorch_lightning as pl
from lightning_fabric import seed_everything
import os
import argparse # Import argparse for command-line arguments

# Ensure reproducibility.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class CrowClassifier(pl.LightningModule):
    """
    Defines the CrowClassifier model architecture as provided by the user.
    This class is essential for loading the PyTorch Lightning checkpoint.
    """
    def __init__(self, input_dim=768, hidden_dim=237, dropout_rate=0.3, seed=18202, lr=0.000145):
        super().__init__()

        seed_everything(seed, workers=True)
        self.lr = lr

        # A simple shared backbone.
        self.backbone = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Output heads for each task.
        self.crowCount_head = nn.Linear(hidden_dim, 5)  # 5 classes (labels: 0,1,2,3,4)
        self.crowAge_head = nn.Linear(hidden_dim, 2)    # 2 classes: adult vs juvenile
        self.alert_head = nn.Linear(hidden_dim, 1)      # binary
        self.begging_head = nn.Linear(hidden_dim, 1)    # binary
        self.softSong_head = nn.Linear(hidden_dim, 1)   # binary
        self.rattle_head = nn.Linear(hidden_dim, 1)     # binary
        self.mob_head = nn.Linear(hidden_dim, 1)        # binary
        self.quality_head = nn.Linear(hidden_dim, 2)    # 3 classes: bad, average, HQ

        # Loss functions (not strictly needed for ONNX export, but good to keep for completeness).
        self.loss_fn_class = nn.CrossEntropyLoss()
        self.loss_fn_bce = nn.BCEWithLogitsLoss()

    def forward(self, x):
        """
        Defines the forward pass of the model.
        """
        rep = self.backbone(x)
        out = {
            "crowCount": self.crowCount_head(rep),
            "crowAge": self.crowAge_head(rep),
            "alert": self.alert_head(rep),
            "begging": self.begging_head(rep),
            "softSong": self.softSong_head(rep),
            "rattle": self.rattle_head(rep),
            "mob": self.mob_head(rep),
            "quality": self.quality_head(rep)
        }
        return out

    def training_step(self, batch, batch_idx):
        """Training step (not used for ONNX conversion, but part of PL model)."""
        embeddings, labels = batch
        outputs = self(embeddings)
        loss = (
            self.loss_fn_class(outputs["crowCount"], labels["crowCount"].long()) +
            self.loss_fn_class(outputs["crowAge"], (labels["crowAge"] - 1).long()) +
            self.loss_fn_bce(outputs["alert"].view(-1), labels["alert"].float().view(-1)) +
            self.loss_fn_bce(outputs["begging"].view(-1), labels["begging"].float().view(-1)) +
            self.loss_fn_bce(outputs["softSong"].view(-1), labels["softSong"].float().view(-1)) +
            self.loss_fn_bce(outputs["rattle"].view(-1), labels["rattle"].float().view(-1)) +
            self.loss_fn_bce(outputs["mob"].view(-1), labels["mob"].float().view(-1)) +
            self.loss_fn_class(outputs["quality"], (labels["quality"] - 1).long())
        )
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step (not used for ONNX conversion, but part of PL model)."""
        embeddings, labels = batch
        outputs = self(embeddings)
        loss = (
            self.loss_fn_class(outputs["crowCount"], labels["crowCount"].long()) +
            self.loss_fn_class(outputs["crowAge"], (labels["crowAge"] - 1).long()) +
            self.loss_fn_bce(outputs["alert"].view(-1), labels["alert"].float().view(-1)) +
            self.loss_fn_bce(outputs["begging"].view(-1), labels["begging"].float().view(-1)) +
            self.loss_fn_bce(outputs["softSong"].view(-1), labels["softSong"].float().view(-1)) +
            self.loss_fn_bce(outputs["rattle"].view(-1), labels["rattle"].float().view(-1)) +
            self.loss_fn_bce(outputs["mob"].view(-1), labels["mob"].float().view(-1)) +
            self.loss_fn_class(outputs["quality"], (labels["quality"] - 1).long())
        )

        pred_crowCount = torch.argmax(outputs["crowCount"], dim=1)
        pred_crowAge = torch.argmax(outputs["crowAge"], dim=1)
        pred_quality = torch.argmax(outputs["quality"], dim=1)
        pred_alert = (outputs["alert"].squeeze() > 0).long().view(-1)
        pred_begging = (outputs["begging"].squeeze() > 0).long().view(-1)
        pred_softSong = (outputs["softSong"].squeeze() > 0).long().view(-1)
        pred_rattle = (outputs["rattle"].squeeze() > 0).long().view(-1)
        pred_mob = (outputs["mob"].squeeze() > 0).long().view(-1)

        out = {
            "crowCount": {"pred": pred_crowCount, "gt": labels["crowCount"]},
            "crowAge": {"pred": pred_crowAge, "gt": labels["crowAge"] - 1},
            "quality": {"pred": pred_quality, "gt": labels["quality"] - 1},
            "alert": {"pred": pred_alert, "gt": labels["alert"]},
            "begging": {"pred": pred_begging, "gt": labels["begging"]},
            "softSong": {"pred": pred_softSong, "gt": labels["softSong"]},
            "rattle": {"pred": pred_rattle, "gt": labels["rattle"]},
            "mob": {"pred": pred_mob, "gt": labels["mob"]},
            "loss": loss
        }
        self._validation_outputs.append(out)
        return out

    def on_validation_epoch_start(self):
        """Clear validation outputs."""
        self._validation_outputs = []

    def on_validation_epoch_end(self):
        """Aggregate and log validation metrics."""
        multi_class_keys = {"crowCount": 5, "crowAge": 2, "quality": 2}
        breakdown = {key: {cls: [0, 0] for cls in range(num)} for key, num in multi_class_keys.items()}
        binary_keys = ["alert", "begging", "softSong", "rattle", "mob"]
        breakdown_binary = {key: {0: [0, 0], 1: [0, 0]} for key in binary_keys}

        total_loss = 0.0
        batch_count = len(self._validation_outputs)
        for batch_out in self._validation_outputs:
            total_loss += batch_out["loss"].item()
            for task in multi_class_keys:
                preds = batch_out[task]["pred"].detach().cpu()
                gts = batch_out[task]["gt"].detach().cpu()
                for cls in range(multi_class_keys[task]):
                    mask = (gts == cls)
                    total = mask.sum().item()
                    correct = ((preds == cls) & mask).sum().item()
                    breakdown[task][cls][0] += correct
                    breakdown[task][cls][1] += total
            for task in binary_keys:
                preds = batch_out[task]["pred"].detach().cpu()
                gts = batch_out[task]["gt"].detach().cpu()
                for val in [0, 1]:
                    mask = (gts == val)
                    total = mask.sum().item()
                    correct = ((preds == val) & mask).sum().item()
                    breakdown_binary[task][val][0] += correct
                    breakdown_binary[task][val][1] += total

        avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
        composite, task_scores = self.compute_composite_score(breakdown, breakdown_binary)

        self.log("val_loss", avg_loss, prog_bar=True)
        self.log("val_composite_score", composite, prog_bar=True)
        for task, score in task_scores.items():
            self.log(f"val_{task}_score", score)

    def compute_composite_score(self, breakdown, breakdown_binary, weights=None):
        """Compute a composite score for validation."""
        if weights is None:
            weights = {
                "crowCount": 2, "crowAge": 1, "quality": 1,
                "alert": 1.5, "begging": 1.5, "softSong": 1.5,
                "rattle": 2, "mob": 2
            }
        task_scores = {}

        multi_class_tasks = ["crowCount", "crowAge", "quality"]
        for task in multi_class_tasks:
            class_accuracies = []
            for cls, (correct, total) in breakdown[task].items():
                if total > 0:
                    class_accuracies.append(correct / total)
            task_scores[task] = sum(class_accuracies) / len(class_accuracies) if class_accuracies else 0.0

        binary_tasks = ["alert", "begging", "softSong", "rattle", "mob"]
        for task in binary_tasks:
            acc_vals = []
            for val in [0, 1]:
                correct, total = breakdown_binary[task][val]
                acc_vals.append((correct / total) if total > 0 else 0.0)
            task_scores[task] = sum(acc_vals) / 2.0

        total_weighted = 0.0
        total_weights = 0.0
        for task, score in task_scores.items():
            weight = weights.get(task, 1)
            total_weighted += weight * score
            total_weights += weight
        composite_score = total_weighted / total_weights if total_weights > 0 else 0.0
        return composite_score, task_scores

    def configure_optimizers(self):
        """Configure optimizers (not used for ONNX conversion, but part of PL model)."""
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def convert_ckpt_to_onnx(
    ckpt_file_path: str,
    onnx_file_path: str,
    input_dim: int = 768,
    hidden_dim: int = 237,
    dropout_rate: float = 0.3,
    seed: int = 18202,
    lr: float = 0.000145
):
    """
    Converts a PyTorch Lightning .ckpt model to an ONNX .onnx file.

    Args:
        ckpt_file_path (str): Path to the PyTorch Lightning checkpoint file (.ckpt).
        onnx_file_path (str): Path where the ONNX model will be saved (.onnx).
        input_dim (int): Input dimension for the model.
        hidden_dim (int): Hidden dimension for the model.
        dropout_rate (float): Dropout rate for the model.
        seed (int): Seed for reproducibility.
        lr (float): Learning rate (not directly used for export, but for model init).
    """
    if not os.path.exists(ckpt_file_path):
        print(f"Error: Checkpoint file not found at '{ckpt_file_path}'")
        return

    try:
        # 1. Load the PyTorch Lightning model from the checkpoint.
        # We need to pass the arguments that the CrowClassifier's __init__ method expects.
        # These should match the arguments used when the model was trained and saved.
        print(f"Loading model from checkpoint: {ckpt_file_path}")
        model = CrowClassifier.load_from_checkpoint(
            ckpt_file_path,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            seed=seed,
            lr=lr
        )
        model.eval()  # Set the model to evaluation mode (important for ONNX export)
        print("Model loaded successfully.")

        # 2. Create a dummy input tensor for ONNX export.
        # The batch size can be 1 or any other reasonable value.
        # The input_dim should match the expected input to the model's forward method.
        # Assuming input is a 1D tensor of shape (batch_size, input_dim).
        dummy_input = torch.randn(1, input_dim)
        print(f"Created dummy input tensor with shape: {dummy_input.shape}")

        # 3. Define the input and output names for the ONNX graph.
        # These names will appear in the ONNX model and are useful for debugging
        # and understanding the model graph in ONNX viewers.
        input_names = ["input_embeddings"]
        output_names = [
            "crowCount_output", "crowAge_output", "alert_output",
            "begging_output", "softSong_output", "rattle_output",
            "mob_output", "quality_output"
        ]

        # 4. Export the model to ONNX format.
        # dynamic_axes allows for variable batch sizes.
        print(f"Exporting model to ONNX: {onnx_file_path}")
        torch.onnx.export(
            model,
            dummy_input,
            onnx_file_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={"input_embeddings": {0: "batch_size"}} # Allow variable batch size
        )
        print(f"Model successfully converted and saved to '{onnx_file_path}'")

    except Exception as e:
        print(f"An error occurred during conversion: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # --- Command-line argument parsing ---
    parser = argparse.ArgumentParser(description="Convert a PyTorch Lightning .ckpt model to ONNX format.")
    parser.add_argument(
        "--ckpt_name",
        type=str,
        required=True,
        help="Name of the .ckpt checkpoint file (e.g., my_model.ckpt)."
    )
    parser.add_argument(
        "--onnx_name",
        type=str,
        default="crow_classifier.onnx",
        help="Name for the output .onnx file (default: crow_classifier.onnx)."
    )
    # Add arguments for model initialization parameters if they might vary
    parser.add_argument("--input_dim", type=int, default=768, help="Input dimension of the model.")
    parser.add_argument("--hidden_dim", type=int, default=237, help="Hidden dimension of the model.")
    parser.add_argument("--dropout_rate", type=float, default=0.3, help="Dropout rate of the model.")
    parser.add_argument("--seed", type=int, default=18202, help="Seed for reproducibility.")
    parser.add_argument("--lr", type=float, default=0.000145, help="Learning rate (for model init).")

    args = parser.parse_args()

    # --- Configuration from arguments ---
    CHECKPOINT_FILENAME = args.ckpt_name
    ONNX_FILENAME = args.onnx_name

    # Ensure these parameters match the exact ones used when training your .ckpt model.
    # If your model was trained with different input_dim, hidden_dim, etc.,
    # you MUST adjust these values accordingly or provide them via command line.
    MODEL_INPUT_DIM = args.input_dim
    MODEL_HIDDEN_DIM = args.hidden_dim
    MODEL_DROPOUT_RATE = args.dropout_rate
    MODEL_SEED = args.seed
    MODEL_LR = args.lr

    print(f"Attempting to convert '{CHECKPOINT_FILENAME}' to '{ONNX_FILENAME}'...")
    convert_ckpt_to_onnx(
        ckpt_file_path=CHECKPOINT_FILENAME,
        onnx_file_path=ONNX_FILENAME,
        input_dim=MODEL_INPUT_DIM,
        hidden_dim=MODEL_HIDDEN_DIM,
        dropout_rate=MODEL_DROPOUT_RATE,
        seed=MODEL_SEED,
        lr=MODEL_LR
    )
