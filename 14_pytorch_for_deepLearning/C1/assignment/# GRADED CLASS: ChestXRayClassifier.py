# GRADED CLASS: ChestXRayClassifier

class ChestXRayClassifier(pl.LightningModule):
    """A LightningModule that is focused on tracking validation loss and accuracy."""

    def __init__(self, model_weights_path, num_classes=3, learning_rate=1e-3, weight_decay=1e-2):
        """
        Initializes the ChestXRayClassifier module.

        Args:
            model_weights_path (str): The file path to the pre-trained ResNet-18 model weights.
            num_classes (int): The number of classes for classification. Defaults to 3.
            learning_rate (float): The learning rate for the optimizer. Defaults to 1e-3.
            weight_decay (float): The weight decay (L2 penalty) for the optimizer. Defaults to 1e-2.
        """
        super().__init__()
        # Save all __init__ arguments (model_weights_path, num_classes, etc.) to self.hparams
        # For example, you’ll access `num_classes` as `self.hparams.num_classes`
        self.save_hyperparameters()
        
        ### START CODE HERE ###

        # Call the `load_resnet18` helper function to get the pre-trained model.
        self.model = load_resnet18(
            self.hparams.num_classes,
            self.hparams.model_weights_path,
        ) 
        
        # Define the Cross Entropy loss function.
        self.loss_fn = nn.CrossEntropyLoss()
        
        # Define the accuracy metric using `Accuracy`.
        # Remember to specify the task and the number of classes.
        self.accuracy = Accuracy(task="multiclass", num_classes=self.hparams.num_classes)
        
        ### END CODE HERE ###

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor of images.
        """
        return self.model(x)

    def training_step(self, batch, batch_idx=None):
        """
        Performs a single training step. Loss calculation is required for backpropagation.

        Args:
        batch (tuple): A tuple containing the input images and their labels.
        batch_idx (int): The index of the current batch. The Lightning Trainer
                         requires this argument, but it's not utilized in this
                         implementation as the logic is the same for all batches.
        """
        
        ### START CODE HERE ###
        
        # Unpack the batch into images and labels.
        x, y = batch
        # Perform a forward pass to get the model's logits.
        logits = self(x)
        # Calculate the loss by comparing the logits to the true labels.
        loss = self.loss_fn(y, logits)
        
        ### END CODE HERE ###
        
        return loss

    def validation_step(self, batch, batch_idx=None):
        """
        Performs a single validation step and logs only the loss and accuracy.

        Args:
        batch (tuple): A tuple containing the input images and their labels.
        batch_idx (int): The index of the current batch. The Lightning Trainer
                         requires this argument, but it's not utilized in this
                         implementation as the logic is the same for all batches.
        """
        
        ### START CODE HERE ###
        
        # Unpack the batch into images and labels.
        x, y = batch
        # Perform a forward pass to get the model's logits.
        logits = self(x)
         # Calculate the loss.
        loss = slef.loss_fn(y, logits)
        # Calculate the accuracy.
        acc = self.accuracy(y, logits)
        
         ### END CODE HERE ###
        
        # Log metrics for this validation epoch and show them in the progress bar.
        self.log_dict({'val_loss': loss, 'val_acc': acc}, prog_bar=True)

    def configure_optimizers(self):
        """Configures the optimizers and learning rate scheduler."""
        
        ### START CODE HERE ###

        # Call the `define_optimizer_and_scheduler` helper function.
        optimizer, scheduler = define_optimizer_and_scheduler(
            self.model,
            self.hparams.learning_rate,
            self.hparams.weight_decay
        ) 
        
         ### END CODE HERE ###
        
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}