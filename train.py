# %%
print("* Importing modules...")

import math
import torch
import pathlib
import datetime
import torch.nn as nn
import model as models
import dataset as datasets

# %%
DATASET_ROOT = "/opt/datasets/imagenet-256-dimensi0n/"
print(f"* Loading dataset from {DATASET_ROOT}")

dataframe_train, dataframe_test = datasets.make_train_test_split(pathlib.Path(DATASET_ROOT), random_state=128)

dataset_train = datasets.ImageNetDataset(dataframe=dataframe_train, transform=datasets.transform_extra)
dataset_test = datasets.ImageNetDataset(dataframe=dataframe_test, transform=datasets.transform_basic)

dataloader_train = torch.utils.data.DataLoader(
    dataset=dataset_train,
    batch_size=64+32+4,
    shuffle=True,
    num_workers=10,
    pin_memory=True,
    persistent_workers=True
)

dataloader_test = torch.utils.data.DataLoader(
    dataset=dataset_test,
    batch_size=512,
    shuffle=False,
    num_workers=8
)

print(f"* Train/test batches: {len(dataloader_train)}/{len(dataloader_test)}")

# %%
print("* Setting up helper functions and variables...")
class_counts = dataframe_train["label"].value_counts().sort_index()
class_weights = class_counts.sum() / (len(class_counts) * torch.tensor(class_counts.values, dtype=torch.float32))

def init_scheduler(num_epochs, steps_per_epoch, num_warmup_epochs):
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = num_warmup_epochs * steps_per_epoch

    def scheduler(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return 0.01 + (1 - 0.01) * cosine

    return scheduler

def checkpoint(model, optimizer, history, epoch, session=1, name="LightViT54M", save_dir="./checkpoints"):
    path = f"{save_dir}/{name}-T{session}-E{epoch:0>3}-{{}}.pth"
    torch.save(model.state_dict(), path.format("model"))
    torch.save(optimizer.state_dict(), path.format("optimizer"))
    torch.save(history, path.format("history"))

def predict(model, dataloader, max_batch=None, device="cuda"):
    model.eval()
    torch.cuda.empty_cache()
    y_pred, y_true = [], []
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader, 1):
            y_logits = model(X.to(device))
            y_pred += y_logits.argmax(dim=1).tolist()
            y_true += y.tolist()
            print(f"* Evaluating {batch}/{max_batch or len(dataloader)}", end="\r")
            if max_batch and batch >= max_batch: break
    return torch.tensor(y_pred), torch.tensor(y_true)

# %%
print("* Initializing the model")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.LightViTClassifier(
    target_resolution=(256, 256),
    num_classes=len(dataset_train.class_to_idx),
    num_patches=(16, 16),
    embedding_dim=512,
    num_layers=16,
    num_heads=8,
    dropout_p=0.1,
    in_channels=3
).to(device)

param_count = sum([param.numel() for param in model.parameters()])
print(f"* LightViTClassifier parameter count: {param_count:,d}")

optimizer = torch.optim.AdamW(
    params=model.parameters(),
    lr=2e-4,
    betas=(0.9, 0.999),
    eps=1e-6,
    weight_decay=0.01
)

criterion = nn.CrossEntropyLoss(
    weight=class_weights.to(device),
    label_smoothing=0.1
)

num_epochs = 128
accumulation_steps = 5

scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=init_scheduler(
        num_epochs=num_epochs,
        steps_per_epoch=len(dataloader_train) // accumulation_steps,
        num_warmup_epochs=10
    )
)

history = {
    "raw_batch_loss": [],
    "avg_batch_loss": [],
    "pred_match": [],
    "pred_count": [],
    "avg_batch_accuracy": [],
    "avg_epoch_accuracy": [],
    "learning_rate": []
}

# %%
timestamp_training_start = datetime.datetime.now()
print(f"* Starting training for {num_epochs} epochs on {timestamp_training_start}\n")

for epoch in range(1, num_epochs+1):
    model.train()
    optimizer.zero_grad()

    history["pred_match"].append(0)
    history["pred_count"].append(0)

    for batch, (X, y) in enumerate(dataloader_train, 1):
        X = X.to(device)
        y = y.to(device)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            y_logits = model(X)
            loss = criterion(y_logits, y) / accumulation_steps

        loss.backward()

        if batch % accumulation_steps == 0 or batch == len(dataloader_train):
            nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        loss_batch = loss.item() * accumulation_steps
        history["raw_batch_loss"].append(loss_batch)
        loss_avg = sum(history["raw_batch_loss"][-500:]) / len(history["raw_batch_loss"][-500:])
        history["avg_batch_loss"].append(loss_avg)

        history["pred_match"][-1] += (y_logits.argmax(dim=1) == y).sum().item()
        history["pred_count"][-1] += y.shape[0]
        accuracy_avg = history["pred_match"][-1] / history["pred_count"][-1]
        history["avg_batch_accuracy"].append(accuracy_avg)

        learning_rate = scheduler.get_last_lr()[0]
        history["learning_rate"].append(learning_rate)
        
        if batch == 1 or batch % accumulation_steps == 0 or batch == len(dataloader_train):
            print(
                f"Epoch [{epoch:0>3}/{num_epochs}] Batch [{batch:0>4}/{len(dataloader_train)}]",
                f"Loss: {loss_batch:.2f} {loss_avg:.2f},",
                f"Accuracy: {accuracy_avg*100:.2f}%,",
                f"LR: {learning_rate:.8f}",
                sep=" ",
                end="\r",
                flush=True
            )
        
        if batch == 1 or batch % 500 == 0 or batch == len(dataloader_train):
            checkpoint(model, optimizer, history, epoch="LAT")
    
    print(end="\n", flush=True)

    if epoch % 16 == 0:
        checkpoint(model, optimizer, history, epoch)
    
    history["avg_epoch_accuracy"].append(history["pred_match"][-1] / history["pred_count"][-1])

timestamp_training_end = datetime.datetime.now()
print(f"* Finished training on {timestamp_training_end}\n")
print(f"* Training time: {timestamp_training_end - timestamp_training_start}\n")
