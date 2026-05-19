"""
Training page.

Trains ResNet-18 on CIFAR-100 with rich real-time feedback:
  - Live progress bar within each epoch (batch-level)
  - Live metrics: loss, accuracy, learning rate, elapsed time, ETA
  - Accuracy chart that updates after every epoch
  - Per-epoch log table
  - Early confirmation after epoch 1 that training is working
  - Checkpoint saves every N epochs and on best accuracy
"""

from __future__ import annotations

import os
import time

import streamlit as st
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

from models.classifier import (
    build_resnet18_cifar100,
    get_device,
    CIFAR100_MEAN,
    CIFAR100_STD,
)
from models.model_state import WEIGHTS_FILE, WEIGHTS_DIR, sidebar_weights_widget
from models.attacks import fgsm_attack, pgd_attack, EPSILON_NORM

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


@st.cache_resource(show_spinner="Downloading CIFAR-100 (~160 MB)...")
def _get_datasets():
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
        T.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    test_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    train_ds = torchvision.datasets.CIFAR100(
        root=DATA_ROOT, train=True,  download=True, transform=train_tf)
    test_ds  = torchvision.datasets.CIFAR100(
        root=DATA_ROOT, train=False, download=True, transform=test_tf)
    return train_ds, test_ds


def _evaluate(model, loader, criterion, device):
    model.eval()
    correct = total = 0
    running_loss = 0.0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out  = model(imgs)
            loss = criterion(out, labels)
            running_loss += loss.item() * imgs.size(0)
            correct      += out.argmax(1).eq(labels).sum().item()
            total        += labels.size(0)
    return 100 * correct / total, running_loss / total


def render():
    st.title("Train Classifier")
    st.markdown(
        "Train a ResNet-18 on CIFAR-100 from scratch. "
        "A live progress bar, loss/accuracy charts, and a per-epoch log "
        "show exactly what is happening at every step."
    )

    sidebar_weights_widget()

    device   = get_device()
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU (slow)"

    c1, c2, c3 = st.columns(3)
    c1.metric("Device",          gpu_name)
    c2.metric("CIFAR-100 classes", "100")
    c3.metric("Target top-1 acc", "~78%")

    st.markdown("---")
    st.subheader("Hyperparameters")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        num_epochs   = st.number_input("Epochs",     min_value=1,   max_value=200, value=100, step=1)
        batch_size   = st.number_input("Batch size", min_value=32,  max_value=512, value=128, step=32)
    with col_b:
        lr           = st.number_input("Learning rate",  min_value=1e-4, max_value=1.0,  value=0.1,  step=0.01, format="%.4f")
        weight_decay = st.number_input("Weight decay",   min_value=0.0,  max_value=1e-2, value=5e-4, step=1e-4, format="%.5f")
    with col_c:
        save_every   = st.number_input("Save checkpoint every N epochs", min_value=1, max_value=50, value=10, step=1)
        label_smooth = st.slider("Label smoothing", min_value=0.0, max_value=0.3, value=0.1, step=0.05)

    if device.type == "cuda":
        st.info(
            f"GPU detected ({gpu_name}). "
            f"Estimated training time: ~{int(num_epochs * 0.45)} minutes for {int(num_epochs)} epochs."
        )
    else:
        st.warning(
            "No GPU detected. Training on CPU is significantly slower. "
            "Consider using 5-10 epochs for a quick demo, or run on a GPU machine."
        )

    if os.path.exists(WEIGHTS_FILE):
        st.warning("Existing weights will be overwritten when training starts.")

    st.markdown("---")
    start_btn = st.button("Start training", type="primary")

    if not start_btn:
        st.info("Configure hyperparameters above, then click **Start training**.")
        return

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    with st.spinner("Preparing CIFAR-100 dataset..."):
        train_ds, test_ds = _get_datasets()

    num_epochs  = int(num_epochs)
    batch_size  = int(batch_size)
    save_every  = int(save_every)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )
    num_batches = len(train_loader)

    st.success(
        f"Dataset ready: {len(train_ds):,} train / {len(test_ds):,} test images  "
        f"({num_batches} batches per epoch)"
    )

    # ------------------------------------------------------------------
    # Model, optimiser, scheduler
    # ------------------------------------------------------------------
    model     = build_resnet18_cifar100().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(label_smooth))
    optimizer = torch.optim.SGD(
        model.parameters(), lr=float(lr),
        momentum=0.9, weight_decay=float(weight_decay), nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-4,
    )
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # UI placeholders
    # ------------------------------------------------------------------
    st.markdown("---")
    st.subheader("Live training progress")

    # Row 1: epoch progress bar + batch progress bar
    epoch_label  = st.empty()
    epoch_bar    = st.progress(0)
    batch_label  = st.empty()
    batch_bar    = st.progress(0)

    # Row 2: live metrics strip
    met_cols     = st.columns(6)
    ph_epoch     = met_cols[0].empty()
    ph_train_loss= met_cols[1].empty()
    ph_test_acc  = met_cols[2].empty()
    ph_best_acc  = met_cols[3].empty()
    ph_lr        = met_cols[4].empty()
    ph_eta       = met_cols[5].empty()

    # Row 3: status / checkpoint message
    status_ph    = st.empty()

    st.markdown("---")
    st.subheader("Accuracy over epochs")

    acc_chart_ph = st.empty()

    st.markdown("---")
    st.subheader("Epoch log")
    log_ph = st.empty()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    history = {
        "epoch":      [],
        "train_loss": [],
        "test_acc":   [],
    }
    best_acc    = 0.0
    train_start = time.time()

    for epoch in range(num_epochs):
        epoch_label.markdown(
            f"**Epoch {epoch + 1} / {num_epochs}**"
        )
        epoch_bar.progress((epoch) / num_epochs)

        model.train()
        train_loss  = 0.0
        train_total = 0
        t_epoch     = time.time()

        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            train_loss  += loss.item() * imgs.size(0)
            train_total += labels.size(0)

            # Update batch progress bar every batch
            frac = (batch_idx + 1) / num_batches
            batch_label.markdown(
                f"Batch **{batch_idx + 1} / {num_batches}** — "
                f"running loss: **{train_loss / train_total:.4f}**"
            )
            batch_bar.progress(frac)

        scheduler.step()
        avg_train_loss = train_loss / train_total

        # Evaluate on test set
        status_ph.info("Evaluating on test set...")
        test_acc, _ = _evaluate(model, test_loader, criterion, device)
        epoch_time = time.time() - t_epoch
        elapsed    = time.time() - train_start
        epochs_left= num_epochs - (epoch + 1)
        eta_s      = epochs_left * epoch_time
        eta_str    = f"{int(eta_s // 60)}m {int(eta_s % 60)}s" if eta_s > 0 else "done"

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(avg_train_loss)
        history["test_acc"].append(test_acc)

        is_best = test_acc > best_acc
        if is_best:
            best_acc = test_acc

        # Save weights
        save_now = (epoch + 1) % save_every == 0 or is_best or epoch == num_epochs - 1
        if save_now:
            torch.save(model.state_dict(), WEIGHTS_FILE)
            if is_best:
                status_ph.success(
                    f"New best accuracy: **{best_acc:.2f}%** — weights saved."
                )
            else:
                status_ph.success(
                    f"Checkpoint saved at epoch {epoch + 1} (test acc: {test_acc:.2f}%)."
                )
        else:
            status_ph.empty()

        # Update live metrics
        ph_epoch.metric("Epoch",       f"{epoch + 1}/{num_epochs}")
        ph_train_loss.metric("Train loss",  f"{avg_train_loss:.4f}")
        ph_test_acc.metric("Test acc",  f"{test_acc:.2f}%")
        ph_best_acc.metric("Best acc",  f"{best_acc:.2f}%")
        ph_lr.metric("LR",             f"{optimizer.param_groups[0]['lr']:.6f}")
        ph_eta.metric("ETA",           eta_str)

        # After epoch 1, show a confirmation that training is actually working
        if epoch == 0:
            if test_acc > 1.5:
                status_ph.success(
                    f"Epoch 1 complete. Test accuracy: **{test_acc:.2f}%** "
                    f"(random chance = 1.0%). Training is working correctly."
                )
            else:
                status_ph.warning(
                    f"Epoch 1 complete. Test accuracy: **{test_acc:.2f}%** — "
                    "this is close to random chance (1%). Check your setup."
                )

        # Update charts
        import pandas as pd
        df = pd.DataFrame({
            "Test accuracy (%)": history["test_acc"],
        }, index=history["epoch"])
        df.index.name = "Epoch"
        acc_chart_ph.line_chart(df, use_container_width=True)

        # Update epoch log
        df_log = pd.DataFrame({
            "Epoch":        history["epoch"],
            "Train loss":   [f"{v:.4f}" for v in history["train_loss"]],
            "Test acc %":   [f"{v:.2f}"  for v in history["test_acc"]],
            "Saved":        [
                "best" if (h == best_acc and h == max(history["test_acc"])) else
                ("ckpt" if (i + 1) % save_every == 0 or i == num_epochs - 1 else "")
                for i, h in enumerate(history["test_acc"])
            ],
        })
        log_ph.dataframe(df_log, use_container_width=True, hide_index=True)

        epoch_bar.progress((epoch + 1) / num_epochs)
        batch_bar.progress(1.0)

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    epoch_label.markdown("**Training complete.**")
    batch_label.empty()
    batch_bar.empty()

    st.success(
        f"Training finished. Best test accuracy: **{best_acc:.2f}%**  \n"
        f"Weights saved to `{WEIGHTS_FILE}`.  \n"
        f"Go to the **Attacker** or **Purifier** page to use the model."
    )
    st.balloons()

    # ------------------------------------------------------------------
    # Post-training pipeline: Attack → Purify → Compare
    # ------------------------------------------------------------------
    import numpy as np
    import pandas as pd
    from models.purification import build_purifier

    st.markdown("---")
    st.subheader("Post-Training Evaluation Pipeline")
    st.markdown(
        "Running the full pipeline on a **500-image test subset**: "
        "**Clean → FGSM attack → Purify → PGD attack → Purify**  \n"
        "ε = 8/255 (standard benchmark)."
    )

    # Fixed 500-image subset (smaller than 1000 so purification stays reasonable)
    rng        = np.random.default_rng(42)
    subset_idx = rng.choice(len(test_ds), 500, replace=False).tolist()
    eval_subset = torch.utils.data.Subset(test_ds, subset_idx)
    eval_loader = DataLoader(
        eval_subset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )

    # Reload best saved weights before every evaluation step
    model.load_state_dict(torch.load(WEIGHTS_FILE, map_location=device))
    model.eval()

    EVAL_EPS_NORM = (8 / 255) / min(CIFAR100_STD)

    # ── Step 1: Clean accuracy ────────────────────────────────────────
    step1_ph = st.empty()
    step1_ph.info("**Step 1/5** — Evaluating clean accuracy…")
    clean_imgs_list, clean_labels_list = [], []
    clean_correct = clean_total = 0
    with torch.no_grad():
        for imgs, labels in eval_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(1)
            clean_correct += preds.eq(labels).sum().item()
            clean_total   += labels.size(0)
            clean_imgs_list.append(imgs.cpu())
            clean_labels_list.append(labels.cpu())
    clean_acc_eval = 100 * clean_correct / clean_total
    all_clean  = torch.cat(clean_imgs_list)
    all_labels = torch.cat(clean_labels_list)
    step1_ph.success(f"**Step 1/5 ✓** — Clean accuracy: **{clean_acc_eval:.2f}%**")

    # ── Step 2: FGSM attack ───────────────────────────────────────────
    step2_ph = st.empty()
    step2_ph.info("**Step 2/5** — Running FGSM attack (5 restarts, ε=8/255)…")
    fgsm_adv_list = []
    fgsm_correct  = fgsm_total = 0
    for imgs, labels in zip(
        all_clean.split(batch_size), all_labels.split(batch_size)
    ):
        imgs, labels = imgs.to(device), labels.to(device)
        x_adv = fgsm_attack(model, imgs, labels,
                             epsilon=EVAL_EPS_NORM, n_restarts=5, device=device)
        with torch.no_grad():
            preds = model(x_adv).argmax(1)
        fgsm_correct  += preds.eq(labels).sum().item()
        fgsm_total    += labels.size(0)
        fgsm_adv_list.append(x_adv.cpu())
    fgsm_acc      = 100 * fgsm_correct / fgsm_total
    all_fgsm_adv  = torch.cat(fgsm_adv_list)
    step2_ph.success(f"**Step 2/5 ✓** — Accuracy after FGSM: **{fgsm_acc:.2f}%**")

    # ── Step 3: Purify FGSM adversarial images ────────────────────────
    step3_ph = st.empty()
    step3_ph.info("**Step 3/5** — Loading MimicDiffusion and purifying FGSM images…  *(this takes a few minutes)*")
    purifier = build_purifier(device, lam=0.8)
    fgsm_pur_list  = []
    fgsm_pur_correct = fgsm_pur_total = 0
    for adv_batch, lbl_batch in zip(
        all_fgsm_adv.split(batch_size), all_labels.split(batch_size)
    ):
        adv_batch = adv_batch.to(device)
        lbl_batch = lbl_batch.to(device)
        with torch.no_grad():
            pur_batch = purifier.purify(adv_batch, t_start=150)
            preds     = model(pur_batch).argmax(1)
        fgsm_pur_correct += preds.eq(lbl_batch).sum().item()
        fgsm_pur_total   += lbl_batch.size(0)
        fgsm_pur_list.append(pur_batch.cpu())
    fgsm_pur_acc = 100 * fgsm_pur_correct / fgsm_pur_total
    step3_ph.success(f"**Step 3/5 ✓** — Accuracy after FGSM + Purify: **{fgsm_pur_acc:.2f}%**")

    # ── Step 4: PGD attack ────────────────────────────────────────────
    step4_ph = st.empty()
    step4_ph.info("**Step 4/5** — Running PGD attack (20 steps, ε=8/255)…")
    pgd_adv_list = []
    pgd_correct  = pgd_total = 0
    for imgs, labels in zip(
        all_clean.split(batch_size), all_labels.split(batch_size)
    ):
        imgs, labels = imgs.to(device), labels.to(device)
        x_adv = pgd_attack(model, imgs, labels,
                            epsilon=EVAL_EPS_NORM, alpha=EVAL_EPS_NORM / 4,
                            steps=20, random_start=True, device=device)
        with torch.no_grad():
            preds = model(x_adv).argmax(1)
        pgd_correct  += preds.eq(labels).sum().item()
        pgd_total    += labels.size(0)
        pgd_adv_list.append(x_adv.cpu())
    pgd_acc     = 100 * pgd_correct / pgd_total
    all_pgd_adv = torch.cat(pgd_adv_list)
    step4_ph.success(f"**Step 4/5 ✓** — Accuracy after PGD: **{pgd_acc:.2f}%**")

    # ── Step 5: Purify PGD adversarial images ─────────────────────────
    step5_ph = st.empty()
    step5_ph.info("**Step 5/5** — Purifying PGD images with MimicDiffusion…  *(this takes a few minutes)*")
    pgd_pur_correct = pgd_pur_total = 0
    for adv_batch, lbl_batch in zip(
        all_pgd_adv.split(batch_size), all_labels.split(batch_size)
    ):
        adv_batch = adv_batch.to(device)
        lbl_batch = lbl_batch.to(device)
        with torch.no_grad():
            pur_batch = purifier.purify(adv_batch, t_start=150)
            preds     = model(pur_batch).argmax(1)
        pgd_pur_correct += preds.eq(lbl_batch).sum().item()
        pgd_pur_total   += lbl_batch.size(0)
    pgd_pur_acc = 100 * pgd_pur_correct / pgd_pur_total
    step5_ph.success(f"**Step 5/5 ✓** — Accuracy after PGD + Purify: **{pgd_pur_acc:.2f}%**")

    # ── Results summary ───────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Results Summary")

    # Metrics strip
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Clean",             f"{clean_acc_eval:.2f}%")
    m2.metric("After FGSM",        f"{fgsm_acc:.2f}%",
              delta=f"{fgsm_acc - clean_acc_eval:.2f}%", delta_color="normal")
    m3.metric("FGSM + Purified",   f"{fgsm_pur_acc:.2f}%",
              delta=f"{fgsm_pur_acc - fgsm_acc:.2f}%", delta_color="normal")
    m4.metric("After PGD",         f"{pgd_acc:.2f}%",
              delta=f"{pgd_acc - clean_acc_eval:.2f}%", delta_color="normal")
    m5.metric("PGD + Purified",    f"{pgd_pur_acc:.2f}%",
              delta=f"{pgd_pur_acc - pgd_acc:.2f}%", delta_color="normal")

    # Summary table
    results_df = pd.DataFrame({
        "Stage": [
            "Clean (no attack)",
            "After FGSM (ε=8/255, 5 restarts)",
            "FGSM → Purified (MimicDiffusion)",
            "After PGD (ε=8/255, 20 steps)",
            "PGD → Purified (MimicDiffusion)",
        ],
        "Accuracy (%)": [
            f"{clean_acc_eval:.2f}",
            f"{fgsm_acc:.2f}",
            f"{fgsm_pur_acc:.2f}",
            f"{pgd_acc:.2f}",
            f"{pgd_pur_acc:.2f}",
        ],
        "vs Clean (%)": [
            "—",
            f"{fgsm_acc     - clean_acc_eval:+.2f}",
            f"{fgsm_pur_acc - clean_acc_eval:+.2f}",
            f"{pgd_acc      - clean_acc_eval:+.2f}",
            f"{pgd_pur_acc  - clean_acc_eval:+.2f}",
        ],
        "Recovery (%)": [
            "—",
            "—",
            f"{fgsm_pur_acc - fgsm_acc:+.2f}",
            "—",
            f"{pgd_pur_acc  - pgd_acc:+.2f}",
        ],
    })
    st.dataframe(results_df, use_container_width=True, hide_index=True)

    # Bar chart comparing all five stages
    chart_df = pd.DataFrame({
        "Accuracy (%)": [
            clean_acc_eval,
            fgsm_acc,
            fgsm_pur_acc,
            pgd_acc,
            pgd_pur_acc,
        ]
    }, index=[
        "Clean",
        "FGSM",
        "FGSM+Purified",
        "PGD",
        "PGD+Purified",
    ])
    st.bar_chart(chart_df, use_container_width=True)

    st.caption(
        "Evaluated on 500 randomly sampled test images (seed=42).  "
        "ε = 8/255 · Purification: MimicDiffusion (t_start=150, λ=0.8, backbone: google/ddpm-cifar10-32)."
    )

    st.cache_resource.clear()
