from pathlib import Path

class RunManager:
    def __init__(self, config, accelerator, model, tag, output_root='checkpoints/', scratch_root='/scratch/dthapa/checkpoints/lam/all'):
        self.accelerator = accelerator
        self.model = model
        self.config = config

        self.output_root = output_root
        self.scratch_root = scratch_root

        self.tag = tag
        self.version = self.get_next_version()
        self.best_val_loss = 10000
    
    def get_next_version(self):
        output_base = Path(self.output_root) / self.tag
        output_base.mkdir(parents=True, exist_ok=True)
        existing = [d for d in output_base.iterdir() if d.is_dir() and d.name.startswith("v") and d.name[1:].isdigit()]
        version_numbers = [int(d.name[1:]) for d in existing]
        next_version = max(version_numbers, default=0) + 1

        return f"v{next_version}"

    def update_state(self, **kwargs):
        for key, value in kwargs.items():
            self.config[key] = value

    def save_checkpoint(self, **kwargs):
        self.update_state(**kwargs)
        epoch = self.config['epoch']
        val_loss = self.config['val_loss']

        output_base = Path(self.output_root) / self.tag / self.version
        scratch_base = Path(self.scratch_root) / self.tag / self.version

        if self.accelerator.is_main_process:
            output_base.mkdir(parents=True, exist_ok=True)
            scratch_base.mkdir(parents=True, exist_ok=True)

            # Save latest
            latest_path = output_base / f"latest"
            self.save_checkpoint_(latest_path)

            # Save best only if val_loss is better than existing
            if val_loss < self.best_val_loss:
                print(f'Saving checkpoint at epoch {epoch:03d} with new best val loss:', val_loss)
                best_path = output_base / f"best_val"
                self.save_checkpoint_(best_path)
                self.best_val_loss = val_loss

            # Save epoch checkpoint to scratch
            epoch_ckpt_path = scratch_base / f"epoch_{epoch:03d}_val_{val_loss:.4f}"
            self.save_checkpoint_(epoch_ckpt_path)

            # Update latest symlink
            latest_symlink = output_base.parent / "last_run"
            if latest_symlink.exists() or latest_symlink.is_symlink():
                latest_symlink.unlink()
            latest_symlink.symlink_to(output_base.resolve())
    
    def save_checkpoint_(self, path):
        self.model.save_pretrained(
            path,
            is_main_process=True,
            save_function=self.accelerator.save,
            config=self.config
        )
        self.accelerator.save_state(str(path))
