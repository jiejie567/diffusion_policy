from typing import Optional, Dict
import os

class TopKCheckpointManager:
    def __init__(self,
            save_dir,
            monitor_key: str,
            mode='min',
            k=1,
            format_str='epoch={epoch:03d}-train_loss={train_loss:.5f}.ckpt'
        ):
        assert mode in ['max', 'min']
        assert k >= 0

        self.save_dir = save_dir
        self.monitor_key = monitor_key
        self.mode = mode
        self.k = k
        self.format_str = format_str
        self.path_value_map = dict()
        
        # Load existing checkpoints from disk
        self._load_existing_checkpoints()
    
    def _load_existing_checkpoints(self):
        """Load existing checkpoints from disk and track their values."""
        if not os.path.exists(self.save_dir):
            return
        
        try:
            for filename in os.listdir(self.save_dir):
                if filename.endswith('.ckpt') and filename != 'latest.ckpt':
                    filepath = os.path.join(self.save_dir, filename)
                    try:
                        # Extract monitor_key value from filename
                        # E.g., from "epoch=0100-val_loss=0.004.ckpt" extract val_loss value
                        key_prefix = f"{self.monitor_key}="
                        if key_prefix in filename:
                            # Find the value after the key prefix
                            start_idx = filename.find(key_prefix) + len(key_prefix)
                            # Find the end (last '.' before 'ckpt')
                            ckpt_idx = filename.find('.ckpt')
                            if ckpt_idx == -1:
                                continue
                            
                            # Value ends at last '.' before '.ckpt'
                            substring = filename[start_idx:ckpt_idx]
                            # Find last '-' which separates metric values
                            last_dash = substring.rfind('-')
                            if last_dash != -1:
                                # There's another metric after this one
                                value_str = substring[:last_dash]
                            else:
                                # This is the last metric
                                value_str = substring
                            
                            value = float(value_str)
                            self.path_value_map[filepath] = value
                        else:
                            # This checkpoint doesn't match current monitor_key
                            # (e.g., old train_loss checkpoint when monitoring val_loss)
                            # Delete it to keep directory clean
                            if os.path.exists(filepath):
                                os.remove(filepath)
                    except (ValueError, IndexError, AttributeError):
                        # Skip if can't parse the value
                        pass
        except Exception as e:
            # If loading fails, just continue with empty map
            print(f"Warning: Failed to load existing checkpoints: {e}")


    
    def get_ckpt_path(self, data: Dict[str, float]) -> Optional[str]:
        if self.k == 0:
            return None

        value = data[self.monitor_key]
        ckpt_path = os.path.join(
            self.save_dir, self.format_str.format(**data))
        
        if len(self.path_value_map) < self.k:
            # under-capacity
            self.path_value_map[ckpt_path] = value
            return ckpt_path
        
        # at capacity
        sorted_map = sorted(self.path_value_map.items(), key=lambda x: x[1])
        min_path, min_value = sorted_map[0]
        max_path, max_value = sorted_map[-1]

        delete_path = None
        if self.mode == 'max':
            if value > min_value:
                delete_path = min_path
        else:
            if value < max_value:
                delete_path = max_path

        if delete_path is None:
            return None
        else:
            del self.path_value_map[delete_path]
            self.path_value_map[ckpt_path] = value

            if not os.path.exists(self.save_dir):
                os.mkdir(self.save_dir)

            if os.path.exists(delete_path):
                os.remove(delete_path)
            return ckpt_path
