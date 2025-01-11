import os
import torchaudio
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    def __init__(self, tsv_path):
        self.data = []
        with open(tsv_path, 'r') as file:
            lines = file.readlines()
            self.base_path = lines[0].strip()
            for line in lines[1:]:
                relative_path = line.strip().split('\t')[0]
                audio_path = os.path.join(self.base_path, relative_path)
                label_dir = os.path.dirname(audio_path)
                parts = relative_path.split('/')
                label_file = os.path.join(label_dir, f"{parts[0]}-{parts[1]}.trans.txt")
                self.data.append((audio_path, label_file))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path, label_file = self.data[idx]
        audio, _ = torchaudio.load(audio_path)
        
        # Extract the label for the specific audio file
        base_name = os.path.basename(audio_path).replace('.flac', '')
        label = ""
        with open(label_file, 'r') as lf:
            for line in lf:
                if line.startswith(base_name):
                    label = line[len(base_name):].strip()
                    break
        
        return audio, label

# # Usage example
# if __name__ == "__main__":
#     tsv_path = '/data/yeonahong/librispeech_final/data/test.tsv'
#     dataset = AudioDataset(tsv_path)
    
#     # Example of iterating over the dataset
#     for audio, label in dataset:
#         print(audio.shape, label)
