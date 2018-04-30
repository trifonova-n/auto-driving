from torch.utils.data import Dataset


class FrameDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.ids = []
        for video_id in range(len(self.data.videos)):
            for frame_id in range(len(self.data.videos[video_id])):
                self.ids.append((video_id, frame_id))

    def __len__(self):
        return self.data.image_count

    def __getitem__(self, idx):
        return self.data.get_frame(*self.ids[idx])
