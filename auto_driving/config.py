
class Config(object):
    def __init__(self):
        self.catIds = [0, 33, 34, 35, 36, 38, 39, 40]
        self.catId_map = {0: 0, 33: 1, 34: 2, 35: 3, 36: 4, 38: 5, 39: 6, 40: 7, 255: 0}
        self.cat_names = ['background', 'car', 'motorcycle', 'bicycle', 'pedestrian', 'truck', 'bus', 'tricycle']
        self.crowdCats = []
        self.max_object_count = 100
        self.size = (400, 512)

    @property
    def num_classes(self):
        return len(self.catIds)
