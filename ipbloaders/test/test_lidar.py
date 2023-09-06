from test.utils import Base


class LiDAR_Test(Base):
    def __init__(self, methodName):
        super().__init__(methodName, path_to_folder='ipb_loaders/lidar/')

    def test_loader(self):
        return super().main_tester()
