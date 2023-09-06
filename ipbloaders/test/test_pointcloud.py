from test.utils import Base


class PointCloud_Test(Base):
    def __init__(self, methodName):
        super().__init__(methodName, path_to_folder='ipb_loaders/pointcloud/')

    def test_loader(self):
        return super().main_tester()
