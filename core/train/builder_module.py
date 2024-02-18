import lightning as L


class BuilderModule(L.LightningModule):
    """
    Unified module to create the structure then color it in.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
    
    
