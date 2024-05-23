import lightning as L

class SimCLR(L.LightningModule):
    """
    A Simple Framework for Contrastive Learning of Visual Representations
    (SimCLR)[1] is a self-supervised contrastive learning framework that uses a
    convolutional neural network (ResNet) to learn representations of images. It
    achieves this by learning to recognise similarities between pairs of
    augmented data points stemming from the same original image (positive pairs)
    and dissimilarities between all other pairs (negative pairs).
    
    This class is used to learn representations of the ACDC dataset, which is
    then used to evaluate the quality of synthetic cardiac segmentation maps. It
    is meant to be an improvement over the FID metric, which does not have good
    performance for evaluating segmentation maps.
    
    [1]: Chen T, Kornblith S, Norouzi M, Hinton G. A simple framework for
    contrastive learning of visual representations. InInternational conference
    on machine learning 2020 Nov 21 (pp. 1597-1607). PMLR.
    """
    
    def __init__(self):
        super().__init__()
        
        self.save_hyperparameters()
        
        print("Hello world!")
        
        import sys
        sys.exit()
