from network.detector import Detector
from network.refiner import VolumeRefiner
from network.selector import ViewpointSelector
from network.dino_detector import Detector as DinoDetector
from network.cascade_refiner import VolumeRefiner as cascadeVolumeRefiner

name2network={
    'refiner': VolumeRefiner,
    'detector': Detector,
    'selector': ViewpointSelector,
    'dino_detector':DinoDetector,
    'cascade_refiner':cascadeVolumeRefiner
}


