from train import traditional_method
import warnings
from seed import setup_seed
warnings.filterwarnings('ignore')

setup_seed(1)
#traditional_method('hog','svm')
traditional_method('sift','svm')