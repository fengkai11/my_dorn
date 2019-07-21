import utils
import os
from metrics import AverageMeter,Result
from data_loaders.path import Path
from data_loaders.kitti_data_loader import KittiFolder

args = utils.parse_command()
print(args)
if args.gpu:
    print('Single Gpu Model')
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
best_result = Result()
best_result.set_to_worst()

def creat_loader(args):
    root_dir = Path.db_root_dir()
    train_set = KittiFolder(root_dir,mode = 'train',size = ())
    test_set = KittiFolder(root_dir,mode = 'test',size = ())

