
from generator.cars_generator import CarsGenerator
from generator.cub_generator import CubGenerator
from generator.custom_generator import CustomGenerator
def get_generator(args):

    if args.dataset == 'cars':
        train_generator = CarsGenerator(root_dir=args.dataset_dir, mode="train",
                                       batch_size=args.batch_size, augment=args.augment)
        val_generator = CarsGenerator(root_dir=args.dataset_dir, mode="valid",
                                     batch_size=args.batch_size, augment=args.augment)
        return train_generator,val_generator
    if args.dataset == 'cub':
        train_generator = CubGenerator(root_dir=args.dataset_dir, mode="train",
                                       batch_size=args.batch_size, augment=args.augment)
        val_generator = CubGenerator(root_dir=args.dataset_dir, mode="valid",
                                     batch_size=args.batch_size, augment=args.augment)
        return train_generator, val_generator
    if args.dataset == 'custom':
        train_generator = CustomGenerator(root_dir=args.dataset_dir, mode="train",
                                       batch_size=args.batch_size, augment=args.augment)
        val_generator = CustomGenerator(root_dir=args.dataset_dir, mode="valid",
                                     batch_size=args.batch_size, augment=args.augment)
        return train_generator, val_generator