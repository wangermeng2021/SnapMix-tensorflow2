
from model.efficientnet import EfficientNet
from model.resnet import ResNet
def get_model(args,num_class):

    if args.model[0:3] == "Res":
        try:
            depth = int(args.model[-3:])
        except:
            depth = int(args.model[-2:])
        model = ResNet(depth=depth, classes=num_class, concat_max_and_average_pool=args.concat_max_and_average_pool,
                       pretrain=args.pretrain)
    elif args.model[0:3] == "Eff":
        model = EfficientNet(type=args.model[-2:], classes=num_class, concat_max_and_average_pool=args.concat_max_and_averal_pool,
                       pretrain=args.pretrain)
    else:
        raise ValueError("{} is not supported!".format(args.model))
    return model