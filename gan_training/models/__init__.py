from gan_training.models import (
    resnet, resnet2,gated_resnet
)

generator_dict = {
    'resnet': resnet.Generator,
    'resnet2': resnet2.Generator,
    'gated_resnet': gated_resnet.Generator,
}

discriminator_dict = {
    'resnet': resnet.Discriminator,
    'resnet2': resnet2.Discriminator,
    'gated_resnet' : gated_resnet.Discriminator,
}
