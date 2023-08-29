import models.modules.MCCDic as MCCDic
####################
# define network
####################
# Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    if which_model == 'MCCDic':
        netG = MCCDic.MCCDic()
        
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    return netG
