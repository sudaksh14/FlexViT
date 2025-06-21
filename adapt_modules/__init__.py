import adapt_modules.adapt_select
import adapt_modules.batchnorm2d
import adapt_modules.conv2d
import adapt_modules.linear
import adapt_modules.module

Module: type[adapt_modules.module.Module] = adapt_modules.module.Module
Conv2d: type[adapt_modules.conv2d.Conv2d] = adapt_modules.conv2d.Conv2d
BatchNorm2d: type[adapt_modules.batchnorm2d.BatchNorm2d] = adapt_modules.batchnorm2d.BatchNorm2d
Linear: type[adapt_modules.linear.Linear] = adapt_modules.linear.Linear
