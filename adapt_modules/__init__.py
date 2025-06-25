import adapt_modules.adapt_select
import adapt_modules.batchnorm2d_select
import adapt_modules.conv2d
import adapt_modules.linear_select
import adapt_modules.module

Module: type[adapt_modules.module.Module] = adapt_modules.module.Module
Conv2d: type[adapt_modules.conv2d.Conv2d] = adapt_modules.conv2d.Conv2d
BatchNorm2dSelect: type[adapt_modules.batchnorm2d_select.BatchNorm2d] = adapt_modules.batchnorm2d_select.BatchNorm2d
LinearSelect: type[adapt_modules.linear_select.Linear] = adapt_modules.linear_select.Linear
