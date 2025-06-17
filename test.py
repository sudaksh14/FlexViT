import networks.resnetadapt as ra
import utils

device = utils.get_device()
model = ra.Resnet(ra.Config().set_num_classes(100)).to(device)
train_loader, val_loader, test_loader = utils.load_data100()
print(utils.evaluate_model(model, test_loader, device))

model = ra.KNOWN_MODEL_PRETRAINED[(100, (3, 3, 3), (16, 32, 64))]()
print(utils.evaluate_model(model, test_loader, device))
