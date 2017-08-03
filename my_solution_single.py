import numpy
import os.path
import shutil
import pandas
import sklearn.preprocessing
import sklearn.metrics
import PIL.Image
import random
import matplotlib.pyplot
import pylab
import time
import glob

import torch
import torch.utils.data
import torchvision.transforms
import torch.nn
import torch.nn.functional
import torch.optim
import torch.autograd

import resnet

class SubsetSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

class LabelledDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, csv_training_labels_path, image_directory, image_extension, classes_path = None, transform = None, debug = False):
        self.csv_training_labels_path = csv_training_labels_path
        self.image_directory = image_directory
        self.image_extension = image_extension
        self.transform = transform

        training_examples = pandas.read_csv(self.csv_training_labels_path)
        self.training_image_names = training_examples['image_name']
        self.training_labels = training_examples['tags']

        if debug == True:
            self.training_image_names = self.training_image_names.head()
            self.training_labels = self.training_labels.head()

        is_file = lambda training_image_name: os.path.isfile(
            self.image_directory + training_image_name + self.image_extension)
        assert self.training_image_names.apply(
            is_file).all(), "Some training images in " + self.csv_training_labels_path + " are not found."

        self.multi_label_binarizer = sklearn.preprocessing.MultiLabelBinarizer()
        self.training_labels = self.multi_label_binarizer.fit_transform(self.training_labels.str.split())
        self.training_labels = self.training_labels.astype(numpy.float32)
        if classes_path != None:
            classes = self.multi_label_binarizer.classes_
            print('Classes:', list(classes))
            numpy.save(classes_path, classes)
            print('Saved classes to %s.' % classes_path)

    def __getitem__(self, index):
        training_image_path = self.image_directory + self.training_image_names[index] + self.image_extension
        training_image = PIL.Image.open(training_image_path)
        training_image = training_image.convert('RGB')
        if self.transform is not None:
            training_image = self.transform(training_image)
        training_label = torch.from_numpy(self.training_labels[index])

        return (training_image, training_label)

    def __len__(self):
        return len(self.training_image_names.index)
        
class UnlabelledDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, classes_path, image_directories, image_extension, transform = None, debug = False):
        self.transform = transform
        self.image_list = []
        for image_directory in image_directories:
            for filename in glob.glob(image_directory + '*' + image_extension):
                self.image_list.append(filename)
        if debug == True:
            self.image_list = [
                './data/test-jpg/test_0.jpg',
                './data/test-jpg/test_1.jpg',
                './data/test-jpg/test_2.jpg'
            ]
        
    def __getitem__(self, index):        
        filename = self.image_list[index]
        image = PIL.Image.open(filename)
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        
        return (image, filename)
        
    def __len__(self):
        return len(self.image_list)
        
def transpose(image, perturbation_index):
    return {
        1: image,
        2: image.transpose(PIL.Image.FLIP_LEFT_RIGHT),
        3: image.transpose(PIL.Image.FLIP_TOP_BOTTOM),
        4: image.transpose(PIL.Image.ROTATE_90),
        5: image.transpose(PIL.Image.ROTATE_180),
        6: image.transpose(PIL.Image.ROTATE_270),
        # Flip about main diagonal.
        7: image.transpose(PIL.Image.TRANSPOSE),
        # Flip about minor diagonal.
        8: image.transpose(PIL.Image.ROTATE_180).transpose(PIL.Image.TRANSPOSE)
    }[perturbation_index]

def transpose_randomly(image):
    perturbation_index = random.randint(1, 8)
    return {
        1: image,
        2: image.transpose(PIL.Image.FLIP_LEFT_RIGHT),
        3: image.transpose(PIL.Image.FLIP_TOP_BOTTOM),
        4: image.transpose(PIL.Image.ROTATE_90),
        5: image.transpose(PIL.Image.ROTATE_180),
        6: image.transpose(PIL.Image.ROTATE_270),
        # Flip about main diagonal.
        7: image.transpose(PIL.Image.TRANSPOSE),
        # Flip about minor diagonal.
        8: image.transpose(PIL.Image.ROTATE_180).transpose(PIL.Image.TRANSPOSE)
    }[perturbation_index]
    
def rotate_scale_translate(image, angle, scaling_factor, horizontal_shift_factor, vertical_shift_factor):
    width = image.size[0]
    height = image.size[1]

    # Create temporary big image with reflected images surrounding original image.
    (temp_width, temp_height) = (3 * width, 3 * height)
    new_image = PIL.Image.new('RGB', (temp_width, temp_height))
    # First row.
    new_image.paste(image.transpose(PIL.Image.FLIP_TOP_BOTTOM).transpose(PIL.Image.FLIP_LEFT_RIGHT), (0, 0))
    new_image.paste(image.transpose(PIL.Image.FLIP_TOP_BOTTOM), (width, 0))
    new_image.paste(image.transpose(PIL.Image.FLIP_TOP_BOTTOM).transpose(PIL.Image.FLIP_LEFT_RIGHT), (2 * width, 0))
    # Second row.
    new_image.paste(image.transpose(PIL.Image.FLIP_LEFT_RIGHT), (0, height))
    new_image.paste(image, (width, height))
    new_image.paste(image.transpose(PIL.Image.FLIP_LEFT_RIGHT), (2 * width, height))
    # Third row.
    new_image.paste(image.transpose(PIL.Image.FLIP_TOP_BOTTOM).transpose(PIL.Image.FLIP_LEFT_RIGHT), (0, 2 * height))
    new_image.paste(image.transpose(PIL.Image.FLIP_TOP_BOTTOM), (width, 2 * height))
    new_image.paste(image.transpose(PIL.Image.FLIP_TOP_BOTTOM).transpose(PIL.Image.FLIP_LEFT_RIGHT), (2 * width, 2 * height))

    # Rotate, scale, crop centre and resize back to original size.
    new_width = scaling_factor * width
    new_height = scaling_factor * height
    horizontal_shift = horizontal_shift_factor * width
    vertical_shift = vertical_shift_factor * height
    left = ((temp_width - new_width) / 2) + horizontal_shift
    top = ((temp_height - new_height) / 2) + vertical_shift
    right = ((temp_width + new_width) / 2) + horizontal_shift
    bottom = ((temp_height + new_height) / 2) + vertical_shift
    cropping_box = (int(round(left)), int(round(top)), int(round(right)), int(round(bottom)))
    new_image = new_image.rotate(angle, resample = PIL.Image.BILINEAR).crop(cropping_box).resize((width, height))

    return new_image
    
def rotate_scale_translate_randomly(image):
    # Don't perturb so much for augmentation.
    if random.random() < 0.5:
        angle = random.uniform(-45, 45)
        scaling_factor = random.uniform(0.85, 1.15)
        horizontal_shift_factor = random.uniform(-0.0625, 0.0625)
        vertical_shift_factor = random.uniform(-0.0625, 0.0625)
        new_image = rotate_scale_translate(image, angle, scaling_factor, horizontal_shift_factor, vertical_shift_factor)
    else:
        new_image = image

    return new_image

def get_train_and_cross_validation_indices(number_of_examples, number_of_cross_validation_examples, cross_validation_fold):
    all_cross_validation_indices = []
    all_train_indices = []

    for i in range(cross_validation_fold):
        start = i * number_of_cross_validation_examples
        if i == cross_validation_fold - 1:
            end = number_of_examples
        else:
            end = start + number_of_cross_validation_examples
        all_cross_validation_indices.append(list(range(start, end)))
        
    for (i, cross_validation_indices) in enumerate(all_cross_validation_indices):
        print('Cross validation start, end:', (cross_validation_indices[0], cross_validation_indices[-1]))
        start = 0
        end = cross_validation_indices[0]
        first_part_of_train_indices = list(range(start, end))
            
        start = cross_validation_indices[-1] + 1
        end = number_of_examples
        second_part_of_train_indices = list(range(start, end))
        
        train_indices = first_part_of_train_indices + second_part_of_train_indices    
        all_train_indices.append(train_indices)
    
    return (all_cross_validation_indices, all_train_indices)
    
def print_image(image, output_file_name):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage()
    ])
    image = transform(image)
    image.save(output_file_name)

def compute_f2_score(threshold, targets, outputs):
    labels = targets.data.cpu().numpy().astype('int')
    predictions = (outputs.data.cpu().numpy() > threshold).astype('int')
    f2_score = sklearn.metrics.fbeta_score(labels, predictions, 2, average = 'samples')
    
    return f2_score

def cross_validate(cross_validation_loader, threshold = 0.5, print_images = False):
    model.eval()
    number_of_examples = 0
    cross_validation_loss = 0
    cross_validation_f2_score = 0
    for (i, (inputs, targets)) in enumerate(cross_validation_loader):
        if print_images == True:
            for (j, image) in enumerate(inputs):
                output_file_name = './debugImages/crossValidationImages/' + 'epoch' + str(epoch) + '_' + 'batch' + str(i) + '_' + 'image' + str(j) + '.png'
                print_image(image, output_file_name)
        
        # Feed forward.
        batch_size = len(inputs)
        number_of_examples = number_of_examples + batch_size
        (inputs, targets) = (inputs.cuda(), targets.cuda())
        (inputs, targets) = (torch.autograd.Variable(inputs, volatile = True), torch.autograd.Variable(targets, volatile = True))
        outputs = model(inputs)
        
        # Accumulate cross validation batch F2 scores.
        batch_f2_score = compute_f2_score(threshold, targets, outputs)
        cross_validation_f2_score = cross_validation_f2_score + (batch_size * batch_f2_score)
        # Accumulate cross validation batch losses.
        loss = criterion(outputs, targets)
        batch_loss = loss.data[0]
        cross_validation_loss = cross_validation_loss + (batch_size * batch_loss)

    # Average F2 scores and cross validation losses.
    cross_validation_loss = cross_validation_loss / number_of_examples
    cross_validation_f2_score = cross_validation_f2_score / number_of_examples    
    print('%d cross validation examples, average cross validation loss = %f, F2 score = %f' % (number_of_examples, cross_validation_loss, cross_validation_f2_score))
    
    return (cross_validation_loss, cross_validation_f2_score)

def train_epoch(epoch, train_loader, cross_validation_loader, threshold = 0.5, print_images = False):
    first_batch_loss = 0
    first_batch_f2_score = 0

    model.train()
    for (i, (inputs, targets)) in enumerate(train_loader):
        if print_images == True:
            for (j, image) in enumerate(inputs):
                output_file_name = './debugImages/trainImages/' + 'epoch' + str(epoch) + '_' + 'batch' + str(i) + '_' + 'image' + str(j) + '.png'
                print_image(image, output_file_name)

        # Feed-forward.
        (inputs, targets) = (inputs.cuda(), targets.cuda())
        (inputs, targets) = (torch.autograd.Variable(inputs), torch.autograd.Variable(targets))
        outputs = model(inputs)
        
        loss = criterion(outputs, targets)
        # Back-propagate and update parameters.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_loss = loss.data[0]
        is_first_batch = (i == 0)
        if is_first_batch:
            first_batch_loss = batch_loss
            first_batch_f2_score = compute_f2_score(threshold, targets, outputs)
        print('Epoch %d iteration %d: loss = %f' % (epoch + 1, i + 1, batch_loss))

    return (first_batch_loss, first_batch_f2_score)
    
# First <number_of_cross_validation_examples> labelled examples will be cross validation examples.
def train(number_of_cross_validation_examples, learning_rate_schedule, save_model_interval = 1, debug = False):
    print('Learning rate schedule:', learning_rate_schedule)
    global model
    
    # Set up train loader.
    transform_with_augmentation = torchvision.transforms.Compose([
        torchvision.transforms.Lambda(lambda image: transpose_randomly(image)),
        torchvision.transforms.Lambda(lambda image: rotate_scale_translate_randomly(image)),
        torchvision.transforms.ToTensor(),
    ])
    train_set = LabelledDataset(training_examples_file_name, training_image_directory, training_image_extension, classes_path = classes_path, transform = transform_with_augmentation)
    # Determine train and cross validation indices.
    cross_validation_indices = list(range(0, number_of_cross_validation_examples))
    train_indices = list(range(number_of_cross_validation_examples, len(train_set)))
    if debug == True:
        cross_validation_indices = list(range(0, 5))
        train_indices = list(range(5, 10))
    print('Train indices:', min(train_indices), 'to', max(train_indices))
    print('Cross validation indices:', min(cross_validation_indices), 'to', max(cross_validation_indices))
    size_of_first_input = train_set[0][0].size()
    size_of_first_label = train_set[0][1].size()
    print('Size of first labelled input:', size_of_first_input)
    print('Size of first label:', size_of_first_label)
    number_of_classes = size_of_first_label[0]
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = 128, sampler = train_sampler, num_workers = 4, pin_memory = True)
    
    # Set up cross validation loader.
    transform_without_augmentation = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    cross_validation_set = LabelledDataset(training_examples_file_name, training_image_directory, training_image_extension, transform = transform_without_augmentation)
    cross_validation_sampler = SubsetSampler(cross_validation_indices)
    cross_validation_loader = torch.utils.data.DataLoader(cross_validation_set, batch_size = 128, sampler = cross_validation_sampler, num_workers = 4, pin_memory = True)
    
    # Set up data structures to hold learning curves and F2 scores.
    batch_losses = numpy.empty(number_of_epochs)
    batch_f2_scores = numpy.empty(number_of_epochs)
    cross_validation_losses = numpy.empty(number_of_epochs)
    cross_validation_f2_scores = numpy.empty(number_of_epochs)
    epochs = numpy.arange(number_of_epochs) + 1
    
    # Load pre-trained model.
    pretrained_model_file = './resnet34_pretrained.pth'
    pretrained_dict = torch.load(pretrained_model_file)
    print('Load pre-trained model:', pretrained_model_file)
    load_valid(model, pretrained_dict, skip_list = ['fc.weight', 'fc.bias'])
    
    # Load previous model and optimizer snapshots, as well as learning curves.
    if start_epoch != 0:
        model.load_state_dict(torch.load(models_directory + 'epoch' + str(start_epoch)))
        optimizer.load_state_dict(torch.load(optimizers_directory + 'epoch' + str(start_epoch)))
        
        batch_losses[0: start_epoch] = numpy.load(learning_curves_directory + 'batch_losses.npy')[0: start_epoch]
        batch_f2_scores[0: start_epoch] = numpy.load(learning_curves_directory + 'batch_f2_scores.npy')[0: start_epoch]
        cross_validation_losses[0: start_epoch] = numpy.load(learning_curves_directory + 'cross_validation_losses.npy')[0: start_epoch]
        cross_validation_f2_scores[0: start_epoch] = numpy.load(learning_curves_directory + 'cross_validation_f2_scores.npy')[0: start_epoch]
    
    torch.manual_seed(123)
    for epoch in range(start_epoch, number_of_epochs):
        start_time = time.time()
        # TODO Change to PyTorch's own scheduler class in the next version of PyTorch. We can only change the learning rate the following way in this current version of PyTorch.
        if epoch in learning_rate_schedule:
            set_learning_rate(optimizer, learning_rate_schedule[epoch])

        # Train and evaluate losses.
        print('Epoch %d: learning rate = %f' % (epoch + 1, optimizer.param_groups[0]['lr']))
        (batch_losses[epoch], batch_f2_scores[epoch]) = train_epoch(epoch, train_loader, cross_validation_loader)
        (cross_validation_losses[epoch], cross_validation_f2_scores[epoch]) = cross_validate(cross_validation_loader)
        
        # Save learning curves and snapshots.
        numpy.save(learning_curves_directory + 'epochs.npy', epochs)
        numpy.save(learning_curves_directory + 'batch_losses.npy', batch_losses)
        numpy.save(learning_curves_directory + 'batch_f2_scores.npy', batch_f2_scores)
        numpy.save(learning_curves_directory + 'cross_validation_losses.npy', cross_validation_losses)
        numpy.save(learning_curves_directory + 'cross_validation_f2_scores.npy', cross_validation_f2_scores)
        if epoch % save_model_interval == (save_model_interval - 1):
            torch.save(model.state_dict(), models_directory + 'epoch' + str(epoch + 1))
            torch.save(optimizer.state_dict(), optimizers_directory + 'epoch' + str(epoch + 1))
        end_time = time.time()
        time_elapsed = (end_time - start_time) / 60
        print('Epoch %d: time taken = %.2f minutes' % (epoch + 1, time_elapsed))
            
    print('Finished training.')
    print('First batch losses:', batch_losses)
    print('Batch F2 scores:', batch_f2_scores)
    print('Cross validation losses:', cross_validation_losses)
    print('Cross validation F2 scores:', cross_validation_f2_scores)
    torch.save(model.state_dict(), models_directory + 'epoch' + str(number_of_epochs))
    torch.save(optimizer.state_dict(), optimizers_directory + 'epoch' + str(number_of_epochs))
    
def predict_labelled(dataset_loader, perturbation_index, threshold = 0.5, print_images = False, print_image_directory = None):
    model.eval()
    all_labels = numpy.empty((0, number_of_classes))
    all_predictions = numpy.empty((0, number_of_classes))
    
    for (i, (inputs, targets)) in enumerate(dataset_loader):
        if print_images == True:
            for (j, image) in enumerate(inputs):
                if i == 0 and j == 3:
                    output_file_name = print_image_directory + ('perturbation%d_batch%d_image%d' % (perturbation_index, i, j) + '.png')
                    print_image(image, output_file_name)
        print('Feedforward for batch:', i + 1)
        (inputs, targets) = (inputs.cuda(), targets.cuda())
        (inputs, targets) = (torch.autograd.Variable(inputs, volatile = True), torch.autograd.Variable(targets, volatile = True))
        outputs = model(inputs)
        
        labels = targets.data.cpu().numpy()
        predictions = outputs.data.cpu().numpy()
        all_labels = numpy.vstack((all_labels, labels))
        all_predictions = numpy.vstack((all_predictions, predictions))
            
    return (all_labels, all_predictions)
    
def predict_unlabelled(dataset_loader, perturbation_index, threshold = 0.5, print_images = False):
    model.eval()
    all_labels = numpy.empty((0, number_of_classes))
    all_predictions = numpy.empty((0, number_of_classes))
    
    all_file_names = []
    for (i, (inputs, file_names)) in enumerate(dataset_loader):
        if print_images == True:
            for (j, image) in enumerate(inputs):
                if i == 0 and j in list(range(3)):
                    output_file_name = './debugImages/testImages/' + ('perturbation%d_batch%d_image%d' % (perturbation_index, i, j) + '.png')
                    print_image(image, output_file_name)
                
        print('Predicting batch %d.' % (i + 1))
        inputs = inputs.cuda()
        inputs = torch.autograd.Variable(inputs, volatile = True)
        outputs = model(inputs)
        
        predictions = outputs.data.cpu().numpy()
        all_predictions = numpy.vstack((all_predictions, predictions))
        all_file_names.extend(file_names)
        
    return (all_file_names, all_predictions)
    
# Find best F2 score based on 17 thresholds, one for each class.
# Code from anokas:
# https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/discussion/32475
import numpy as np
from sklearn.metrics import fbeta_score

def optimise_f2_thresholds(y, p, verbose = True, resolution = 100):
  def mf(x):
    p2 = np.zeros_like(p)
    for i in range(17):
      p2[:, i] = (p[:, i] > x[i]).astype(np.int)
    score = fbeta_score(y, p2, beta=2, average='samples')
    return score

  x = [0.2]*17
  for i in range(17):
    best_i2 = 0
    best_score = 0
    for i2 in range(resolution):
      i2 /= resolution
      x[i] = i2
      score = mf(x)
      if score > best_score:
        best_i2 = i2
        best_score = score
    x[i] = best_i2
    if verbose:
      print(i, best_i2, best_score)

  return x

# Find best F2 score based on fixed threshold.
def optimise_f2_threshold(labels, predictions, predictions_directory):
    thresholds = numpy.arange(0, 1, 0.01)
    f2_scores = numpy.empty(len(thresholds))
    for (i, threshold) in enumerate(thresholds):
        f2_score = sklearn.metrics.fbeta_score(
            labels.astype('int'), 
            (predictions > threshold).astype('int'), 
            2, average = 'samples'
        )
        f2_scores[i] = f2_score
        print('Threshold', threshold, ': F2 score =', f2_score)
    numpy.save(predictions_directory + 'thresholds.npy', thresholds)
    numpy.save(predictions_directory + 'f2_scores.npy', f2_scores)
    
    max_f2_score = numpy.amax(f2_scores)
    corresponding_threshold = thresholds[numpy.argmax(f2_scores)]
    print('Max F2 score:', max_f2_score)
    print('Corresponding threshold:', corresponding_threshold)
    
    return corresponding_threshold
  
def set_learning_rate(optimizer, learning_rate):
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate
        
def load_valid(model, pretrained_dict, skip_list=[]):
    model_dict = model.state_dict()
    to_load_dict = {
        description: weights for (description, weights) in pretrained_dict.items() if (
            (description in model_dict) and (description not in skip_list)
        )
    }
    print('Missing keys:')
    for k in model_dict.keys():
        if k not in to_load_dict.keys():
            print(k)
    model_dict.update(to_load_dict)
    model.load_state_dict(model_dict)
    
def find_thresholds(epoch_of_chosen_model, threshold_indices, already_predicted = False, debug = False):
    if debug == True:
        threshold_indices = list(range(0, 5))
    print('Threshold indices:', min(threshold_indices), 'to', max(threshold_indices))
    global model
    
    model.load_state_dict(torch.load(models_directory + 'epoch' + str(epoch_of_chosen_model)))
    model = model.cuda()
    predictions_directory = output_directory + 'predictions/crossValidationSet/'
    number_of_perturbations = 8
    all_predictions = []
    all_labels = []
    
    if already_predicted == False:
        for perturbation_index in range(1, number_of_perturbations + 1):
            print('Perturbation index:', perturbation_index)
            # Perform TTA on cross validation set, which we call thresholds set here.
            transform_with_transpose_augmentation = torchvision.transforms.Compose([
                torchvision.transforms.Lambda(lambda image: transpose(image, perturbation_index)),
                torchvision.transforms.ToTensor(),
            ])
            thresholds_set = LabelledDataset(training_examples_file_name, training_image_directory, training_image_extension, transform = transform_with_transpose_augmentation)
            thresholds_sampler = SubsetSampler(threshold_indices)
            thresholds_loader = torch.utils.data.DataLoader(thresholds_set, batch_size = 128, sampler = thresholds_sampler, num_workers = 4, pin_memory = True)
            
            # Predict and save probabilistic outputs.
            (labels, predictions) = predict_labelled(thresholds_loader, perturbation_index, print_images = False, print_image_directory = './debugImages/crossValidationImages/')
            if perturbation_index == 1:
                numpy.save(predictions_directory + 'labels.npy', labels)
            numpy.save(predictions_directory + ('predictions_perturbation%d.npy' % perturbation_index), predictions)
            all_labels.append(labels)
            all_predictions.append(predictions)
        # Combine predictions from various TTAs by averaging.
        all_predictions = numpy.stack(all_predictions, axis = 0)
        combined_predictions = numpy.mean(all_predictions, axis = 0)
        numpy.save(predictions_directory + 'predictions.npy', combined_predictions)

    # Optimize thresholds.
    optimized_thresholds_file = predictions_directory + 'optimized_thresholds.npy'
    predictions_file = predictions_directory + 'predictions.npy'
    labels = numpy.load(predictions_directory + 'labels.npy')
    predictions = numpy.load(predictions_file)
    # Uncomment following line to use fixed thresholds, and comment out next line as appropriate.
    # optimized_thresholds = optimise_f2_threshold(labels, predictions, predictions_directory)
    optimized_thresholds = optimise_f2_thresholds(labels, predictions, verbose = True, resolution = 100)
    print('Optimized thresholds:', optimized_thresholds)
    numpy.save(optimized_thresholds_file, optimized_thresholds)
    print('Saved optimized thresholds to %s.' % optimized_thresholds_file)
    
    return optimized_thresholds_file

def test(epoch_of_chosen_model, threshold = 0.5, already_predicted = False):
    model.load_state_dict(torch.load(models_directory + 'epoch' + str(epoch_of_chosen_model)))
    number_of_perturbations = 8
    predictions_directory = output_directory + 'predictions/testSet/'
    
    # Save predictions for each kind of test augmentation.
    if already_predicted == False:
        for perturbation_index in range(1, number_of_perturbations + 1):
            print('Perturbation index:', perturbation_index)
            transform_with_transpose_augmentation = torchvision.transforms.Compose([
                torchvision.transforms.Lambda(lambda image: transpose(image, perturbation_index)),
                torchvision.transforms.ToTensor(),
            ])
            test_set = UnlabelledDataset(classes_path, test_image_directories, test_image_extension, transform = transform_with_transpose_augmentation, debug = False)
            print('Number of test images:', len(test_set))
            test_sampler = torch.utils.data.sampler.SequentialSampler(test_set)
            test_loader = torch.utils.data.DataLoader(test_set, batch_size = 256, sampler = test_sampler, num_workers = 4, pin_memory = True)
            (all_file_names, all_predictions) = predict_unlabelled(test_loader, perturbation_index, print_images = False)
            # Save list of file names only once.
            if perturbation_index == 1:
                with open(predictions_directory + 'file_names.txt','w') as f:
                    for filename in all_file_names:
                        f.write('%s\n' % filename)
            numpy.save(predictions_directory + ('predictions_perturbation%d.npy' % perturbation_index), all_predictions)
        
    # Combine all predictions and save overall result.
    all_predictions = []
    for perturbation_index in range(1, number_of_perturbations + 1):
        predictions = numpy.load(predictions_directory + ('predictions_perturbation%d.npy' % perturbation_index))
        all_predictions.append(predictions)
    all_predictions = numpy.stack(all_predictions, axis = 0)
    combined_predictions = numpy.mean(all_predictions, axis = 0)
    numpy.save(predictions_directory + 'predictions.npy', combined_predictions)

    # Write submission file.
    predictions_file = predictions_directory + 'predictions.npy'
    predictions = numpy.load(predictions_file)
    predictions = (predictions > threshold).astype('int')
    multi_label_binarizer = sklearn.preprocessing.MultiLabelBinarizer()
    classes = numpy.load(classes_path)
    multi_label_binarizer.fit([list(classes)])
    print('Load and fit classes into multi-label binarizer:', multi_label_binarizer.classes_)
    predictions = multi_label_binarizer.inverse_transform(predictions)
    with open(predictions_directory + 'file_names.txt') as f:
        file_names = f.readlines()
        file_names = [file_name.strip() for file_name in file_names] 
    
    submission_file = output_directory + 'submission.csv'
    with open(submission_file,'w') as f:
        f.write('image_name,tags\n')
        print('Number of file names:', len(file_names))
        print('Number of predictions:', len(predictions))
        for i in range (0, len(predictions)):
            file_name = file_names[i]
            file_name = file_name.split('/')[-1].replace('.jpg','')
            f.write(file_name + ',' + ' '.join(predictions[i]) + '\n')
    print('Submission file written to %s.' % submission_file)

if __name__ == '__main__':
    # Use this to select GPU for training. Put gpu_index = 0 if you only have 1 GPU, or want to use the first GPU.
    gpu_index = 0
    torch.cuda.set_device(gpu_index)
    
    training_examples_file_name = './data/train.csv'
    training_image_directory = './data/train-jpg/'
    training_image_extension = '.jpg'
    test_image_directories = ['./data/test-jpg/', './data/test-jpg-additional/']
    test_image_extension = training_image_extension
    # Index your experiment.
    experiment = 43
    output_directory = './results/experiment' + str(experiment) + '/'
    template_output_directory = './results/experiment_template'
    if not os.path.exists(output_directory):
        print('Creating', output_directory)
        shutil.copytree(template_output_directory, output_directory)
    learning_curves_directory = output_directory + 'learningCurves/'
    snapshots_directory = output_directory + 'snapshots/'
    models_directory = snapshots_directory + 'models/'
    optimizers_directory = snapshots_directory + 'optimizers/'
    classes_path = output_directory + 'classes.npy'
    number_of_cross_validation_examples = 8000
    (start_epoch, number_of_epochs) = (0, 50)
    number_of_classes = 17
    model = resnet.resnet34(in_shape = (3, 256, 256), num_classes = number_of_classes).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr = 1e-2, momentum = 0.9, weight_decay = 1e-4)
    criterion = torch.nn.BCELoss().cuda()
    # Dictionary with epochs as keys and learning rates as values.
    learning_rate_schedule = {
        0: 1e-2, 
        20: 1e-3, 
    }
    
    to_train = True
    to_find_thresholds = True
    to_test = True
    if to_train:
        print('Phase: Train.')
        train(number_of_cross_validation_examples, learning_rate_schedule, save_model_interval = 5, debug = False)
    # Choose epoch for early stopping after examining learning curves.
    epoch_of_chosen_model = 30
    if to_find_thresholds:
        print('Phase: Optimize thresholds.')
        cross_validation_indices = list(range(0, number_of_cross_validation_examples))
        optimized_thresholds_file = find_thresholds(epoch_of_chosen_model, cross_validation_indices, already_predicted = False, debug = False)
    if to_test:
        print('Phase: Test.')
        if to_find_thresholds == False:
            # Manually point to thresholds file.
            optimized_thresholds_file = output_directory + 'predictions/crossValidationSet/' + 'optimized_thresholds.npy'
        test(epoch_of_chosen_model, threshold = numpy.load(optimized_thresholds_file), already_predicted = False)

