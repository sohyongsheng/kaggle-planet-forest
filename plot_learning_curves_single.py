import numpy
import matplotlib.pyplot
import pylab
import sys
    
def plot_learning_curves(epochs, batch_losses, batch_f2_scores, cross_validation_losses, cross_validation_f2_scores, output_directory):
    axes = matplotlib.pyplot.figure().gca()
    x_axis = axes.get_xaxis()
    x_axis.set_major_locator(pylab.MaxNLocator(integer = True))
    
    matplotlib.pyplot.plot(epochs, batch_losses)
    matplotlib.pyplot.plot(epochs, cross_validation_losses)
    matplotlib.pyplot.plot(epochs, batch_f2_scores)
    matplotlib.pyplot.plot(epochs, cross_validation_f2_scores)
    matplotlib.pyplot.legend(['First training batch loss of epoch', 'Cross validation loss', 'First training batch F2 score of epoch', 'Cross validation F2 score'])
    matplotlib.pyplot.xlabel('Epochs')
    matplotlib.pyplot.ylabel('Loss or F2 score')
    matplotlib.pyplot.title('Experiment ' + str(experiment))
    matplotlib.pyplot.xlim((1, 50))
    matplotlib.pyplot.ylim((0.083, 0.084))
    
    matplotlib.pyplot.savefig(output_directory + 'learning_curves.png')
    matplotlib.pyplot.show()
    
# Usage: python3 plot_learning_curves.py <number_of_epochs>, where <number_of_epochs> is the progress of training.    
if __name__ == '__main__':
    experiment = 43
    output_directory = './results/experiment' + str(experiment) + '/learningCurves/'
    epochs = numpy.load(output_directory + 'epochs.npy')
    batch_losses = numpy.load(output_directory + 'batch_losses.npy')
    batch_f2_scores = numpy.load(output_directory + 'batch_f2_scores.npy')
    cross_validation_losses = numpy.load(output_directory + 'cross_validation_losses.npy')
    cross_validation_f2_scores = numpy.load(output_directory + 'cross_validation_f2_scores.npy')

    if len(sys.argv) > 1:
        number_of_epochs = int(sys.argv[1])
        (epochs, batch_losses, batch_f2_scores, cross_validation_losses, cross_validation_f2_scores) = (epochs[: number_of_epochs], batch_losses[: number_of_epochs], batch_f2_scores[: number_of_epochs], cross_validation_losses[: number_of_epochs], cross_validation_f2_scores[: number_of_epochs])
    plot_learning_curves(epochs, batch_losses, batch_f2_scores, cross_validation_losses, cross_validation_f2_scores, output_directory)
    
    training_curves = numpy.column_stack((epochs, batch_losses, cross_validation_losses, batch_f2_scores, cross_validation_f2_scores))
    numpy.savetxt(
        output_directory + 'training_curves.csv', 
        training_curves,
        fmt = '%d, %.5f, %.5f, %.5f, %.5f', 
        header = 'Epochs, Batch losses, Cross validation losses, Batch F2 scores, Cross validation F2 scores'
    )
    
