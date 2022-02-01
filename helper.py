# CIFAR - 10

import pickle
import numpy as np
from keras.datasets import cifar10
from keras.utils import np_utils
from matplotlib import pyplot as plt
import pandas as pd
import requests
from tqdm import tqdm


def perturb_image(xs, img):
    # If this function is passed just one perturbation vector,
    # pack it in a list to keep the computation the same
    if xs.ndim < 2:
        xs = np.array([xs])

    # Copy the image n == len(xs) times so that we can 
    # create n new perturbed images
    tile = [len(xs)] + [1] * (xs.ndim + 1)
    imgs = np.tile(img, tile)

    # Make sure to floor the members of xs as int types
    xs = xs.astype(int)

    for x, img in zip(xs, imgs):
        # Split x into an array of 5-tuples (perturbation pixels)
        # i.e., [[x,y,r,g,b], ...]
        pixels = np.split(x, len(x) // 5)
        for pixel in pixels:
            # At each pixel's x,y position, assign its rgb value
            x_pos, y_pos, *rgb = pixel
            img[x_pos, y_pos] = rgb

    return imgs

def perturb_pixels(xs, img):
    # If this function is passed just one perturbation vector,
    # pack it in a list to keep the computation the same
    
    if xs.ndim < 2:
        xs = np.array([xs])
    
    # Copy the image n == len(xs) times so that we can 
    # create n new perturbed images
    tile = [len(xs)] + [1]*(xs.ndim+1)
    imgs = np.tile(img, tile)
    # Make sure to floor the members of xs as int types
    xs = xs.astype(int)
    
    for x,img in zip(xs, imgs):
        # Split x into an array of 5-tuples (perturbation pixels)
        # i.e., [[x,y,r,g,b], ...]
        pixels = np.split(x, len(x) // 9)
        for pixel in pixels:
            x_pos1, y_pos1, red, x_pos2, y_pos2, green, x_pos3, y_pos3, blue= pixel
            img[x_pos1, y_pos1,0] = red
            img[x_pos2, y_pos2,0] = blue
            img[x_pos3, y_pos3,0] = green
    
    return imgs

def plot_image(image, label_true=None, class_names=None, label_pred=None):
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]

    plt.grid()
    plt.imshow(image.astype(np.uint8))

    # Show true and predicted classes
    if label_true is not None and class_names is not None:
        labels_true_name = class_names[label_true]
        if label_pred is None:
            xlabel = "True: " + labels_true_name
        else:
            # Name of the predicted class
            labels_pred_name = class_names[label_pred]

            xlabel = "True: " + labels_true_name + "\nPredicted: " + labels_pred_name

        # Show the class on the x-axis
        plt.xlabel(xlabel)

    plt.xticks([])  # Remove ticks from the plot
    plt.yticks([])
    plt.show()  # Show the plot


def plot_images(images, labels_true, class_names, labels_pred=None,
                confidence=None, titles=None):
    assert len(images) == len(labels_true)

    # Create a figure with sub-plots
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))

    # Adjust the vertical spacing
    hspace = 0.2
    if labels_pred is not None:
        hspace += 0.2
    if titles is not None:
        hspace += 0.2

    fig.subplots_adjust(hspace=hspace, wspace=0.0)

    for i, ax in enumerate(axes.flat):
        # Fix crash when less than 9 images
        if i < len(images):
            # Plot the image
            ax.imshow(images[i])

            # Name of the true class
            labels_true_name = class_names[labels_true[i]]

            # Show true and predicted classes
            if labels_pred is None:
                xlabel = "True: " + labels_true_name
            else:
                # Name of the predicted class
                labels_pred_name = class_names[labels_pred[i]]

                xlabel = "True: " + labels_true_name + "\nPred: " + labels_pred_name
                if (confidence is not None):
                    xlabel += " (" + "{0:.1f}".format(confidence[i] * 100) + "%)"

            # Show the class on the x-axis
            ax.set_xlabel(xlabel)

            if titles is not None:
                ax.set_title(titles[i])

        # Remove ticks from the plot
        ax.set_xticks([])
        ax.set_yticks([])

    # Show the plot
    plt.show()


def plot_model(model_details):
    # Create sub-plots
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # Summarize history for accuracy
    axs[0].plot(range(1, len(model_details.history['acc']) + 1), model_details.history['acc'])
    axs[0].plot(range(1, len(model_details.history['val_acc']) + 1), model_details.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1, len(model_details.history['acc']) + 1), len(model_details.history['acc']) / 10)
    axs[0].legend(['train', 'val'], loc='best')

    # Summarize history for loss
    axs[1].plot(range(1, len(model_details.history['loss']) + 1), model_details.history['loss'])
    axs[1].plot(range(1, len(model_details.history['val_loss']) + 1), model_details.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1, len(model_details.history['loss']) + 1), len(model_details.history['loss']) / 10)
    axs[1].legend(['train', 'val'], loc='best')

    # Show the plot
    plt.show()


def visualize_attack(df, class_names):
    _, (x_test, _) = cifar10.load_data()

    results = df[df.success].sample(9)

    z = zip(results.perturbation, x_test[results.image])
    images = np.array([perturb_image(p, img)[0]
                       for p, img in z])

    labels_true = np.array(results.true)
    labels_pred = np.array(results.predicted)
    titles = np.array(results.model)

    # Plot the first 9 images.
    plot_images(images=images,
                labels_true=labels_true,
                class_names=class_names,
                labels_pred=labels_pred,
                titles=titles)


def attack_stats(df, models, network_stats):
    stats = []
    for model in models:
        val_accuracy = np.array(network_stats[network_stats.name == model.name].accuracy)[0]
        m_result = df[df.model == model.name]
 
        pixels = list(set(m_result.pixels))
        
        for pixel in pixels:
            
            p_result = m_result[m_result.pixels == pixel]
            success_rate = len(p_result[p_result.success]) / len(p_result)
            stats.append([model.name, val_accuracy, pixel, success_rate])

    return pd.DataFrame(stats, columns=['model', 'accuracy', 'pixels', 'attack_success_rate'])

def evaluate_models(models, x_test, y_test):
    correct_imgs = []
    network_stats = []
    for model in models:
        print('Evaluating', model.name)

        predictions = model.predict(x_test)

        correct = [[model.name, i, label, np.max(pred), pred]
                   for i, (label, pred)
                   in enumerate(zip(y_test[:, 0], predictions))
                   if label == np.argmax(pred)]
        accuracy = len(correct) / len(x_test)

        correct_imgs += correct
        network_stats += [[model.name, accuracy, model.count_params()]]
    return network_stats, correct_imgs

import os.path

def load_results(targeted):
    if (targeted==False):
        with open('networks/results/untargeted_results.pkl', 'rb') as file:
            result = pickle.load(file)
    else:
        with open('networks/results/targeted_results.pkl', 'rb') as file:
            result = pickle.load(file)
    return result

def load_predicted_results():
    with open('networks/results/attack_results.pkl', 'rb') as file:
            result = pickle.load(file)
    return result

def checkpoint(results,model,pixel_count,method, targeted=False):
    filename = 'targeted' if targeted else 'untargeted'
    filename=filename + '_' + str(method) +'_' + str(pixel_count) + '_'+ model.name
    with open('networks/results/' + filename + '_results.pkl', 'wb') as file:
        file.truncate(0)
        pickle.dump(results, file)
# Visualizing of confusion matrix
import seaborn as sn
import pandas  as pd
from sklearn.metrics import classification_report, confusion_matrix

def heatmap(df):
    columns = ['model', 'pixels','method', 'image', 'true', 'predicted', 'success', 'cdiff', 'prior_probs', 'predicted_probs',
               'perturbation','perturbed']
    results_table = pd.DataFrame(df, columns=columns)
    Y_pred = df.predicted
    y_pred = np.argmax(Y_pred, axis=1)
    Y_test = df.true
    y_test=np.argmax(Y_test, axis=1)
    for ix in range(len(y_pred)):
        print(ix, confusion_matrix(np.argmax(y_test,axis=1),y_pred)[ix].sum())
    cm = confusion_matrix(np.argmax(y_test,axis=1),y_pred)
    print(cm)

    df_cm = pd.DataFrame(cm, range(len(y_pred)),
                       range(len(y_pred)))
    plt.figure(figsize = (10,7))
    sn.set(font_scale=1.4)#for label size
    sn.heatmap(df_cm, annot=True,annot_kws={"size": 12})# font size
    plt.show()
                    
def download_from_url(url, dst):
    """
    @param: url to download file
    @param: dst place to put the file
    """
    # Streaming, so we can iterate over the response.
    r = requests.get(url, stream=True)

    with open(dst, 'wb') as f:
        for data in tqdm(r.iter_content(), unit='B', unit_scale=True):
            f.write(data)

# def load_imagenet():
#     with open('data/imagenet_class_index.json', 'r') as f:
#         class_names = json.load(f)
#     class_names = pd.DataFrame([[i,wid,name] for i,(wid,name) in class_names.items()], columns=['id', 'wid', 'text'])

#     wid_to_id = {wid:int(i) for i,wid in class_names[['id', 'wid']].as_matrix()}

#     imagenet_urls = pd.read_csv('data/imagenet_urls.txt', delimiter='\t', names=['label', 'url'])
#     imagenet_urls['label'], imagenet_urls['id'] = zip(*imagenet_urls.label.apply(lambda x: x.split('_')))
#     imagenet_urls.label = imagenet_urls.label.apply(lambda wid: wid_to_id[wid])
