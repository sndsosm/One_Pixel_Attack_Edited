import argparse

import numpy as np
import pandas as pd
from keras.datasets import cifar10
import pickle

# Custom Networks
from networks.lenet import LeNet
from networks.pure_cnn import PureCnn
from networks.network_in_network import NetworkInNetwork
from networks.resnet import ResNet


# Helper functions
from differential_evolution import differential_evolution
import helper
from scipy.optimize import dual_annealing, basinhopping
import random
from random import randint 

class PixelAttacker:
    def __init__(self, base, models, data, class_names, dimensions=(32, 32)):
        # Load data and model
        self.models=base
        self.all_models = models
        self.x_test, self.y_test = data
        self.class_names = class_names
        self.dimensions = dimensions

        network_stats, correct_imgs = helper.evaluate_models(self.all_models, self.x_test, self.y_test)
        self.correct_imgs = pd.DataFrame(correct_imgs, columns=['name', 'img', 'label', 'confidence', 'pred'])
        self.network_stats = pd.DataFrame(network_stats, columns=['name', 'accuracy', 'param_count'])

    def predict_classes(self, xs, img, target_class, model, minimize=True):
        # Perturb the image with the given pixel(s) x and get the prediction of the model
        imgs_perturbed = helper.perturb_image(xs, img)
        predictions = model.predict(imgs_perturbed)[:, target_class]
        # This function should always be minimized, so return its complement if needed
        return predictions if minimize else 1 - predictions

    def predict_classe(self,xs, img, target_class, model, minimize=True):
      # Perturb the image with the given pixel(s) x and get the prediction of the model
      imgs_perturbed = helper.perturb_pixels(xs, img)
      predictions = model.predict(imgs_perturbed)[:,target_class]
      # This function should always be minimized, so return its complement if needed
      return predictions if minimize else 1 - predictions
    
    def attack_success(self, x, img, target_class, model, targeted_attack=False, verbose=False):
        # Perturb the image with the given pixel(s) and get the prediction of the model
        attack_image = helper.perturb_image(x, img)

        confidence = model.predict(attack_image)[0]
        predicted_class = np.argmax(confidence)

        # If the prediction is what we want (misclassification or 
        # targeted classification), return True
        if verbose:
            print('Confidence:', confidence[target_class])
        if ((targeted_attack and predicted_class == target_class) or
                (not targeted_attack and predicted_class != target_class)):
            return True
            
    def attack_successes(self,x,img, target_class, model, targeted_attack=False, verbose=False):
      # Perturb the image with the given pixel(s) and get the prediction of the model
      attack_image = helper.perturb_pixels(x,img)

      confidence = model.predict(attack_image)[0]
      predicted_class = np.argmax(confidence)

      # If the prediction is what we want (misclassification or 
      # targeted classification), return True
      if verbose:
          print('Confidence:', confidence[target_class])
      if ((targeted_attack and predicted_class == target_class) or
          (not targeted_attack and predicted_class != target_class)):
          return True
        
    def attack(self, img_id, model, target=None, pixel_count=1,
               method='DE',maxiter=250, popsize=500,tempertature=5230, T=1.0, verbose=False, plot=False):
        # Change the target class based on whether this is a targeted attack or not
        print("Exectuting original attack")
        targeted_attack = target is not None
        target_class = target if targeted_attack else self.y_test[img_id, 0]

        # Define bounds for a flat vector of x,y,r,g,b values
        # For more pixels, repeat this layout
        dim_x, dim_y = self.dimensions
                
        def predict_fn(xs):
            return self.predict_classes(xs, self.x_test[img_id], target_class, model, target is None)

        def callback_fn(x, convergence):
            return self.attack_success(x, self.x_test[img_id], target_class, model, targeted_attack, verbose)

        if (method=='DE'):
          bounds = [(0, dim_x), (0, dim_y), (0, 256), (0, 256), (0, 256)] * pixel_count
          # Population multiplier, in terms of the size of the perturbation vector x
          popmul = max(1, popsize // len(bounds))
          attack_result = differential_evolution(predict_fn, bounds, maxiter=maxiter, popsize=popmul,recombination=1,callback=callback_fn, atol=-1, polish=False)
        elif (method=='DA'):
          bounds = bounds = [(0, dim_x), (0, dim_y), (0, 256), (0, 256), (0, 256)] * pixel_count
          attack_result =dual_annealing(predict_fn, bounds, maxiter=maxiter, intital_temp=temperature)
        elif (method=='BH'):
          bounds = [(0, dim_x), (0, dim_y), (0, 255), (0, 255), (0, 255)] * pixel_count
          minimizer_kwargs = { "method": "L-BFGS-B","bounds":bounds }
          init=[randint(0,dim_x),randint(0,dim_y),randint(0,255),randint(0,255),randint(0,255)]*pixel_count
          attack_result = basinhopping(predict_fn,init,niter=maxiter,T=T)
        
        
        # Calculate some useful statistics to return from this function
        attack_image = helper.perturb_image(attack_result.x, self.x_test[img_id])[0]
        prior_probs = model.predict(np.array([self.x_test[img_id]]))[0]
        predicted_probs = model.predict(np.array([attack_image]))[0]
        predicted_class = np.argmax(predicted_probs)
        actual_class = self.y_test[img_id, 0]
        success = predicted_class != actual_class
        cdiff = prior_probs[actual_class] - predicted_probs[actual_class]

        # Show the best attempt at a solution (successful or not)
        if plot:
            helper.plot_image(attack_image, actual_class, self.class_names, predicted_class)

        return [model.name, pixel_count, img_id, actual_class, predicted_class, success, cdiff, prior_probs,
                predicted_probs, attack_result.x, attack_image]
    
    def new_attack(img_id, model, target=None, pixel_count=1, method='DE', 
           maxiter=250, popsize=500,tempertature=5230, T=1.0, verbose=False):
      # Change the target class based on whether this is a targeted attack or not
      print("Executing new attack")
      targeted_attack = target is not None
      target_class = target if targeted_attack else self.y_test[img_id, 0]
      dim_x, dim_y = self.dimensions
      # Format the predict/callback functions for the differential evolution algorithm
      def predict_fn(xs):
          return self.predict_classe(xs, self.x_test[img_id], target_class, 
                                 model, target is None)

      def callback_fn(x, convergence):
          return self.attack_successes(x, self.x_test[img_id], target_class, 
                                model, targeted_attack, verbose)
                                
      if (method=='DE'):
        bounds = [(0,dim_x), (0,dim_y),(0,256), (0,dim_x), (0,dim_y),(0,256),(0,dim_x), (0,dim_y),(0,256)] * pixel_count
        # Population multiplier, in terms of the size of the perturbation vector x
        popmul = max(1, popsize // len(bounds))
        attack_result = differential_evolution(predict_fn, bounds, maxiter=maxiter, popsize=popmul,recombination=1, atol=-1, polish=False)
      elif (method=='DA'):
          bounds = [(0,dim_x), (0,dim_y),(0,256), (0,dim_x), (0,dim_y),(0,256),(0,dim_x), (0,dim_y),(0,256)] * pixel_count
          attack_result =dual_annealing(predict_fn, bounds, maxiter=maxiter, intital_temp=temperature)
      elif (method=='BH'):
        bounds = [(0,dim_x), (0,dim_y),(0,255), (0,dim_x), (0,dim_y),(0,255),(0,dim_x), (0,dim_y),(0,255)] * pixel_count
        minimizer_kwargs = { "method": "L-BFGS-B","bounds":bounds }
        init=[randint(0,dim_x),randint(0,dim_y),randint(0,255),randint(0,dim_x),randint(0,dim_y),randint(0,255),randint(0,dim_x),randint(0,dim_y),randint(0,255)]*pixel_count
        attack_result = basinhopping(predict_fn,init,niter=maxiter,T=T)

      attack_image = helper.perturb_pixels(attack_result.x, self.x_test[img_id])[0]
      prior_probs = model.predict_one(self.x_test[img_id])
      predicted_probs = model.predict_one(attack_image)
      predicted_class = np.argmax(predicted_probs)
      actual_class = self.y_test[img_id, 0]
      success = predicted_class != actual_class
      cdiff = prior_probs[actual_class] - predicted_probs[actual_class]

      # Show the best attempt at a solution (successful or not)
      #helper.plot_image(attack_image, actual_class, class_names, predicted_class)

      return [model.name, pixel_count, img_id, actual_class, predicted_class, success, cdiff, prior_probs, predicted_probs, attack_result.x,attack_image]
    
    def attack_all(self, models, samples=500, pixels=(1, 3, 5), targeted=False,
                   maxiter=75, popsize=400, verbose=False):
        results = []
        for model in models:
            model_results = []
            valid_imgs = self.correct_imgs[self.correct_imgs.name == model.name].img
            img_samples = np.random.choice(valid_imgs, samples)

            for pixel_count in pixels:
                for i, img in enumerate(img_samples):
                    print(model.name, '- image', img, '-', i + 1, '/', len(img_samples))
                    targets = [None] if not targeted else range(10)

                    for target in targets:
                        if targeted:
                            print('Attacking with target', self.class_names[target])
                            if target == self.y_test[img, 0]:
                                continue
                        result = self.attack(img, model, target, pixel_count,method=method,
                                             maxiter=maxiter, temperature=temperature,T=T, popsize=popsize, 
                                             verbose=verbose)
                        model_results.append(result)

            results += model_results
            helper.checkpoint(results, targeted)
        return results
        
    def new_attack_all(self, models, samples=500, pixels=(1),method=method,temperature=temperature,T=T, targeted=False, 
               maxiter=1250, popsize=550, verbose=False):
      results = []
      for model in models:
          model_results = []
          valid_imgs = correct_imgs[self.correct_imgs.name == model.name].img
          img_samples = np.random.choice(valid_imgs, samples, replace=False)

          for i, img_id in enumerate(img_samples):
                  print('\n', model.name, '- image', img_id, '-', i+1, '/', len(img_samples))
                  targets = [None] if not targeted else range(10)

                  for target in targets:
                      if targeted:
                          print('Attacking with target', selfclass_names[target])
                          if target == y_test[img_id, 0]:
                              continue
                      result = self.new_attack(img_id, model, target, pixels, method=method,
                                      maxiter=maxiter,temperature=temperature,T=T, popsize=popsize, 
                                      verbose=verbose)
                      model_results.append(result)

          results += model_results
          helper.checkpoint(results, targeted)
      return results

def predict_attack(self, base, models, df):
  new_stats=[]
  base_name=base[0].name
  df2=helper.attack_stats(df, base, self.network_stats)
  s=float(df2.attack_success_rate)
  m_result = df[df.model == base_name]
  pixels = list(set(m_result.pixels))
  for pixel in pixels:
            p_result = m_result[m_result.pixels == pixel]
            img=[]
            for x in p_result.perturbed:
              img.append(x)
            imgs=np.asarray(img)
            labels=np.array(p_result.true).reshape(len(p_result.true),1)
            for model in models:
                    val_accuracy = np.array(network_stats[network_stats.name == model.name].accuracy)[0]
                    net_stats,_ =helper.evaluate_models([model],imgs,labels)
                    new_stats.append([base_name,model.name, val_accuracy, pixel,s, net_stats[0][1]])
          
  return pd.DataFrame(new_stats, columns=['attack_model', 'evaluation_model', 'accuracy', 'pixels', 'attack_success_rate','after_attack_accuracy'])


if __name__ == '__main__':
    model_defs = {
        'lenet': LeNet,
        'pure_cnn': PureCnn,
        'net_in_net': NetworkInNetwork,
        'resnet': ResNet,
    }

    parser = argparse.ArgumentParser(description='Attack models on Cifar10')
    parser.add_argument('--model', nargs='+', choices=model_defs.keys(), default='resnet',
                        help='Specify one model by name to evaluate.')
    parser.add_argument('--method', nargs='+', default='DE',
                        help='Specify optimization algorithm.')
    parser.add_argument('--pixels', nargs='+', default=(1), type=int,
                        help='The number of pixels that can be perturbed.')
    parser.add_argument('--attack', nargs='+', default='new',
                        help='Specify oif you desire to conduct the old or new attack.')
    parser.add_argument('--maxiter', default=75, type=int,
                        help='The maximum number of iterations in the differential evolution algorithm before giving up and failing the attack.')
    parser.add_argument('--temperature', default=5230, type=int,
                        help='The initial temperature for dual annealing algorithm. Increasing this number requires more computation.')
    parser.add_argument('--popsize', default=400, type=int,
                        help='The number of adversarial images generated each iteration in the differential evolution algorithm. Increasing this number requires more computation.')
    parser.add_argument('--T', default=T, type=int,
                        help='The temperature for basin hopping algoirthm. Increasing this number requires more computation.')
    parser.add_argument('--samples', default=500, type=int,
                        help='The number of image samples to attack. Images are sampled randomly from the dataset.')
    parser.add_argument('--targeted', action='store_true', help='Set this switch to test for targeted attacks.')
    parser.add_argument('--save', default='networks/results/results.pkl', help='Save location for the results (pickle)')
    parser.add_argument('--save_attack', default='networks/results/attack_results.pkl', help='Save location for the results (pickle)')
    parser.add_argument('--verbose', action='store_true', help='Print out additional information every iteration.')

    args = parser.parse_args()

    # Load data and model
    _, test = cifar10.load_data()
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    base = [model_defs[m](load_weights=True) for m in args.model]
    models=[model_defs[m](load_weights=True) for m in model_defs.keys()]
    attack_type=args.attack
    attacker = PixelAttacker(base,models, test, class_names)

    print('Starting attack')
    if (attack_type=='old'):
      results = attacker.attack_all(base, samples=args.samples, pixels=args.pixels, method=args.method,temperature=args.temperature,T=args.T, targeted=args.targeted,
                                    maxiter=args.maxiter, popsize=args.popsize, verbose=args.verbose)
    elif(attack_type=='new'):
      results = attacker.new_attack_all(base, samples=args.samples, pixels=args.pixels, method=args.method,temperature=args.temperature,T=args.T, targeted=args.targeted,
                                    maxiter=args.maxiter, popsize=args.popsize, verbose=args.verbose)
    columns = ['model', 'pixels', 'image', 'true', 'predicted', 'success', 'cdiff', 'prior_probs', 'predicted_probs',
               'perturbation','perturbed']
    results_table = pd.DataFrame(results, columns=columns)

    print(results_table[['model', 'pixels', 'image', 'true', 'predicted', 'success']])

    print('Saving to', args.save)
    with open(args.save, 'wb') as file:
        pickle.dump(results, file)
    
    attack_prediction=attacker.predict_attack(args.model,results_table)
    print(attack_prediction)
    
    print('Saving to', args.save_attack)
    with open(args.save_attack, 'wb') as file:
        pickle.dump(results, file)
