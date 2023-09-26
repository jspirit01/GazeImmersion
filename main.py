
from src.population_loocv import population_model
from src.population_loocv_traintest import population_model_traintest
from src.individual_loocv import individual_model
from src.individual_loocv_traintest import individual_model_traintest
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description='Per-Session experiment')
    parser.add_argument('--fs', default=False, help='Run feature selection', type=bool)
    parser.add_argument('--load_fs', default=True, help='=use exsiting feature selection list', type=bool)
    parser.add_argument('--print_test', default=True, help='use test data', type=bool)
    parser.add_argument('--data', help='dataset path', type=str)
    parser.add_argument('--load_pretrained_model', default=False, help='use pretrained model', type=str)
    parser.add_argument('--exp', default='population_allfeatures', required=True, help='define experiment name to save log', type=str)
    parser.add_argument('--seed', default=42, help='random seed', type=int)
    args = parser.parse_args() 

    os.makedirs(f'./results/{args.exp}/', exist_ok=True)

    
    population_model(
        data_path=args.data,
        fs = args.fs,
        load_fs = args.load_fs,
        print_test = args.print_test,
        load_pretrained_model = args.load_pretrained_model,
        save_dir=f'./results/{args.exp}',
        log_path=f'./results/{args.exp}/run.log',
        seed=args.seed)
    
    ''' for debug - population
        train / valid / test '''
    # population_model(
    #     'dataset/exp1+2_usernorm2_population.csv',
    #     fs = False,
    #     load_fs = False,
    #     print_test = True,
    #     load_pretrained_model = False,
    #     save_dir=f'./results/{args.exp}',
    #     log_path=f'./results/{args.exp}/run.log',
    #     seed=42)
    
    ''' for debug - population
        train / test '''
    # population_model_traintest(
    #     'dataset/exp1+2_usernorm2_population.csv',
    #     fs = True,
    #     load_fs = True,
    #     print_test = True,
    #     load_pretrained_model = False,
    #     save_dir=f'./results/{args.exp}',
    #     log_path=f'./results/{args.exp}/run.log',
    #     seed=42)
    

    ''' for debug - individual
        train / valid / test '''
    # individual_model(
    #     'dataset/exp1+2_usernorm2_individual.csv',
    #     fs = True,
    #     load_fs = True,
    #     print_test = True,
    #     load_pretrained_model = False,
    #     save_dir=f'./results/{args.exp}',
    #     log_path=f'./results/{args.exp}/run.log',
    #     seed=42)
    
    # ''' for debug - individual
    #     train / test '''
    # individual_model_traintest(
    #     'dataset/exp1+2_usernorm2_individual.csv',
    #     fs = False,
    #     load_fs = True,
    #     print_test = True,
    #     load_pretrained_model = False,
    #     save_dir=f'./results/{args.exp}',
    #     log_path=f'./results/{args.exp}/run.log',
    #     seed=42)
      
if __name__ == '__main__':
    main()
