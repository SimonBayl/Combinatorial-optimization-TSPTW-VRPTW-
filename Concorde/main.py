import os
from Data_loader.load_data import lecteur
from tsp_concorde import main_solve

def main():
    path_file = os.getcwd() + '/Sujet-Données/Instance.xlsx'
    path_file = ""
    dist = lecteur(path_file)

    main_solve(distance_matrix=dist,
               path_concorde=os.getcwd() + "/Concorde_exe")


if __name__=="__main__": # la fonction main est lancée avec cette expression particulière
    main()
