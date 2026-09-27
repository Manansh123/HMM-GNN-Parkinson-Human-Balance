To create the '.arff' files for the experiments involving one single value of the
time-shift KAPPA (e.g., KAPPA = 1, 5, 10) as it is done [1], you must compile then
run the program 'MakeArffs.c' located in the three following subdirectories:
  - "./Shift_1/"
  - "./Shift_5/"
  - "./Shift_10/"

The compilation command line of that program is:

   gcc -Wall -o MakeArffs.exe MakeArffs.c

After the compilation, the execution command line:

   MakeArffs.exe InfoFile VectorFile NbComponents
   
Where:   
      1- 'InfoFile' is a file containing insights  (id, age-group and mini-BEST total score) related 
	      to each subject in the dataset proposed in [2]. The file 'BDSinfoSmallClean.csv' was used in [1].
      2- 'VectorFile' is the file containing the final vectors of all the subjects in the 
	      dataset proposed in [2]. The file 'Vectors.txt' was used in [1].
      3- 'NbComponents' is the number of components of each final vector. 
	      The value 52 was used in [1].
		  
Thus, a typical example of execution command line is: 
   
   MakeArffs.exe BDSinfoSmallClean.csv Vectors.txt 52

This command produces as output the files 'Age.arff' and 'BEST.arff' containing the corresponding final vectors.
These files were respectivelly renamed as 'Age$N$_shift_$KAPPA$.arff'  and 'BEST$N$_shift_$KAPPA$.arff'  
then copied in one of the two subdirectories "../Arff_Files/Age$N$/" and "../Arff_Files/BEST$N$/", where:
    1- N is the number of samples of the interval [0,100]. The value (N=10) was used in [1].
	2- KAPPA is the time-shift. The values (KAPPA = 1, 5, 10) were used in [1].
________________________________________________________________________________________________   

To create the '.arff' files for the experiments involving several values of the
time-shift KAPPA (e.g., KAPPA = {1,5},{1,10},{5,10},{1,5,10}) as it is done [1], you must compile then
run the program 'MergeArffs.c' located in the eight following subdirectories:
  - "./Shift_1_5/Age10/"
  - "./Shift_1_5/BEST10/"
  - "./Shift_1_10/Age10/"
  - "./Shift_1_10/BEST10/"
  - "./Shift_5_10/Age10/"
  - "./Shift_5_10/BEST10/"
  - "./Shift_1_5_10/Age10/"
  - "./Shift_1_5_10/BEST10/"

The compilation command line of that program is:

   gcc -Wall -o MergeArffs.exe MergeArffs.c

After the compilation, the execution command line:

   MergeArffs.exe File1 File2 ResultFile
   
Where:   
      1- 'File1' and 'File2' are the two '.arff' files that must be merged. 
	      These two files are already available in each corresponding subdirectory.
      2- 'ResultFile' is the resulting '.arff' file. 
		  
Thus, a typical example of execution command line in the 
subdirectory "./Shift_1_5/Age10/" is: 
   
   MergeArffs.exe Age10_shift_1.arff Age10_shift_5.arff Age10_shift_1_5.arff

The resulting output file was renamed accordingly and copied in the suitable
subdirectories of the directories  "../Arff_Files/Age$N$/" and "../Arff_Files/BEST$N$/", where 
N is the number of samples of the interval [0,100]. The value (N=10) was used in [1].
________________________________________________________________________________________________   

[1] Denkeng, A. T., Mourad, A. M., Iloga, S., Mba, R. M.,Baazaoui, H., Ndié, T. D., & Romain, O. (2025).
"Efficient Characterization Of The Human BalanceUsing HMMS". 
IEEE Access. Vol. 13, pp. 183456 - 183479, doi: 10.1109/ACCESS.2025.3622375.

[2] Santos, D.A., Duarte, M.
"A public data set of human balance evaluations."
PeerJ 4, 2648 (2016)